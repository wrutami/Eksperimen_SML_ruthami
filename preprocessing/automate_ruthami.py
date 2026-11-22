"""Otomatisasi preprocessing dataset Bank Marketing.

Langkahnya sama dengan notebook Eksperimen_ruthami.ipynb:
hapus duplikat, buang `duration` (data leakage), fitur `previously_contacted` dari `pdays`,
isi `unknown` pada job/education dengan modus, binning umur, encoding biner dan target,
split stratified 80/20, capping outlier IQR (batas dari data latih), lalu one-hot encoding
dan standardisasi.

Pemakaian:
    python automate_ruthami.py --input ../bank_marketing_raw/bank-full.csv --output bank_marketing_preprocessing
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
NUMERIC_FEATURES = ["age", "balance", "day", "campaign", "pdays", "previous"]
BINARY_FEATURES = ["default", "housing", "loan", "previously_contacted"]
CATEGORICAL_FEATURES = ["job", "marital", "education", "contact", "month", "poutcome", "age_group"]
CAP_COLUMNS = ["balance", "campaign", "pdays", "previous"]
AGE_BINS = [0, 25, 35, 45, 60, 120]
AGE_LABELS = ["<=25", "26-35", "36-45", "46-60", ">60"]


def load_data(path):
    """Baca bank-full.csv (pemisah ';')."""
    return pd.read_csv(path, sep=";")


def clean_data(df):
    """Pembersihan dan rekayasa fitur yang tidak bergantung pada pembagian data."""
    data = df.drop_duplicates().drop(columns=["duration"])

    data["previously_contacted"] = (data["pdays"] != -1).astype(int)
    data["pdays"] = data["pdays"].where(data["pdays"] != -1, 0)

    for col in ["job", "education"]:
        mode = data.loc[data[col] != "unknown", col].mode()[0]
        data[col] = data[col].replace("unknown", mode)

    data["age_group"] = pd.cut(data["age"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)

    for col in ["default", "housing", "loan"]:
        data[col] = (data[col] == "yes").astype(int)
    data["y"] = (data["y"] == "yes").astype(int)
    return data


def cap_outliers(X_train, X_test):
    """Capping IQR dengan batas yang dihitung dari data latih saja."""
    X_train, X_test = X_train.copy(), X_test.copy()
    bounds = {}
    for col in CAP_COLUMNS:
        q1, q3 = X_train[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr) if iqr > 0 else (X_train[col].min(), X_train[col].quantile(0.99))
        X_train[col] = X_train[col].clip(*bounds[col])
        X_test[col] = X_test[col].clip(*bounds[col])
    return X_train, X_test, bounds


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("bin", "passthrough", BINARY_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


def preprocess(df):
    """Kembalikan (train_processed, test_processed) yang siap dilatih, kolom terakhir = y."""
    data = clean_data(df)
    X, y = data.drop(columns=["y"]), data["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    X_train, X_test, _ = cap_outliers(X_train, X_test)

    preprocessor = build_preprocessor()
    train_arr = preprocessor.fit_transform(X_train)
    test_arr = preprocessor.transform(X_test)
    columns = [name.split("__", 1)[1] for name in preprocessor.get_feature_names_out()]

    train = pd.DataFrame(train_arr, columns=columns)
    test = pd.DataFrame(test_arr, columns=columns)
    train["y"] = y_train.to_numpy()
    test["y"] = y_test.to_numpy()
    return train, test


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Preprocessing otomatis dataset Bank Marketing")
    parser.add_argument("--input", default=str(here.parent / "bank_marketing_raw" / "bank-full.csv"))
    parser.add_argument("--output", default=str(here / "bank_marketing_preprocessing"))
    args = parser.parse_args()

    train, test = preprocess(load_data(args.input))
    assert not train.isna().any().any() and not test.isna().any().any(), "masih ada nilai kosong"

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    print(f"train: {train.shape} | test: {test.shape} | proporsi y train {train['y'].mean():.4f} -> {out}")


if __name__ == "__main__":
    main()
