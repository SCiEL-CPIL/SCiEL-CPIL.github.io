"""Symbolic regression with rich feature expansion and deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations_with_replacement


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280.0  # 20s safety margin

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).flatten()
    n_samples, n_dims = X.shape

    def make_features(X, degree=3):
        """Generate polynomial features up to given degree."""
        cols = [np.ones(len(X))]
        # Linear
        for j in range(n_dims):
            cols.append(X[:, j])
        # Quadratic and higher
        for d in range(2, degree + 1):
            for combo in combinations_with_replacement(range(n_dims), d):
                feat = np.ones(len(X))
                for idx in combo:
                    feat = feat * X[:, idx]
                cols.append(feat)
        return np.column_stack(cols)

    def make_nonlinear_features(X):
        """Generate nonlinear features: sin, cos, exp, log, tanh."""
        cols = []
        for j in range(n_dims):
            xj = X[:, j]
            cols.append(np.sin(xj))
            cols.append(np.cos(xj))
            cols.append(np.tanh(xj))
            cols.append(np.log1p(xj))  # log(1+x) to handle small values
            cols.append(np.exp(-xj / 5.0))  # damped exponential
        # Products of nonlinear features (pairwise)
        for i in range(n_dims):
            for j in range(i, n_dims):
                cols.append(np.sin(X[:, i]) * X[:, j])
                cols.append(np.cos(X[:, i]) * X[:, j])
                cols.append(X[:, i] * np.tanh(X[:, j]))
        return np.column_stack(cols)

    def ridge_fit(Feat, y, alpha=1.0):
        """Ridge regression."""
        A = Feat.T @ Feat + alpha * np.eye(Feat.shape[1])
        b = Feat.T @ y
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w, *_ = np.linalg.lstsq(Feat, y, rcond=None)
        return w

    def predict_with(Feat, w):
        return Feat @ w

    # Baseline: degree-2 polynomial
    Feat2 = make_features(X, degree=2)
    w2 = ridge_fit(Feat2, y, alpha=1.0)
    pred2 = predict_with(Feat2, w2)
    mse2 = np.mean((pred2 - y) ** 2)

    # Try degree-3 polynomial
    Feat3 = make_features(X, degree=3)
    w3 = ridge_fit(Feat3, y, alpha=10.0)
    pred3 = predict_with(Feat3, w3)
    mse3 = np.mean((pred3 - y) ** 2)

    # Try nonlinear features
    Feat_nl = make_nonlinear_features(X)
    # Combine polynomial degree 2 + nonlinear
    Feat_comb = np.column_stack([Feat2, Feat_nl])
    w_comb = ridge_fit(Feat_comb, y, alpha=1.0)
    pred_comb = predict_with(Feat_comb, w_comb)
    mse_comb = np.mean((pred_comb - y) ** 2)

    # Try just nonlinear + linear
    Feat_nl2 = np.column_stack([np.ones(n_samples), X, Feat_nl])
    w_nl2 = ridge_fit(Feat_nl2, y, alpha=1.0)
    pred_nl2 = predict_with(Feat_nl2, w_nl2)
    mse_nl2 = np.mean((pred_nl2 - y) ** 2)

    # Pick the best so far
    candidates = [
        (mse2, lambda X_test: make_features(X_test, degree=2) @ w2),
        (mse3, lambda X_test: make_features(X_test, degree=3) @ w3),
        (mse_comb, lambda X_test: np.column_stack([make_features(X_test, degree=2), make_nonlinear_features(X_test)]) @ w_comb),
        (mse_nl2, lambda X_test: np.column_stack([np.ones(len(X_test)), X_test, make_nonlinear_features(X_test)]) @ w_nl2),
    ]

    candidates.sort(key=lambda x: x[0])
    best_mse, best_predictor = candidates[0]

    # If we have time, try a few more things
    if time.time() < deadline - 30:
        # Try adding interaction terms with nonlinear
        Feat_extra = np.column_stack([Feat2, Feat_nl, X[:, 0] * np.sin(X[:, 1] if n_dims > 1 else X[:, 0])])
        if n_dims > 1:
            Feat_extra = np.column_stack([Feat2, Feat_nl, X[:, 0] * np.sin(X[:, 1]), X[:, 1] * np.cos(X[:, 0])])
        else:
            Feat_extra = np.column_stack([Feat2, Feat_nl, X[:, 0] * np.sin(X[:, 0])])
        w_extra = ridge_fit(Feat_extra, y, alpha=1.0)
        pred_extra = predict_with(Feat_extra, w_extra)
        mse_extra = np.mean((pred_extra - y) ** 2)
        if mse_extra < best_mse:
            best_mse = mse_extra
            best_predictor = lambda X_test: np.column_stack([
                make_features(X_test, degree=2),
                make_nonlinear_features(X_test),
                X_test[:, 0] * np.sin(X_test[:, 1] if n_dims > 1 else X_test[:, 0])
            ]) @ w_extra

    # Final safety: if best predictor fails, fall back to simple mean
    def safe_predict(X_test):
        try:
            result = best_predictor(X_test)
            if not np.all(np.isfinite(result)):
                return np.full(len(X_test), np.mean(y))
            return result
        except Exception:
            return np.full(len(X_test), np.mean(y))

    return safe_predict
# EVOLVE-BLOCK-END
