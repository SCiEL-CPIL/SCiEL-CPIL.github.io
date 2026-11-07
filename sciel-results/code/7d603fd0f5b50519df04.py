"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product


def fit(X_train, y_train):
    start = time.time()
    deadline = start + 280.0  # 50s safety margin before 330s kill

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).flatten()
    n, d = X.shape

    # 80/20 train/val split for model selection
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)
    n_val = max(1, n // 5)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    y_var = np.var(y)

    best_score = np.inf
    best_w = None
    best_feat_fn = None

    def _eval_model(F_tr, F_val, y_tr, y_val):
        """Return normalized MSE on validation, or inf on failure."""
        try:
            w, *_ = np.linalg.lstsq(F_tr, y_tr, rcond=None)
            pred_val = F_val @ w
            mse = np.mean((pred_val - y_val) ** 2)
            return mse / y_var if y_var > 1e-12 else mse
        except Exception:
            return np.inf

    # --- Feature builder helpers ---
    def poly_features(X, max_deg=4):
        """All monomials up to max_deg."""
        cols = [np.ones(len(X))]
        for deg in range(1, max_deg + 1):
            for combo in product(range(d), repeat=deg):
                if sorted(combo) != combo:
                    continue
                term = np.ones(len(X))
                for j in combo:
                    term = term * X[:, j]
                cols.append(term)
        return np.column_stack(cols)

    def trig_features(X):
        """Sin and cos of each input and pairwise products."""
        cols = [np.ones(len(X))]
        for j in range(d):
            cols.append(np.sin(X[:, j]))
            cols.append(np.cos(X[:, j]))
        for i in range(d):
            for j in range(i, d):
                cols.append(X[:, i] * np.sin(X[:, j]))
                cols.append(X[:, i] * np.cos(X[:, j]))
        return np.column_stack(cols)

    def exp_log_features(X):
        """Exp, log, and combinations."""
        cols = [np.ones(len(X))]
        for j in range(d):
            cols.append(np.exp(X[:, j] / 5.0))  # scaled to avoid overflow
            cols.append(np.log1p(X[:, j]))
            cols.append(X[:, j] * np.log1p(X[:, j]))
        for i in range(d):
            for j in range(i, d):
                cols.append(X[:, i] * X[:, j])
                cols.append(np.log1p(X[:, i]) * X[:, j])
        return np.column_stack(cols)

    def rational_features(X):
        """Rational-like features: x/(x+1), x^2/(x^2+1), etc."""
        cols = [np.ones(len(X))]
        for j in range(d):
            cols.append(X[:, j] / (X[:, j] + 1.0))
            cols.append(X[:, j] ** 2 / (X[:, j] ** 2 + 1.0))
            cols.append(1.0 / (1.0 + np.exp(-X[:, j])))
        for i in range(d):
            for j in range(i, d):
                cols.append(X[:, i] * X[:, j] / (X[:, i] + X[:, j] + 1.0))
        return np.column_stack(cols)

    def mixed_features(X):
        """Combine polynomial + trig + exp_log + rational."""
        F = np.column_stack([
            poly_features(X, max_deg=3),
            trig_features(X),
            exp_log_features(X),
            rational_features(X),
        ])
        return F

    # --- Try models in order of increasing complexity ---
    candidates = [
        ("poly2", lambda X: poly_features(X, 2)),
        ("poly3", lambda X: poly_features(X, 3)),
        ("poly4", lambda X: poly_features(X, 4)),
        ("trig", trig_features),
        ("exp_log", exp_log_features),
        ("rational", rational_features),
        ("mixed", mixed_features),
    ]

    for name, feat_fn in candidates:
        if time.time() > deadline - 5.0:
            break
        try:
            F_tr = feat_fn(X_tr)
            F_val = feat_fn(X_val)
            score = _eval_model(F_tr, F_val, y_tr, y_val)
            if score < best_score:
                best_score = score
                # Fit on full training data for final prediction
                F_full = feat_fn(X)
                best_w, *_ = np.linalg.lstsq(F_full, y, rcond=None)
                best_feat_fn = feat_fn
        except Exception:
            continue

    # If no model worked, fall back to simple linear
    if best_w is None:
        F_full = np.column_stack([np.ones(n)] + [X[:, j] for j in range(d)])
        best_w, *_ = np.linalg.lstsq(F_full, y, rcond=None)
        best_feat_fn = lambda X: np.column_stack([np.ones(len(X))] + [X[:, j] for j in range(X.shape[1])])

    feat_fn = best_feat_fn
    w = best_w

    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        F = feat_fn(X_test)
        return F @ w

    return predict
# EVOLVE-BLOCK-END
