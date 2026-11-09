"""Symbolic regression with hybrid trig+poly basis and ridge regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def fit(X_train, y_train):
    t0 = time.time()
    DEADLINE = 320.0  # safety margin before 330s kill
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # Standardize for numerical stability
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Xs = (X - mu) / sd

    def make_features(Xraw):
        Xr = np.asarray(Xraw, dtype=np.float64)
        Xrn = (Xr - mu) / sd
        cols = [np.ones(len(Xr))]
        # Linear terms
        for j in range(d):
            cols.append(Xrn[:, j])
        # Quadratic terms
        for i in range(d):
            for j in range(i, d):
                cols.append(Xrn[:, i] * Xrn[:, j])
        # Cubic terms
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    cols.append(Xrn[:, i] * Xrn[:, j] * Xrn[:, k])
        # Trig terms
        for j in range(d):
            cols.append(np.sin(Xrn[:, j]))
            cols.append(np.cos(Xrn[:, j]))
        # Exp terms
        for j in range(d):
            cols.append(np.exp(Xrn[:, j]))
        return np.column_stack(cols)

    F = make_features(X)
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    def objective(log_lambda):
        lam = np.exp(log_lambda)
        # Ridge regression: min ||F w - y||^2 + lam * ||w||^2
        try:
            w = np.linalg.solve(F.T @ F + lam * np.eye(F.shape[1]), F.T @ y_norm)
            pred = F @ w
            return np.mean((pred - y_norm) ** 2)
        except np.linalg.LinAlgError:
            return 1e10

    # Try a range of lambda values
    best_score = float('inf')
    best_w = None
    for log_lam in np.linspace(-8, 8, 17):
        if time.time() > DEADLINE - 10:
            break
        score = objective(log_lam)
        if score < best_score:
            best_score = score
            best_w = np.linalg.solve(F.T @ F + np.exp(log_lam) * np.eye(F.shape[1]), F.T @ y_norm)

    # Also try unregularized least squares
    try:
        w_ls, *_ = np.linalg.lstsq(F, y_norm, rcond=None)
        score_ls = np.mean((F @ w_ls - y_norm) ** 2)
        if score_ls < best_score:
            best_score = score_ls
            best_w = w_ls
    except np.linalg.LinAlgError:
        pass

    def predict(X):
        if time.time() > DEADLINE:
            # Fallback: return mean
            return np.full(len(X), y_mean)
        F_pred = make_features(X)
        pred_norm = F_pred @ best_w
        return pred_norm * y_std + y_mean

    return predict
# EVOLVE-BLOCK-END
