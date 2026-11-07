"""Symbolic regression over 10 hidden equations: GP + GBDT ensemble with CV selection."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel


def _norm_mse(pred, y):
    y = np.asarray(y, dtype=np.float64).ravel()
    pred = np.asarray(pred, dtype=np.float64).ravel()
    var = np.var(y)
    if var <= 1e-15:
        var = 1.0
    return float(np.mean((pred - y) ** 2) / var)


def _fit_gp(X, y, deadline):
    n = X.shape[0]
    # Standardize
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Xs = (X - mu) / sd

    kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(Xs.shape[1]), nu=2.5) + WhiteKernel(1e-3)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True,
                                  alpha=1e-6, copy_X_train=True, random_state=0)
    gp.fit(Xs, y)

    def predict(Xq):
        Xq = np.asarray(Xq, dtype=np.float64)
        Xqs = (Xq - mu) / sd
        return gp.predict(Xqs)

    return predict


def _fit_gbdt(X, y, deadline):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n = X.shape[0]
    # Adaptive depth / n_estimators based on time budget and problem size
    remaining = max(1.0, deadline - time.time())
    if remaining > 60:
        n_estimators = 400
        depth = 5
    else:
        n_estimators = 200
        depth = 4
    gb = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=depth,
                                   learning_rate=0.05, subsample=0.9,
                                   min_samples_leaf=2, random_state=0)
    gb.fit(X, y)

    def predict(Xq):
        Xq = np.asarray(Xq, dtype=np.float64)
        return gb.predict(Xq)

    return predict


def fit(X_train, y_train):
    start = time.time()
    deadline = start + 275.0  # safety margin before 300s wall-clock limit

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).ravel()
    n, d = X.shape

    # Baseline: constant predictor (mean)
    mean_pred = float(np.mean(y))

    candidates = []  # (predict_fn, cv_score)

    # ---- Gradient Boosted Trees (fast, robust) ----
    try:
        gb_predict = _fit_gbdt(X, y, deadline)
        # quick CV estimate
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        errs = []
        for tr, va in kf.split(X):
            g = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05,
                                          subsample=0.9, min_samples_leaf=2, random_state=0)
            g.fit(X[tr], y[tr])
            errs.append(_norm_mse(g.predict(X[va]), y[va]))
        gb_cv = float(np.mean(errs))
        candidates.append((gb_predict, gb_cv))
    except Exception:
        pass

    # ---- Gaussian Process (flexible, smooth) ----
    if time.time() < deadline - 20.0:
        try:
            gp_predict = _fit_gp(X, y, deadline)
            # quick CV estimate with fewer restarts for speed
            kf = KFold(n_splits=5, shuffle=True, random_state=0)
            errs = []
            for tr, va in kf.split(X):
                mu = X[tr].mean(axis=0)
                sd = X[tr].std(axis=0)
                sd[sd < 1e-8] = 1.0
                Xtr_s = (X[tr] - mu) / sd
                Xva_s = (X[va] - mu) / sd
                kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(d), nu=2.5) + WhiteKernel(1e-3)
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1,
                                              normalize_y=True, alpha=1e-6, copy_X_train=True, random_state=0)
                gp.fit(Xtr_s, y[tr])
                errs.append(_norm_mse(gp.predict(Xva_s), y[va]))
            gp_cv = float(np.mean(errs))
            candidates.append((gp_predict, gp_cv))
        except Exception:
            pass

    # ---- Select best ----
    if not candidates:
        return lambda X: np.full(np.asarray(X).shape[0], mean_pred)

    candidates.sort(key=lambda t: t[1])
    best_pred, best_cv = candidates[0]

    # Safety: ensure we return a finite predictor
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        out = best_pred(X)
        out = np.asarray(out, dtype=np.float64).ravel()
        # Replace any non-finite with mean
        if not np.all(np.isfinite(out)):
            out = np.where(np.isfinite(out), out, mean_pred)
        return out

    return predict
# EVOLVE-BLOCK-END
