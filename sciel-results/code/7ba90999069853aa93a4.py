# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.linalg import solve

_TOTAL_DEADLINE = time.time() + 315.0


def _remaining():
    return _TOTAL_DEADLINE - time.time()


def _safe_predict(make_fn):
    """Wrap a prediction function to always return finite values."""
    def predict(X):
        try:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            y = make_fn(X)
            y = np.asarray(y, dtype=np.float64)
            y = np.nan_to_num(y, nan=0.0, posinf=1e12, neginf=-1e12)
            return y
        except Exception:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return np.zeros(X.shape[0])
    return predict


def _fit_pchip(X, y):
    """Fit a PCHIP interpolator. Handles duplicates by averaging."""
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        
        # Handle duplicates: average y values for same x
        unique_xs, unique_indices, counts = np.unique(xs, return_inverse=True, return_counts=True)
        if len(unique_xs) < len(xs):
            ys_sum = np.zeros_like(unique_xs)
            np.add.at(ys_sum, unique_indices, ys)
            ys = ys_sum / counts
        
        if len(xs) < 4:
            return None

        s = PchipInterpolator(unique_xs, ys)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            yp = s(xp)
            return np.nan_to_num(yp, nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None


def _fit_cubic_spline(X, y):
    """Fit a CubicSpline interpolator. Handles duplicates by averaging."""
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        
        # Handle duplicates: average y values for same x
        unique_xs, unique_indices, counts = np.unique(xs, return_inverse=True, return_counts=True)
        if len(unique_xs) < len(xs):
            ys_sum = np.zeros_like(unique_xs)
            np.add.at(ys_sum, unique_indices, ys)
            ys = ys_sum / counts
        
        if len(xs) < 4:
            return None

        s = CubicSpline(unique_xs, ys)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            yp = s(xp)
            return np.nan_to_num(yp, nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None


def _poly_features(X, degree):
    """Polynomial features up to given degree in d=1."""
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        for p in range(1, degree + 1):
            cols.append(X[:, j] ** p)
    return np.column_stack(cols)


def _trig_features(X):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        for k in range(1, 10):
            cols.append(np.sin(k * x))
            cols.append(np.cos(k * x))
        cols.append(np.tanh(x))
    return np.column_stack(cols)


def _exp_features(X):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        # Safe exp/log
        cols.append(np.exp(np.clip(x, -20, 20)))
        cols.append(np.exp(-np.clip(x, -20, 20)))
        cols.append(np.sqrt(np.abs(x) + 1e-9))
        cols.append(np.log(np.abs(x) + 1e-9))
        cols.append(1.0 / (np.abs(x) + 1e-6))
        cols.append(np.exp(-x**2))
    return np.column_stack(cols)


def _mixed_features(X):
    """Combine polynomial, trig, and exp features."""
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        for p in range(1, 8):
            cols.append(x ** p)
        for k in range(1, 8):
            cols.append(np.sin(k * x))
            cols.append(np.cos(k * x))
        cols.append(np.tanh(x))
        cols.append(np.exp(-x ** 2))
        cols.append(np.exp(np.clip(x, -20, 20)))
        cols.append(np.exp(-np.clip(x, -20, 20)))
        cols.append(np.log(np.abs(x) + 1e-9))
        cols.append(1.0 / (np.abs(x) + 1e-6))
        cols.append(np.sqrt(np.abs(x) + 1e-9))
    return np.column_stack(cols)


def _fit_robust_linear(F, y, ridge_alpha=1e-6):
    """Solve least squares with ridge regularization using SVD for stability."""
    try:
        n, p = F.shape
        # Add ridge term
        # We want to solve (F^T F + alpha I) w = F^T y
        # Using SVD of F is more stable
        U, s, Vt = np.linalg.svd(F, full_matrices=False)
        # Regularize singular values
        s_reg = s / (s**2 + ridge_alpha)
        # w = V^T U^T y * s_reg
        Ut_y = U.T @ y
        w = Vt.T @ (Ut_y * s_reg)
        
        if not np.all(np.isfinite(w)):
            return None
        return w
    except Exception:
        return None


def _try_fit_family(feat_fn, X, y, ridge_alpha=1e-6):
    """Try fitting a linear model on given features."""
    try:
        F = feat_fn(X)
        if not np.all(np.isfinite(F)):
            F = np.nan_to_num(F, nan=0.0, posinf=1e6, neginf=-1e6)
        
        w = _fit_robust_linear(F, y, ridge_alpha)
        if w is None:
            return None, np.inf
        
        pred = F @ w
        if not np.all(np.isfinite(pred)):
            return None, np.inf
        
        rmse = np.sqrt(np.mean((pred - y) ** 2))
        predict = _safe_predict(lambda Xp: feat_fn(Xp) @ w)
        return predict, rmse
    except Exception:
        return None, np.inf


def _fit_kernel_ridge(X, y, length_scale=1.0, alpha=1e-4):
    """Kernel Ridge Regression with RBF kernel."""
    try:
        n, d = X.shape
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sigma
        
        diff = Xn[:, np.newaxis, :] - Xn[np.newaxis, :, :]
        sq_dist = np.sum(diff**2, axis=2) / (length_scale**2)
        K = np.exp(-0.5 * sq_dist) + alpha * np.eye(n)
        
        # Use solve, fallback to lstsq if singular
        try:
            w = solve(K, y)
        except Exception:
            w, *_ = np.linalg.lstsq(K, y, rcond=None)
            
        if not np.all(np.isfinite(w)):
            return None

        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            Xpn = (Xp - mu) / sigma
            diff = Xpn[:, np.newaxis, :] - Xn[np.newaxis, :, :]
            sq_dist = np.sum(diff**2, axis=2) / (length_scale**2)
            Ks = np.exp(-0.5 * sq_dist)
            preds = Ks @ w
            return np.nan_to_num(preds, nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n, d = X_train.shape

    best_predict = None
    best_rmse = np.inf

    # Baseline: constant
    const = np.mean(y_train)
    const_rmse = np.sqrt(np.mean((y_train - const) ** 2))
    best_rmse = const_rmse
    best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    # 1. Try PCHIP Interpolation (Excellent for noiseless 1D)
    if d == 1 and _remaining() > 5.0:
        pchip_pred = _fit_pchip(X_train, y_train)
        if pchip_pred is not None:
            pred = pchip_pred(X_train)
            rmse = np.sqrt(np.mean((pred - y_train) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_predict = pchip_pred

    # 2. Try Cubic Spline
    if d == 1 and _remaining() > 5.0:
        cs_pred = _fit_cubic_spline(X_train, y_train)
        if cs_pred is not None:
            pred = cs_pred(X_train)
            rmse = np.sqrt(np.mean((pred - y_train) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_predict = cs_pred

    # 3. Try Kernel Ridge Regression (Robust for smooth functions)
    if _remaining() > 10.0:
        for ls in [0.5, 1.0, 2.0, 5.0]:
            if _remaining() < 5.0:
                break
            kr_pred = _fit_kernel_ridge(X_train, y_train, length_scale=ls, alpha=1e-4)
            if kr_pred is not None:
                pred = kr_pred(X_train)
                rmse = np.sqrt(np.mean((pred - y_train) ** 2))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_predict = kr_pred

    # 4. Try Polynomial Regression (Low to High Degree)
    for deg in [1, 2, 3, 4, 5, 6, 8, 10]:
        if _remaining() < 5.0:
            break
        predict, rmse = _try_fit_family(lambda Xp, d=deg: _poly_features(Xp, d), X_train, y_train)
        if predict is not None and rmse < best_rmse:
            best_rmse = rmse
            best_predict = predict

    # 5. Try Trigonometric Basis
    if _remaining() > 5.0:
        predict, rmse = _try_fit_family(_trig_features, X_train, y_train)
        if predict is not None and rmse < best_rmse:
            best_rmse = rmse
            best_predict = predict

    # 6. Try Exponential/Log Basis
    if _remaining() > 5.0:
        predict, rmse = _try_fit_family(_exp_features, X_train, y_train)
        if predict is not None and rmse < best_rmse:
            best_rmse = rmse
            best_predict = predict

    # 7. Try Mixed Basis
    if _remaining() > 5.0:
        predict, rmse = _try_fit_family(_mixed_features, X_train, y_train)
        if predict is not None and rmse < best_rmse:
            best_rmse = rmse
            best_predict = predict

    # Final safety check
    if best_predict is None:
        best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    return best_predict
# EVOLVE-BLOCK-END
