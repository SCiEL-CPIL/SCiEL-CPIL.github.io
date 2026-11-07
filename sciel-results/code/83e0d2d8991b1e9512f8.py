"""Strong symbolic regression with diverse model portfolio and selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares, curve_fit
from itertools import product

_START_TIME = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0


def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)


def _try_linear_model(x, y, feature_fn, max_iter_check=None):
    """Fit y = X @ w where X = feature_fn(x). Returns (mse, predict_fn) or None."""
    try:
        X = feature_fn(x)
        if np.any(~np.isfinite(X)):
            return None
        # Use lstsq for stability
        w, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ w
        if np.any(~np.isfinite(pred)):
            return None
        mse = np.mean((pred - y) ** 2)
        if not np.isfinite(mse):
            return None
        return (mse, w, feature_fn)
    except Exception:
        return None


def _fit_nonlinear(x, y, model_fn, p0, bounds=(-np.inf, np.inf)):
    """Fit nonlinear model. Returns (mse, params) or None."""
    try:
        def residuals(params):
            try:
                pred = model_fn(x, *params)
                if np.any(~np.isfinite(pred)):
                    return np.full_like(y, 1e10)
                return pred - y
            except Exception:
                return np.full_like(y, 1e10)
        
        result = least_squares(residuals, p0, bounds=bounds, max_nfev=500)
        if result.success or result.cost < 1e10:
            pred = model_fn(x, *result.x)
            if np.any(~np.isfinite(pred)):
                return None
            mse = np.mean((pred - y) ** 2)
            if np.isfinite(mse):
                return (mse, result.x)
    except Exception:
        pass
    return None


def fit(X_train, y_train):
    global _START_TIME
    _START_TIME = time.time()
    
    x = X_train[:, 0].copy()
    y = y_train.copy()
    n = len(x)
    
    # Normalize x to [-1, 1] for better conditioning
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    if x_range < 1e-10:
        x_norm = np.zeros_like(x)
    else:
        x_norm = (x - x_min) / x_range * 2 - 1  # map to [-1, 1]
    
    # y stats
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-10:
        # Constant function
        def predict(X):
            return np.full(X.shape[0], y_mean)
        return predict
    
    y_norm = (y - y_mean) / y_std
    
    best_mse = np.inf
    best_predict_fn = None
    
    # ---- Try polynomial models ----
    for deg in range(1, 15):
        def poly_features(xx, d=deg):
            return np.vander(xx, N=d+1, increasing=True)
        result = _try_linear_model(x_norm, y_norm, poly_features)
        if result and result[0] < best_mse:
            best_mse = result[0]
            w = result[1]
            def predict_poly(X, w=w):
                xx = (X[:, 0] - x_min) / x_range * 2 - 1
                Xf = np.vander(xx, N=deg+1, increasing=True)
                return Xf @ w * y_std + y_mean
            best_predict_fn = predict_poly
    
    # ---- Try trigonometric models ----
    # y = a*sin(b*x + c) + d
    for b_val in [0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10]:
        def trig_model(xx, a, c, d, b=b_val):
            return a * np.sin(b * xx + c) + d
        # Initial guess
        p0 = [y_std, 0, y_mean]
        result = _fit_nonlinear(x_norm, y_norm, trig_model, p0)
        if result and result[0] < best_mse:
            best_mse = result[0]
            a, c, d = result[1]
            b = b_val
            def predict_trig(X, a=a, c=c, d=d, b=b):
                xx = (X[:, 0] - x_min) / x_range * 2 - 1
                return a * np.sin(b * xx + c) + d
            best_predict_fn = predict_trig
    
    # y = a*cos(b*x + c) + d
    for b_val in [0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10]:
        def cos_model(xx, a, c, d, b=b_val):
            return a * np.cos(b * xx + c) + d
        p0 = [y_std, 0, y_mean]
        result = _fit_nonlinear(x_norm, y_norm, cos_model, p0)
        if result and result[0] < best_mse:
            best_mse = result[0]
            a, c, d = result[1]
            b = b_val
            def predict_cos(X, a=a, c=c, d=d, b=b):
                xx = (X[:, 0] - x_min) / x_range * 2 - 1
                return a * np.cos(b * xx + c) + d
            best_predict_fn = predict_cos
    
    # Combined sin+cos: y = a*sin(bx) + c*cos(bx) + d (linear in a,c,d for fixed b)
    for b_val in [0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10]:
        def sincos_features(xx, b=b_val):
            return np.column_stack([np.ones_like(xx), np.sin(b*xx), np.cos(b*xx)])
        result = _try_linear_model(x_norm, y_norm, sincos_features)
        if result and result[0] < best_mse:
            best_mse = result[0]
            w = result[1]
            b = b_val
            def predict_sincos(X, w=w, b=b):
                xx = (X[:, 0] - x_min) / x_range * 2 - 1
                Xf = np.column_stack([np.ones_like(xx), np.sin(b*xx), np.cos(b*xx)])
                return Xf @ w
            best_predict_fn = predict_sincos
    
    # ---- Exponential models ----
    # y = a*exp(b*x) + c
    for b_val in [-2, -1, -0.5, 0.5, 1, 2]:
        def exp_model(xx, a, c, b=b_val):
            return a * np.exp(b * xx) + c
        p0 = [y_std, y_mean, b_val]
        result = _fit_nonlinear(x_norm, y_norm, exp_model, p0)
        if result and result[0] < best_mse:
            best_mse = result[0]
            a, c = result[1][0], result[1][1]
            b = b_val
            def predict_exp(X, a=a, c=c, b=b):
                xx = (X[:, 0] - x_min) / x_range * 2 - 1
                return a * np.exp(b * xx) + c
            best_predict_fn = predict_exp
    
    # y = a*exp(b*x) + c*exp(d*x)
    for b_val, d_val in [(1, 2), (1, 3), (2, 3), (1, 4), (0.5, 1), (1, 1.5), (2, 4), (0.5, 2)]:
        def exp2_features(xx, b=b_val, d=d_val):
            return np.column_stack([np.ones_like(xx), np.exp(b*xx), np.exp(d*xx)])
        result = _try_linear_model(x_norm, y_norm, exp2_features)
        if result and result[0] < best_mse:
            best_mse = result[0]
            w = result[1]
            b, d = b_val, d_val
            def predict_exp2(X, w=w, b=b, d=d):
                xx = (X[:, 0] - x_min) / x_range * 2 - 1
                Xf = np.column_stack([np.ones_like(xx), np.exp(b*xx), np.exp(d*xx)])
                return Xf @ w
            best_predict_fn = predict_exp2
    
    # ---- Rational models ----
    # y = (a*x + b) / (c*x + d)
    # This is nonlinear but we can try
    def rational_model(xx, a, b, c, d):
        denom = c * xx + d
        if np.any(np.abs(denom) < 1e-10):
            return np.full_like(xx, 1e10)
        return (a * xx + b) / denom
    
    p0_rat = [y_std, y_mean, 0.1, 1.0]
    result = _fit_nonlinear(x_norm, y_norm, rational_model, p0_rat)
    if result and result[0] < best_mse:
        best_mse = result[0]
        a, b, c, d = result[1]
        def predict_rat(X, a=a, b=b, c=c, d=d):
            xx = (X[:, 0] - x_min) / x_range * 2 - 1
            denom = c * xx + d
            safe_denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
            return (a * xx + b) / safe_denom
        best_predict_fn = predict_rat
    
    # ---- Power law: y = a * x^b + c ----
    for b_val in [-2, -1, -0.5, 0.5, 1, 2, 3]:
        def power_model(xx, a, c, b=b_val):
            # Handle sign issues
            base = np.where(xx > 0, xx, np.where(xx < 0, -xx, 1e-10))
            if b_val == int(b_val):
                result_base = xx ** int(b_val)
            else:
                result_base = np.sign(xx) * (np.abs(xx) ** b_val)
            return a * result_base + c
        p0 = [y_std, y_mean]
        result = _fit_nonlinear(x_norm, y_norm, power_model, p0)
        if result and result[0] < best_mse:
            best_mse = result[0]
            a, c = result[1]
            b = b_val
            def predict_pow(X, a=a, c=c, b=b):
                xx = (X[:, 0] - x_min) / x_range * 2 - 1
                if b == int(b):
                    base = xx ** int(b)
                else:
                    base = np.sign(xx) * (np.abs(xx) ** b)
                return a * base + c
            best_predict_fn = predict_pow
    
    # ---- Gaussian: y = a * exp(-(x-c)^2 / (2*s^2)) + d ----
    for s_val in [0.2, 0.5, 1.0, 2.0]:
        def gauss_model(xx, a, c, d, s=s_val):
            return a * np.exp(-((xx - c) ** 2) / (2 * s ** 2)) + d
        p0 = [y_std, 0, y_mean]
        result = _fit_nonlinear(x_norm, y_norm, gauss_model, p0)
        if result and result[0] < best_mse:
            best_mse = result[0]
            a, c, d = result[1]
            s = s_val
            def predict_gauss(X, a=a, c=c, d=d, s=s):
                xx = (X[:, 0] - x_min) / x_range * 2 - 1
                return a * np.exp(-((xx - c) ** 2) / (2 * s ** 2)) + d
            best_predict_fn = predict_gauss
    
    # ---- Sigmoid: y = a / (1 + exp(-b*(x-c))) + d ----
    for b_val in [1, 2, 5, 10]:
        def sigmoid_model(xx, a, c, d, b=b_val):
            return a / (1 + np.exp(-b * (xx - c))) + d
        p0 = [y_std, 0, y_mean]
        result = _fit_nonlinear(x_norm, y_norm, sigmoid_model, p0)
        if result and result[0] < best_mse:
            best_mse = result[0]
            a, c, d = result[1]
            b = b_val
            def predict_sig(X, a=a, c=c, d=d, b=b):
                xx = (X[:, 0] - x_min) / x_range * 2 - 1
                return a / (1 + np.exp(-b * (xx - c))) + d
            best_predict_fn = predict_sig
    
    # ---- tanh: y = a * tanh(b*(x-c)) + d ----
    for b_val in [0.5, 1, 2, 5]:
        def tanh_model(xx, a, c, d, b=b_val):
            return a * np.tanh(b * (xx - c)) + d
        p0 = [y_std, 0, y_mean]
        result = _fit_nonlinear(x_norm, y_norm, tanh_model, p0)
        if result and result[0] < best_mse:
            best_mse = result[0]
            a, c, d = result[1]
            b = b_val
            def predict_tanh(X, a=a, c=c, d=d, b=b):
                xx = (X[:, 0] - x_min) / x_range * 2 - 1
                return a * np.tanh(b * (xx - c)) + d
            best_predict_fn = predict_tanh
    
    # ---- Log models: y = a * log(x + b) + c (for x > 0 domains) ----
    if x_min > 0:
        for b_val in [0.1, 0.5, 1, 2]:
            def log_model(xx, a, c, b=b_val):
                return a * np.log(xx + b) + c
            p0 = [y_std, y_mean]
            result = _fit_nonlinear(x_norm, y_norm, log_model, p0)
            if result and result[0] < best_mse:
                best_mse = result[0]
                a, c = result[1]
                b = b_val
                def predict_log(X, a=a, c=c, b=b):
                    xx = (X[:, 0] - x_min) / x_range * 2 - 1
                    return a * np.log(xx + b) + c
                best_predict_fn = predict_log
    
    # ---- Fallback: if nothing worked well, use high-degree polynomial ----
    if best_mse > 1e-6 or best_predict_fn is None:
        # Use polynomial interpolation (degree n-1 would be perfect but unstable)
        # Use degree min(n-1, 20) with regularization
        deg = min(n - 1, 20)
        Xf = np.vander(x_norm, N=deg+1, increasing=True)
        # Add small regularization
        lam = 1e-4
        A = Xf.T @ Xf + lam * np.eye(deg+1)
        b_vec = Xf.T @ y_norm
        w = np.linalg.solve(A, b_vec)
        pred = Xf @ w
        mse = np.mean((pred - y_norm) ** 2)
        if mse < best_mse or best_predict_fn is None:
            best_mse = mse
            def predict_poly_fallback(X, w=w, deg=deg):
                xx = (X[:, 0] - x_min) / x_range * 2 - 1
                Xf = np.vander(xx, N=deg+1, increasing=True)
                return Xf @ w
            best_predict_fn = predict_poly_fallback
    
    # Final safety: ensure predict returns finite values
    def predict(X):
        try:
            result = best_predict_fn(X)
            result = np.asarray(result, dtype=float)
            # Replace any non-finite with y_mean
            result = np.where(np.isfinite(result), result, y_mean)
            return result
        except Exception:
            return np.full(X.shape[0], y_mean)
    
    return predict
# EVOLVE-BLOCK-END
