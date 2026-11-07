# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.linalg import cho_factor, cho_solve

_TOTAL_DEADLINE = time.time() + 320.0


def _remaining():
    return _TOTAL_DEADLINE - time.time()


def _safe_predict(make_fn):
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


def _fit_poly(X, y, degree):
    try:
        x = X[:, 0]
        coeffs = np.polyfit(x, y, degree)
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            return np.polyval(coeffs, xp)
        return _safe_predict(predict)
    except Exception:
        return None


def _fit_spline_interp(X, y):
    """Interpolating cubic spline."""
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        s = CubicSpline(xs, ys)
        
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


def _fit_pchip(X, y):
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        s = PchipInterpolator(xs, ys)
        
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


def _fit_krr(X, y, length_scale=1.0, noise=1e-8):
    """Kernel Ridge Regression (equivalent to GP inference)."""
    try:
        n, d = X.shape
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sigma
        
        def kernel(X1, X2, ls):
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq_dist = np.sum(diff**2, axis=2) / (ls**2)
            return np.exp(-0.5 * sq_dist)
        
        K = kernel(Xn, Xn, length_scale) + noise * np.eye(n)
        try:
            cf = cho_factor(K)
            alpha = cho_solve(cf, y)
        except:
            K += 1e-4 * np.eye(n)
            cf = cho_factor(K)
            alpha = cho_solve(cf, y)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            Xpn = (Xp - mu) / sigma
            Ks = kernel(Xpn, Xn, length_scale)
            preds = Ks @ alpha
            return preds.flatten()
        return _safe_predict(predict)
    except Exception:
        return None


def _fit_rbf(X, y, num_centers=50, width=0.5):
    try:
        n, d = X.shape
        x = X[:, 0]
        x_min, x_max = x.min(), x.max()
        centers = np.linspace(x_min, x_max, num_centers)
        
        F = np.column_stack([np.exp(-0.5 * ((x - c) / width) ** 2) for c in centers])
        w_coeff, *_ = np.linalg.lstsq(F, y, rcond=None)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            Fp = np.column_stack([np.exp(-0.5 * ((xp - c) / width) ** 2) for c in centers])
            return Fp @ w_coeff
        return _safe_predict(predict)
    except Exception:
        return None


def _try_method(method_fn, X, y):
    try:
        predict = method_fn(X, y)
        if predict is None:
            return None, np.inf
        pred = predict(X)
        pred = np.asarray(pred, dtype=np.float64)
        if not np.all(np.isfinite(pred)):
            return None, np.inf
        mse = np.mean((pred - y) ** 2)
        if not np.isfinite(mse):
            return None, np.inf
        return predict, mse
    except Exception:
        return None, np.inf


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n, d = X_train.shape

    best_predict = None
    best_mse = np.inf

    # Baseline: constant
    const = np.mean(y_train)
    const_mse = np.mean((y_train - const) ** 2)
    best_mse = const_mse
    best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    # List of (name, method_fn)
    # We prioritize fast, strong interpolators and kernels
    candidates = []
    
    # 1. Interpolators (Exact fit for smooth noiseless data)
    candidates.append(("pchip", lambda X, y: _fit_pchip(X, y)))
    candidates.append(("spline_interp", lambda X, y: _fit_spline_interp(X, y)))
    
    # 2. Polynomials (Good for low complexity)
    for deg in [1, 2, 3, 4, 5, 6]:
        candidates.append(("poly_%d" % deg, lambda X, y, d=deg: _fit_poly(X, y, d)))
        
    # 3. Kernel Ridge Regression (Very strong for 1D smooth)
    # Estimate typical length scale from data range
    x_range = X_train[:, 0].max() - X_train[:, 0].min()
    if x_range > 0:
        base_ls = x_range / 10.0
    else:
        base_ls = 1.0
        
    for mult in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        ls = base_ls * mult
        candidates.append(("krr_ls_%.2f" % ls, lambda X, y, l=ls: _fit_krr(X, y, l)))
        
    # 4. RBF (Fixed centers, varied width)
    for w in [0.1, 0.2, 0.5, 1.0, 2.0]:
        candidates.append(("rbf_w_%.1f" % w, lambda X, y, w=w: _fit_rbf(X, y, 50, w)))

    # Evaluate candidates on training data
    # Since data is noiseless, low training MSE is a good indicator of success
    # We prefer simpler models if MSE is similar, but here we just take the best MSE
    # to be safe against overfitting of complex models to noise (which is 0)
    
    results = []
    for name, method in candidates:
        if _remaining() < 5.0:
            break
        predict, mse = _try_method(method, X_train, y_train)
        if predict is not None and np.isfinite(mse):
            results.append((name, predict, mse))
            
    # Sort by MSE
    results.sort(key=lambda x: x[2])
    
    # Select the best one
    if results:
        best_name, best_predict_candidate, best_mse_candidate = results[0]
        if best_mse_candidate < best_mse:
            best_mse = best_mse_candidate
            best_predict = best_predict_candidate

    if best_predict is None:
        best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    return best_predict
# EVOLVE-BLOCK-END
