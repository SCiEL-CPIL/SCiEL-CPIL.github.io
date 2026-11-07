# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from scipy.interpolate import PchipInterpolator, CubicSpline, make_interp_spline
from numpy.polynomial import chebyshev as C

_TOTAL_DEADLINE = time.time() + 325.0


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


def _deduplicate(x, y):
    """Remove duplicate x values by averaging y."""
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    unique_x, inverse, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
    if len(unique_x) == len(x_sorted):
        return x_sorted, y_sorted
    # Average duplicates
    y_sum = np.zeros(len(unique_x))
    np.add.at(y_sum, inverse, y_sorted)
    y_avg = y_sum / counts
    return unique_x, y_avg


def _fit_pchip(X, y):
    """Fit PCHIP (monotone cubic) interpolator."""
    try:
        x = X[:, 0]
        x, y = _deduplicate(x, y)
        if len(x) < 2:
            return None, np.inf
        pchip = PchipInterpolator(x, y)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            # Clip to training range to avoid extrapolation issues
            xp_clipped = np.clip(xp, x[0], x[-1])
            preds = pchip(xp_clipped)
            return np.nan_to_num(preds, nan=y.mean(), posinf=1e12, neginf=-1e12)
        
        pred_train = predict(X)
        mse = np.mean((pred_train - y) ** 2)
        return _safe_predict(predict), mse
    except Exception:
        return None, np.inf


def _fit_cubic_spline(X, y):
    """Fit natural cubic spline interpolator."""
    try:
        x = X[:, 0]
        x, y = _deduplicate(x, y)
        if len(x) < 3:
            return None, np.inf
        cs = CubicSpline(x, y, bc_type='natural')
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            xp_clipped = np.clip(xp, x[0], x[-1])
            preds = cs(xp_clipped)
            return np.nan_to_num(preds, nan=y.mean(), posinf=1e12, neginf=-1e12)
        
        pred_train = predict(X)
        mse = np.mean((pred_train - y) ** 2)
        return _safe_predict(predict), mse
    except Exception:
        return None, np.inf


def _fit_chebyshev(X, y, max_degree=50):
    """
    Fit using Chebyshev polynomial basis with exact least squares.
    For noiseless data, higher degree should interpolate well.
    """
    try:
        x = X[:, 0]
        x, y = _deduplicate(x, y)
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-12:
            return None, np.inf
        
        # Map to [-1, 1]
        x_norm = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
        
        best_predict = None
        best_mse = np.inf
        best_w = None
        
        for deg in [10, 20, 30, 40, 50, 60, 80, 100]:
            if _remaining() < 2.0:
                break
            if deg >= len(x):
                continue
                
            F = C.chebvander(x_norm, deg)
            
            # Solve least squares
            try:
                w, residuals, rank, sv = np.linalg.lstsq(F, y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            pred = F @ w
            if not np.all(np.isfinite(pred)):
                continue
            
            mse = np.mean((pred - y) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_w = w.copy()
                best_deg = deg
        
        if best_w is not None:
            def predict(Xp):
                Xp = np.asarray(Xp, dtype=np.float64)
                if Xp.ndim == 1:
                    Xp = Xp.reshape(1, -1)
                xp = Xp[:, 0]
                xp_norm = 2.0 * (xp - x_min) / (x_max - x_min) - 1.0
                Fp = C.chebvander(xp_norm, best_deg)
                preds = Fp @ best_w
                return np.nan_to_num(preds, nan=y.mean(), posinf=1e12, neginf=-1e12)
            return _safe_predict(predict), best_mse
        
        return None, np.inf
    except Exception:
        return None, np.inf


def _fit_local_regression(X, y, bandwidths=None):
    """
    Local linear regression (LOESS-like) with adaptive bandwidth.
    For each test point, fit a linear model using weighted nearby training points.
    """
    try:
        x = X[:, 0]
        x, y = _deduplicate(x, y)
        n = len(x)
        
        if bandwidths is None:
            bandwidths = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
        
        x_range = x.max() - x.min()
        if x_range < 1e-12:
            return None, np.inf
        
        best_predict = None
        best_mse = np.inf
        
        for bw_frac in bandwidths:
            if _remaining() < 3.0:
                break
            bw = bw_frac * x_range
            
            def predict(Xp):
                Xp = np.asarray(Xp, dtype=np.float64)
                if Xp.ndim == 1:
                    Xp = Xp.reshape(1, -1)
                xp = Xp[:, 0]
                preds = np.zeros(len(xp))
                
                for i, xi in enumerate(xp):
                    # Gaussian weights
                    dist = np.abs(x - xi) / bw
                    weights = np.exp(-0.5 * dist ** 2)
                    
                    # Ensure at least some weight
                    if weights.max() < 1e-10:
                        # Find nearest point
                        nearest = np.argmin(np.abs(x - xi))
                        preds[i] = y[nearest]
                        continue
                    
                    # Weighted linear regression
                    w_sqrt = np.sqrt(weights)
                    xw = x * w_sqrt
                    yw = y * w_sqrt
                    ww = weights
                    
                    # Design matrix for linear: [1, x]
                    A = np.column_stack([np.ones(n), xw])
                    b = yw
                    
                    try:
                        coef, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        preds[i] = coef[0] + coef[1] * xi
                    except np.linalg.LinAlgError:
                        # Fallback to weighted mean
                        preds[i] = np.sum(weights * y) / np.sum(weights)
                
                return np.nan_to_num(preds, nan=y.mean(), posinf=1e12, neginf=-1e12)
            
            # Evaluate on training data (subsample for speed)
            if n > 50:
                eval_idx = np.random.choice(n, 50, replace=False)
                pred_eval = predict(X[eval_idx])
                mse = np.mean((pred_eval - y[eval_idx]) ** 2)
            else:
                pred_eval = predict(X)
                mse = np.mean((pred_eval - y) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_predict = _safe_predict(predict)
        
        return best_predict, best_mse
    except Exception:
        return None, np.inf


def _fit_rbf_interpolation(X, y, epsilon_range=None):
    """
    Radial Basis Function interpolation with adaptive epsilon.
    For noiseless data, RBF can provide exact interpolation.
    """
    try:
        x = X[:, 0]
        x, y = _deduplicate(x, y)
        n = len(x)
        
        if epsilon_range is None:
            epsilon_range = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0]
        
        x_range = x.max() - x.min()
        if x_range < 1e-12:
            return None, np.inf
        
        best_predict = None
        best_mse = np.inf
        
        for eps in epsilon_range:
            if _remaining() < 3.0:
                break
            
            # Build RBF kernel matrix
            diff = x[:, np.newaxis] - x[np.newaxis, :]
            K = np.exp(-eps ** 2 * diff ** 2)
            K += 1e-10 * np.eye(n)
            
            try:
                w = np.linalg.solve(K, y)
            except np.linalg.LinAlgError:
                continue
            
            def predict(Xp, w=w, eps=eps):
                Xp = np.asarray(Xp, dtype=np.float64)
                if Xp.ndim == 1:
                    Xp = Xp.reshape(1, -1)
                xp = Xp[:, 0]
                diff = xp[:, np.newaxis] - x[np.newaxis, :]
                Ks = np.exp(-eps ** 2 * diff ** 2)
                preds = Ks @ w
                return np.nan_to_num(preds, nan=y.mean(), posinf=1e12, neginf=-1e12)
            
            pred_train = predict(X)
            mse = np.mean((pred_train - y) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_predict = _safe_predict(predict)
        
        return best_predict, best_mse
    except Exception:
        return None, np.inf


def _fit_adaptive_grid(X, y):
    """
    Create an adaptive grid based on data density and interpolate.
    Useful for functions with varying smoothness.
    """
    try:
        x = X[:, 0]
        x, y = _deduplicate(x, y)
        n = len(x)
        
        if n < 5:
            return None, np.inf
        
        # Create a denser grid
        x_min, x_max = x.min(), x.max()
        n_grid = max(100, min(1000, n * 5))
        x_grid = np.linspace(x_min, x_max, n_grid)
        
        # Interpolate to grid
        pchip = PchipInterpolator(x, y)
        y_grid = pchip(x_grid)
        
        # Smooth the grid values with a moving average
        kernel_size = max(3, min(15, n_grid // 50))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones(kernel_size) / kernel_size
        y_smooth = np.convolve(y_grid, kernel, mode='same')
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            xp_clipped = np.clip(xp, x_min, x_max)
            preds = np.interp(xp_clipped, x_grid, y_smooth)
            return np.nan_to_num(preds, nan=y.mean(), posinf=1e12, neginf=-1e12)
        
        pred_train = predict(X)
        mse = np.mean((pred_train - y) ** 2)
        return _safe_predict(predict), mse
    except Exception:
        return None, np.inf


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n, d = X_train.shape

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
        n, d = X_train.shape

    best_predict = None
    best_mse = np.inf

    # Baseline: constant
    const = np.mean(y_train)
    const_mse = np.mean((y_train - const) ** 2)
    best_mse = const_mse
    best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    if d != 1:
        # For multi-dimensional, fall back to kernel ridge
        try:
            mu = X_train.mean(axis=0)
            sigma = X_train.std(axis=0) + 1e-8
            Xn = (X_train - mu) / sigma
            
            for ls in [0.5, 1.0, 2.0]:
                if _remaining() < 5.0:
                    break
                diff = Xn[:, np.newaxis, :] - Xn[np.newaxis, :, :]
                sq_dist = np.sum(diff**2, axis=2) / (ls**2)
                K = np.exp(-0.5 * sq_dist) + 1e-8 * np.eye(n)
                w = np.linalg.solve(K, y_train)
                
                def predict(Xp, w=w, ls=ls):
                    Xp = np.asarray(Xp, dtype=np.float64)
                    if Xp.ndim == 1:
                        Xp = Xp.reshape(1, -1)
                    Xpn = (Xp - mu) / sigma
                    diff = Xpn[:, np.newaxis, :] - Xn[np.newaxis, :, :]
                    sq_dist = np.sum(diff**2, axis=2) / (ls**2)
                    Ks = np.exp(-0.5 * sq_dist)
                    preds = Ks @ w
                    return np.nan_to_num(preds, nan=y_train.mean(), posinf=1e12, neginf=-1e12)
                
                pred_train = predict(X_train)
                mse = np.mean((pred_train - y_train) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_predict = _safe_predict(predict)
        except Exception:
            pass
        return best_predict

    # For 1D: try multiple interpolation methods and pick the best

    # 1. PCHIP
    if _remaining() > 10.0:
        pchip_pred, pchip_mse = _fit_pchip(X_train, y_train)
        if pchip_pred is not None and pchip_mse < best_mse:
            best_mse = pchip_mse
            best_predict = pchip_pred

    # 2. Cubic Spline
    if _remaining() > 10.0:
        cs_pred, cs_mse = _fit_cubic_spline(X_train, y_train)
        if cs_pred is not None and cs_mse < best_mse:
            best_mse = cs_mse
            best_predict = cs_pred

    # 3. Chebyshev
    if _remaining() > 15.0:
        ch_pred, ch_mse = _fit_chebyshev(X_train, y_train, max_degree=80)
        if ch_pred is not None and ch_mse < best_mse:
            best_mse = ch_mse
            best_predict = ch_pred

    # 4. RBF Interpolation
    if _remaining() > 15.0:
        rbf_pred, rbf_mse = _fit_rbf_interpolation(X_train, y_train)
        if rbf_pred is not None and rbf_mse < best_mse:
            best_mse = rbf_mse
            best_predict = rbf_pred

    # 5. Local Regression
    if _remaining() > 20.0:
        lr_pred, lr_mse = _fit_local_regression(X_train, y_train)
        if lr_pred is not None and lr_mse < best_mse:
            best_mse = lr_mse
            best_predict = lr_pred

    # 6. Adaptive Grid
    if _remaining() > 10.0:
        ag_pred, ag_mse = _fit_adaptive_grid(X_train, y_train)
        if ag_pred is not None and ag_mse < best_mse:
            best_mse = ag_mse
            best_predict = ag_pred

    # Final safety check
    if best_predict is None:
        best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    return best_predict
# EVOLVE-BLOCK-END
