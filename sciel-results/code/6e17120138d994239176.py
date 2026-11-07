import time
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline

_DEADLINE = time.time() + 290.0

def _fit_pchip(X, y):
    """Fit PCHIP interpolator."""
    X = np.asarray(X).flatten()
    y = np.asarray(y).flatten()
    # Sort by X
    idx = np.argsort(X)
    Xs = X[idx]
    ys = y[idx]
    
    # Check for duplicate X values
    # If duplicates exist, average them
    unique_X, unique_indices, counts = np.unique(Xs, return_inverse=True, return_counts=True)
    if len(unique_X) < len(Xs):
        ys_sum = np.zeros_like(unique_X)
        np.add.at(ys_sum, unique_indices, ys)
        ys_avg = ys_sum / counts
        Xs = unique_X
        ys = ys_avg
        idx = np.argsort(Xs)
        Xs = Xs[idx]
        ys = ys[idx]

    try:
        pchip = PchipInterpolator(Xs, ys)
    except Exception:
        # Fallback to linear if PCHIP fails (shouldn't happen with unique X)
        pchip = None
        def predict_lin(Xq):
            Xq = np.asarray(Xq).flatten()
            return np.interp(Xq, Xs, ys)
        return predict_lin

    def predict(Xq):
        Xq = np.asarray(Xq).flatten()
        return pchip(Xq)
    return predict

def _fit_poly(X, y, degree=10):
    """Fit polynomial using Chebyshev nodes for stability."""
    X = np.asarray(X).flatten()
    y = np.asarray(y).flatten()
    
    # Map to [-1, 1] for better conditioning
    x_min, x_max = X.min(), X.max()
    if x_max == x_min:
        x_max = x_min + 1.0
    Xs = 2.0 * (X - x_min) / (x_max - x_min) - 1.0
    
    # Use Chebyshev polynomials basis for stability
    # V[:, k] = T_k(Xs)
    # T_0 = 1, T_1 = x, T_{k+1} = 2x T_k - T_{k-1}
    V = np.zeros((len(Xs), degree + 1))
    V[:, 0] = 1.0
    if degree >= 1:
        V[:, 1] = Xs
    for k in range(2, degree + 1):
        V[:, k] = 2.0 * Xs * V[:, k-1] - V[:, k-2]
        
    # Least squares
    w, *_ = np.linalg.lstsq(V, y, rcond=None)
    
    def predict(Xq):
        Xq = np.asarray(Xq).flatten()
        Xqs = 2.0 * (Xq - x_min) / (x_max - x_min) - 1.0
        Vq = np.zeros((len(Xqs), degree + 1))
        Vq[:, 0] = 1.0
        if degree >= 1:
            Vq[:, 1] = Xqs
        for k in range(2, degree + 1):
            Vq[:, k] = 2.0 * Xqs * Vq[:, k-1] - Vq[:, k-2]
        return Vq @ w
    return predict

def _fit_rbf(X, y, epsilon=1.0):
    """Fit RBF interpolator."""
    X = np.asarray(X).flatten()
    y = np.asarray(y).flatten()
    n = len(X)
    
    # Kernel matrix
    d = np.abs(X[:, None] - X[None, :])
    K = np.exp(-epsilon * d)
    K[np.diag_indices_from(K)] += 1e-8 # jitter
    
    try:
        alpha = np.linalg.solve(K, y)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(K, y, rcond=None)[0]
        
    def predict(Xq):
        Xq = np.asarray(Xq).flatten()
        d = np.abs(Xq[:, None] - X[None, :])
        Kq = np.exp(-epsilon * d)
        return Kq @ alpha
    return predict

def fit(X_train, y_train):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X_use = X[:, 0]
    y_use = y
    
    # Handle constant y
    if np.std(y_use) < 1e-10:
        mean_y = np.mean(y_use)
        return lambda Xq: np.full(len(np.asarray(Xq).flatten()), mean_y)
        
    # Split for validation
    n = len(X_use)
    indices = np.arange(n)
    np.random.shuffle(indices)
    val_size = max(20, n // 10)
    val_idx = indices[:val_size]
    trn_idx = indices[val_size:]
    
    X_trn = X_use[trn_idx]
    y_trn = y_use[trn_idx]
    X_val = X_use[val_idx]
    y_val = y_use[val_idx]
    
    candidates = []
    
    # 1. PCHIP
    if time.time() < _DEADLINE:
        try:
            pred_pchip = _fit_pchip(X_trn, y_trn)
            pred_val = pred_pchip(X_val)
            err = np.mean((pred_val - y_val)**2)
            candidates.append((err, _fit_pchip, X_trn, y_trn))
        except Exception:
            pass
            
    # 2. Poly degree 8
    if time.time() < _DEADLINE:
        try:
            pred_poly8 = _fit_poly(X_trn, y_trn, degree=8)
            pred_val = pred_poly8(X_val)
            err = np.mean((pred_val - y_val)**2)
            candidates.append((err, _fit_poly, X_trn, y_trn, 8))
        except Exception:
            pass
            
    # 3. Poly degree 12
    if time.time() < _DEADLINE:
        try:
            pred_poly12 = _fit_poly(X_trn, y_trn, degree=12)
            pred_val = pred_poly12(X_val)
            err = np.mean((pred_val - y_val)**2)
            candidates.append((err, _fit_poly, X_trn, y_trn, 12))
        except Exception:
            pass

    # 4. RBF
    if time.time() < _DEADLINE:
        try:
            # Estimate epsilon based on data range
            x_range = X_trn.max() - X_trn.min()
            if x_range > 0:
                eps = 1.0 / (x_range / 5.0) # rough heuristic
            else:
                eps = 1.0
            pred_rbf = _fit_rbf(X_trn, y_trn, epsilon=eps)
            pred_val = pred_rbf(X_val)
            err = np.mean((pred_val - y_val)**2)
            candidates.append((err, _fit_rbf, X_trn, y_trn, eps))
        except Exception:
            pass

    if not candidates:
        # Fallback: Linear
        X_lin = np.column_stack([np.ones(len(X_use)), X_use])
        w, *_ = np.linalg.lstsq(X_lin, y_use, rcond=None)
        def predict_lin(Xq):
            Xq = np.asarray(Xq).flatten()
            Xq_lin = np.column_stack([np.ones(len(Xq)), Xq])
            return Xq_lin @ w
        return predict_lin

    # Sort by error
    candidates.sort(key=lambda x: x[0])
    best_err, fit_func, *args = candidates[0]
    
    # Refit on all data
    if fit_func == _fit_pchip:
        final_pred = _fit_pchip(X_use, y_use)
    elif fit_func == _fit_poly:
        final_pred = _fit_poly(X_use, y_use, degree=args[0])
    elif fit_func == _fit_rbf:
        final_pred = _fit_rbf(X_use, y_use, epsilon=args[0])
    else:
        final_pred = _fit_pchip(X_use, y_use) # Fallback
        
    def predict(Xq):
        Xq = np.asarray(Xq, dtype=float)
        if Xq.ndim == 1:
            Xq = Xq.reshape(-1, 1)
        Xq_use = Xq[:, 0]
        result = final_pred(Xq_use)
        result = np.asarray(result, dtype=float)
        result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
        return result
        
    return predict
