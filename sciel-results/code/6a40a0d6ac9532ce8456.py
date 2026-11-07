"""
Symbolic regression using Gaussian Process with Matérn kernel and adaptive hyperparameter search.
"""
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")

TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0


def _time_left():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _t_start)


def _matern_kernel(X1, X2, length_scale, nu=2.5):
    """
    Compute Matérn kernel matrix.
    X1, X2: (N, D), (M, D)
    length_scale: scalar or (D,)
    """
    # Standardize X1 and X2 internally for stability if not done outside, 
    # but here we assume inputs are standardized or handle scaling via length_scale.
    # Actually, better to standardize X in fit and pass standardized X to kernel.
    
    if length_scale.ndim == 0:
        ls = float(length_scale)
    else:
        ls = length_scale
    
    # Compute squared distances
    # X1: (N, D), X2: (M, D)
    # d^2 = sum((x1 - x2)^2)
    sqdist = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
    sqdist = np.maximum(sqdist, 0)
    
    r = np.sqrt(sqdist) / ls
    
    if nu == 2.5:
        sqrt3_r = np.sqrt(3.0) * r
        K = (1.0 + sqrt3_r + (sqrt3_r**2) / 3.0) * np.exp(-sqrt3_r)
    elif nu == 1.5:
        sqrt3_r = np.sqrt(3.0) * r
        K = (1.0 + sqrt3_r) * np.exp(-sqrt3_r)
    elif nu == 0.5:
        K = np.exp(-r)
    else:
        # Fallback to RBF
        K = np.exp(-0.5 * r**2)
        
    return K


def _fit_gp(X, y, length_scale, noise, nu=2.5):
    """
    Fit GP regression.
    X: (N, D) standardized
    y: (N,) standardized
    Returns: (predict_func, train_mse)
    """
    n = X.shape[0]
    K = _matern_kernel(X, X, length_scale, nu)
    K_reg = K + noise * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K_reg)
    except np.linalg.LinAlgError:
        # Add jitter
        K_reg += 1e-6 * np.eye(n)
        try:
            L = np.linalg.cholesky(K_reg)
        except np.linalg.LinAlgError:
            return None, np.inf

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    
    train_pred = K @ alpha
    train_mse = np.mean((train_pred - y) ** 2)
    
    # Precompute inverse for prediction
    # We need K(X_test, X_train) @ alpha
    # To be efficient, we can store L and alpha.
    # predict: K_test @ alpha
    
    def predict(X_test):
        K_test = _matern_kernel(X_test, X, length_scale, nu)
        return K_test @ alpha
        
    return predict, train_mse


def fit(X_train, y_train):
    global _t_start
    _t_start = time.time()
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    n, d = X.shape
    
    # Standardize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    Xn = (X - X_mean) / X_std
    
    y_mean = y.mean()
    y_std = y.std() + 1e-8
    yn = (y - y_mean) / y_std
    
    # Hyperparameter search
    # Since d is small (1), length_scale is a scalar or vector of size 1.
    # We search over a grid of length scales and noise levels.
    
    best_predictor = None
    best_train_mse = np.inf
    best_params = None
    
    # Grid search
    # Length scales: from very short (wiggly) to very long (smooth)
    # Standardized X is roughly in [-2, 2] or similar.
    # Length scale 0.1 -> very local. 1.0 -> medium. 5.0 -> global.
    length_scales = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    noise_levels = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    nus = [0.5, 1.5, 2.5]
    
    # To save time, we can do a coarse-to-fine search or just a reasonable grid.
    # 7 * 6 * 3 = 126 fits. Each fit is O(N^3) = 200^3 = 8e6 ops.
    # 126 * 8e6 = 1e9 ops. Might be too slow in pure Python/NumPy loops if not optimized.
    # Cholesky is fast in BLAS/LAPACK. 200x200 Cholesky is very fast (<1ms).
    # So 126 fits is likely < 0.5 seconds.
    
    for nu in nus:
        for ls in length_scales:
            for noise in noise_levels:
                if _time_left() < 5.0:
                    break
                
                # If d > 1, we might need ARD, but here d=1 mostly.
                # If d > 1, use scalar ls for simplicity in grid, or vector.
                # Let's use scalar ls for all dims for now, or if d==1, scalar.
                # If d > 1, we can try a few diagonal length scales.
                # Given the problem description, all inputs are 1D.
                
                try:
                    predict_func, train_mse = _fit_gp(Xn, yn, ls, noise, nu)
                    if predict_func is not None and np.isfinite(train_mse) and train_mse < best_train_mse:
                        best_train_mse = train_mse
                        best_predictor = predict_func
                        best_params = (nu, ls, noise)
                except Exception:
                    continue
            
            if _time_left() < 5.0:
                break
        
        if _time_left() < 5.0:
            break
    
    if best_predictor is None:
        # Fallback: Linear regression
        A = np.column_stack([np.ones(n)] + [X[:, j] for j in range(d)])
        w = np.linalg.lstsq(A, y, rcond=None)[0]
        def predict_fallback(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            A_test = np.column_stack([np.ones(len(X_test))] + [X_test[:, j] for j in range(d)])
            return A_test @ w
        return predict_fallback
    
    # Wrap predictor to denormalize
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        Xn_test = (X_test - X_mean) / X_std
        y_pred_std = best_predictor(Xn_test)
        return y_pred_std * y_std + y_mean

    return predict
