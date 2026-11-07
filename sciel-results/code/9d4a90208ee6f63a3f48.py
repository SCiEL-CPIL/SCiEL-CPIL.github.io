"""Symbolic regression: ensemble of polynomial and RBF kernels, with global time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

_TOTAL_START = None
_TIME_LIMIT = 325.0  # Safety margin before 330s kill

def _elapsed():
    global _TOTAL_START
    if _TOTAL_START is None:
        _TOTAL_START = time.time()
    return time.time() - _TOTAL_START

def _remaining():
    return max(0.0, _TIME_LIMIT - _elapsed())

def _safe_2d(X):
    """Ensure X is 2D float64."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X

def _rbf_kernel(X, Y, gamma):
    """Compute RBF kernel matrix between X (n,d) and Y (m,d)."""
    X2 = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y2 = np.sum(Y ** 2, axis=1).reshape(1, -1)
    dist2 = X2 - 2.0 * X @ Y.T + Y2
    dist2 = np.maximum(dist2, 0.0)
    return np.exp(-gamma * dist2)

def _poly_kernel(X, Y, degree, coef0=1.0, gamma=1.0):
    """Compute polynomial kernel matrix."""
    return (gamma * X @ Y.T + coef0) ** degree

def _linear_kernel(X, Y):
    return X @ Y.T

def _make_predict(kernel_func, alpha, y_mean, y_std, X_train):
    """Create a prediction callable for kernel ridge regression."""
    def predict(X_test):
        X_test = _safe_2d(X_test)
        K = kernel_func(X_test, X_train, **{})
        y_pred_norm = K @ alpha
        return y_pred_norm * y_std + y_mean
    return predict

def fit(X_train, y_train):
    global _TOTAL_START
    if _TOTAL_START is None:
        _TOTAL_START = time.time()
    
    X = _safe_2d(X_train)
    y = y_train.astype(np.float64).reshape(-1)
    n, d = X.shape
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Normalize X for better kernel behavior
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X - X_mean) / X_std
    
    # Store for predict
    X_train_norm = X_norm.copy()
    
    best_predict = None
    best_mse = np.inf
    
    # Helper to compute MSE for a kernel model
    def evaluate_kernel(K_train, lam):
        """Solve kernel ridge regression and return (alpha, mse)."""
        try:
            A = K_train + lam * np.eye(n)
            alpha = np.linalg.solve(A, y_norm)
            pred = K_train @ alpha
            mse = np.mean((pred - y_norm) ** 2)
            return alpha, mse
        except:
            return None, np.inf
    
    # Try RBF kernels with various gamma values
    gamma_values = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
    lam_values = [1e-6, 1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0]
    
    # Pre-compute RBF kernels for different gammas
    rbf_kernels = {}
    for gamma in gamma_values:
        if _remaining() < 5.0:
            break
        try:
            K = _rbf_kernel(X_norm, X_norm, gamma)
            # Ensure symmetry
            K = (K + K.T) / 2.0
            rbf_kernels[gamma] = K
        except:
            continue
    
    # Evaluate RBF models
    for gamma, K in rbf_kernels.items():
        if _remaining() < 3.0:
            break
        for lam in lam_values:
            if _remaining() < 2.0:
                break
            alpha, mse = evaluate_kernel(K, lam)
            if alpha is not None and mse < best_mse:
                best_mse = mse
                best_predict = _make_predict(lambda Xt, Yt, g=gamma: _rbf_kernel(Xt, Yt, g), alpha, y_mean, y_std, X_train_norm)
    
    # Try polynomial kernels
    poly_configs = [
        (2, 1.0, 1.0),
        (2, 0.5, 1.0),
        (3, 1.0, 1.0),
        (3, 0.5, 1.0),
        (4, 1.0, 0.5),
    ]
    
    for degree, coef0, gamma in poly_configs:
        if _remaining() < 5.0:
            break
        try:
            K = _poly_kernel(X_norm, X_norm, degree, coef0, gamma)
            K = (K + K.T) / 2.0
        except:
            continue
        
        for lam in lam_values:
            if _remaining() < 2.0:
                break
            alpha, mse = evaluate_kernel(K, lam)
            if alpha is not None and mse < best_mse:
                best_mse = mse
                best_predict = _make_predict(
                    lambda Xt, Yt, d=degree, c=coef0, g=gamma: _poly_kernel(Xt, Yt, d, c, g),
                    alpha, y_mean, y_std, X_train_norm
                )
    
    # Try linear kernel
    if _remaining() > 5.0:
        try:
            K = _linear_kernel(X_norm, X_norm)
            K = (K + K.T) / 2.0
            for lam in lam_values:
                if _remaining() < 2.0:
                    break
                alpha, mse = evaluate_kernel(K, lam)
                if alpha is not None and mse < best_mse:
                    best_mse = mse
                    best_predict = _make_predict(_linear_kernel, alpha, y_mean, y_std, X_train_norm)
        except:
            pass
    
    # If no kernel model worked, fallback to simple linear regression
    if best_predict is None:
        try:
            # Add bias
            X_b = np.hstack([X_norm, np.ones((n, 1))])
            w, *_ = np.linalg.lstsq(X_b, y_norm, rcond=None)
            def predict(X_test):
                X_t = _safe_2d(X_test)
                X_t_norm = (X_t - X_mean) / X_std
                X_t_b = np.hstack([X_t_norm, np.ones((X_t_norm.shape[0], 1))])
                return (X_t_b @ w) * y_std + y_mean
            best_predict = predict
        except:
            # Absolute fallback: predict mean
            def predict(X_test):
                return np.full(X_test.shape[0], y_mean)
            best_predict = predict
    
    return best_predict
# EVOLVE-BLOCK-END
