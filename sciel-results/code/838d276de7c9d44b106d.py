"""Kernel Ridge Regression with composite kernel for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _rbf_kernel(X1, X2, length_scale):
    """RBF kernel: exp(-||x-y||^2 / (2 * length_scale^2))"""
    sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
    sq_dists = np.maximum(sq_dists, 0)
    return np.exp(-sq_dists / (2 * length_scale**2))

def _poly_kernel(X1, X2, degree, gamma, coef0):
    """Polynomial kernel: (gamma * x^T y + coef0)^degree"""
    return (gamma * X1 @ X2.T + coef0) ** degree

def _periodic_kernel(X1, X2, period, length_scale):
    """Periodic kernel: exp(-2 * sum(sin^2(pi*(x_i-y_i)/period)) / length_scale^2)"""
    diff = X1[:, :, None] - X2[:, None, :]
    sin_diff = np.sin(np.pi * diff / period)
    sq_dists = 2.0 * np.sum(sin_diff**2, axis=1)
    return np.exp(-sq_dists / (length_scale**2))

def _composite_kernel(X1, X2, params):
    """
    Composite kernel: sum of RBF, polynomial, and periodic kernels.
    params: (rbf_ls, poly_degree, poly_gamma, poly_coef0, period, per_ls, w_rbf, w_poly, w_per)
    """
    rbf_ls, poly_degree, poly_gamma, poly_coef0, period, per_ls, w_rbf, w_poly, w_per = params
    
    K = w_rbf * _rbf_kernel(X1, X2, rbf_ls)
    K += w_poly * _poly_kernel(X1, X2, poly_degree, poly_gamma, poly_coef0)
    K += w_per * _periodic_kernel(X1, X2, period, per_ls)
    return K

def _neg_log_likelihood(params, X, y, alpha):
    """Negative log-likelihood for kernel ridge regression (Gaussian process)."""
    try:
        K = _composite_kernel(X, X, params)
        n = len(y)
        # Add small jitter for numerical stability
        K += 1e-8 * np.eye(n)
        
        # Solve for coefficients
        try:
            L = np.linalg.cholesky(K + alpha * np.eye(n))
            alpha_coeff = np.linalg.solve(L, y)
            alpha_coeff = np.linalg.solve(L.T, alpha_coeff)
            
            # Log-likelihood
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            ll = -0.5 * (y @ alpha_coeff + log_det + n * np.log(2 * np.pi))
            return -ll
        except np.linalg.LinAlgError:
            return 1e10
    except Exception:
        return 1e10

def fit(X_train, y_train):
    """
    Kernel Ridge Regression with composite kernel.
    Returns a predict callable.
    """
    start_time = time.time()
    deadline = start_time + 290  # 290s safety margin
    
    X = X_train.astype(np.float64)
    y = y_train.astype(np.float64)
    n, d = X.shape
    
    # Normalize inputs to [0, 1] for better kernel scaling
    x_min = X.min(axis=0, keepdims=True)
    x_max = X.max(axis=0, keepdims=True)
    x_range = x_max - x_min
    x_range[x_range == 0] = 1.0
    X_norm = (X - x_min) / x_range
    
    # Initialize parameters
    # (rbf_ls, poly_degree, poly_gamma, poly_coef0, period, per_ls, w_rbf, w_poly, w_per)
    # Use reasonable initial values
    init_params = np.array([1.0, 2, 1.0, 1.0, 2.0, 1.0, 1.0, 0.5, 0.5])
    
    # Define bounds for continuous parameters (poly_degree is integer, handled separately)
    bounds = [
        (0.01, 5.0),    # rbf_ls
        (1, 5),         # poly_degree (integer)
        (0.01, 10.0),   # poly_gamma
        (0.0, 5.0),     # poly_coef0
        (0.5, 10.0),    # period
        (0.01, 5.0),    # per_ls
        (0.01, 10.0),   # w_rbf
        (0.01, 10.0),   # w_poly
        (0.01, 10.0),   # w_per
    ]
    
    # Simple grid search over a few key hyperparameters
    # To keep it fast, we'll do a small grid
    rbf_ls_values = [0.3, 0.5, 1.0, 2.0]
    poly_degree_values = [2, 3, 4]
    w_rbf_values = [1.0, 2.0]
    w_poly_values = [0.5, 1.0]
    w_per_values = [0.1, 0.5, 1.0]
    
    best_score = np.inf
    best_params = init_params.copy()
    
    # Evaluate a small grid
    for rbf_ls in rbf_ls_values:
        if time.time() > deadline - 30:
            break
        for poly_degree in poly_degree_values:
            if time.time() > deadline - 30:
                break
            for w_rbf in w_rbf_values:
                if time.time() > deadline - 30:
                    break
                for w_poly in w_poly_values:
                    if time.time() > deadline - 30:
                        break
                    for w_per in w_per_values:
                        if time.time() > deadline - 30:
                            break
                        
                        params = np.array([rbf_ls, poly_degree, 1.0, 1.0, 2.0, 1.0, w_rbf, w_poly, w_per])
                        
                        # Compute kernel matrix
                        K = _composite_kernel(X_norm, X_norm, params)
                        K += 1e-6 * np.eye(n)
                        
                        # Solve for coefficients
                        try:
                            alpha_coeff = np.linalg.solve(K, y)
                            pred = K @ alpha_coeff
                            mse = np.mean((pred - y)**2)
                            var_y = np.var(y)
                            if var_y > 1e-10:
                                norm_mse = mse / var_y
                            else:
                                norm_mse = mse
                            
                            if norm_mse < best_score:
                                best_score = norm_mse
                                best_params = params.copy()
                        except np.linalg.LinAlgError:
                            continue
    
    # Use the best parameters found
    K_train = _composite_kernel(X_norm, X_norm, best_params)
    K_train += 1e-6 * np.eye(n)
    
    # Solve for coefficients
    try:
        alpha_coeff = np.linalg.solve(K_train, y)
    except np.linalg.LinAlgError:
        # Fallback to lstsq
        alpha_coeff, *_ = np.linalg.lstsq(K_train, y, rcond=None)
    
    # Create predict function
    def predict(X_test):
        X_test = X_test.astype(np.float64)
        X_test_norm = (X_test - x_min) / x_range
        K_test = _composite_kernel(X_test_norm, X_norm, best_params)
        return K_test @ alpha_coeff
    
    return predict
# EVOLVE-BLOCK-END
