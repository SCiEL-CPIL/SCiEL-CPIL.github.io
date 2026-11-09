# EVOLVE-BLOCK-START
import numpy as np
import time


def _matern52_kernel(X1, X2, length_scales, signal_var, noise_var):
    """
    Compute Matérn-5/2 kernel matrix between X1 (n,3) and X2 (m,3).
    Uses anisotropic length scales.
    """
    # X1: (n, 3), X2: (m, 3), length_scales: (3,)
    # Compute squared distance with anisotropic scaling
    # d^2 = sum_j ((X1[i,j] - X2[k,j]) / l_j)^2
    X1_scaled = X1 / length_scales  # (n, 3)
    X2_scaled = X2 / length_scales  # (m, 3)
    
    # Squared distances
    d2 = np.sum(X1_scaled ** 2, axis=1, keepdims=True) + \
         np.sum(X2_scaled ** 2, axis=1, keepdims=True).T - \
         2.0 * (X1_scaled @ X2_scaled.T)
    d2 = np.maximum(d2, 0.0)
    d = np.sqrt(d2)
    
    # Matérn-5/2 kernel: (1 + sqrt(3)*d) * exp(-sqrt(3)*d)
    sqrt3_d = np.sqrt(3.0) * d
    K = (1.0 + sqrt3_d) * np.exp(-sqrt3_d)
    K *= signal_var
    
    # Add noise to diagonal
    if X1.shape[0] == X2.shape[0]:
        K[np.diag_indices_from(K)] += noise_var
    
    return K


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    GP with Matérn-5/2 kernel.
    params[0:3] = log length scales (for x, y, z)
    params[3]   = log signal variance
    params[4]   = log noise variance
    params[5:10] = unused (kept for interface compatibility)
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Ensure inputs are 1D arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()
    n = x.shape[0]
    
    # Stack inputs: (n, 3)
    X_train = np.column_stack([x, y, z])
    
    # Extract parameters with safe transforms
    # Length scales: exp(params[0:3]) to ensure positivity
    log_ls = params[0:3]
    length_scales = np.exp(np.clip(log_ls, -5.0, 5.0))  # clip to avoid overflow/underflow
    
    # Signal variance
    log_sv = params[3]
    signal_var = np.exp(np.clip(log_sv, -10.0, 10.0))
    
    # Noise variance
    log_nv = params[4]
    noise_var = np.exp(np.clip(log_nv, -10.0, 5.0))
    
    # Compute kernel matrix
    K = _matern52_kernel(X_train, X_train, length_scales, signal_var, noise_var)
    
    # Solve for GP coefficients: alpha = (K + I*eps)^{-1} * y_train
    # Use Cholesky for stability
    eps = 1e-8
    K_reg = K + eps * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K_reg)
    except np.linalg.LinAlgError:
        # Fallback: add more regularization
        K_reg = K + 1e-4 * np.eye(n)
        try:
            L = np.linalg.cholesky(K_reg)
        except np.linalg.LinAlgError:
            # Last resort: use pinv
            alpha = np.linalg.pinv(K_reg) @ y
            # Predict: f = K(x_new, x_train) @ alpha
            # For training data prediction:
            K_pred = K  # same as K without noise for prediction
            pred = K_pred @ alpha
            return pred
    
    # Solve L @ L^T @ alpha = y
    # First: L @ v = y
    v = np.linalg.solve(L, y)
    # Then: L^T @ alpha = v
    alpha = np.linalg.solve(L.T, v)
    
    # Prediction for training points (for fitting)
    # f(x_i) = sum_j K(x_i, x_j) * alpha_j
    K_pred = _matern52_kernel(X_train, X_train, length_scales, signal_var, 0.0)
    pred = K_pred @ alpha
    
    # Check time
    if time.time() > deadline:
        return pred
    
    # Ensure finite
    if not np.all(np.isfinite(pred)):
        return np.zeros(n)
    
    return pred
# EVOLVE-BLOCK-END
