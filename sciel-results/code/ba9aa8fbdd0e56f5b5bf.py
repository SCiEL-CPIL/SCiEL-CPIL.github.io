"""Symbolic regression via multi-kernel ridge with LOOCV and robust hyperparameter search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _rbf_kernel(X1, X2, length_scales):
    """Compute RBF kernel between two sets of points with per-dimension length scales."""
    X1s = X1 / length_scales
    X2s = X2 / length_scales
    sq1 = np.sum(X1s**2, axis=1)[:, None]
    sq2 = np.sum(X2s**2, axis=1)[None, :]
    sq_dists = sq1 + sq2 - 2.0 * (X1s @ X2s.T)
    np.maximum(sq_dists, 0.0, out=sq_dists)
    return np.exp(-0.5 * sq_dists)


def _poly_kernel(X1, X2, degree, gamma, coef0):
    """Compute polynomial kernel."""
    return (gamma * (X1 @ X2.T) + coef0) ** degree


def _periodic_kernel(X1, X2, period, length_scale):
    """Compute periodic kernel."""
    diff = X1[:, None, :] - X2[None, :, :]
    sin_diff = np.sin(np.pi * diff / period)
    sq_sum = np.sum(sin_diff**2, axis=2)
    return np.exp(-2.0 * sq_sum / (length_scale**2))


def _compute_kernel(X1, X2, params):
    """
    Compute composite kernel: sum of RBF + polynomial + periodic kernels.
    """
    K = np.zeros((X1.shape[0], X2.shape[0]))
    
    if params.get('rbf_weight', 0) > 1e-12:
        K += params['rbf_weight'] * _rbf_kernel(X1, X2, params['rbf_ls'])
    
    if params.get('poly_weight', 0) > 1e-12:
        K += params['poly_weight'] * _poly_kernel(X1, X2, params['poly_degree'], 
                                                   params['poly_gamma'], params['poly_coef0'])
    
    if params.get('per_weight', 0) > 1e-12:
        K += params['per_weight'] * _periodic_kernel(X1, X2, params['per_period'], params['per_ls'])
    
    return K


def _kernel_ridge_solve(K, y, alpha_reg):
    """Solve kernel ridge regression: (K + alpha*I)^{-1} y"""
    n = K.shape[0]
    A = K + alpha_reg * np.eye(n)
    try:
        coeffs = np.linalg.solve(A, y)
    except np.linalg.LinAlgError:
        coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coeffs


def _loocv_mse(K, y, alpha):
    """
    Compute LOOCV MSE efficiently using the kernel matrix inverse.
    For kernel ridge regression, LOOCV prediction for point i is:
    y_i - K_ii / (K + alpha*I)^{-1}[i,i] * (K + alpha*I)^{-1} y
    Equivalently: pred_i = y_i - (K @ coeffs)[i] / (K + alpha*I)^{-1}[i,i]
    where coeffs = (K + alpha*I)^{-1} y
    """
    n = K.shape[0]
    A = K + alpha * np.eye(n)
    try:
        # Solve A * inv_A = I, or just solve A * coeffs = y and use diagonal of inv
        # More efficient: compute coeffs = A^{-1} y, then use formula
        coeffs = np.linalg.solve(A, y)
        # The diagonal of A^{-1} can be computed by solving A * diag = e_i for each i
        # But that's O(n^3). Instead, use the identity:
        # LOOCV: pred_i = y_i - (K @ coeffs)[i] / (A^{-1})[i,i]
        # We need (A^{-1})[i,i]. We can get all diagonal entries by:
        # A^{-1} = (A^{-1} A) A^{-1} ... no.
        # Better: solve A * X = I to get A^{-1}, but that's expensive.
        # Alternative: use the fact that for KRR, the LOOCV error is:
        # e_i = (y_i - pred_i) where pred_i = sum_j K_ij * c_j - K_ii * c_i
        # Actually the standard formula is:
        # pred_i^{(i)} = (K @ coeffs)_i - K_ii * coeffs_i / (1 + K_ii * alpha^{-1})... 
        # Let me use the cleaner formula:
        # Let H = (K + alpha*I)^{-1} K. Then LOOCV pred_i = (H @ y)_i
        # And H = I - alpha * (K + alpha*I)^{-1}
        # So we need the diagonal of (K + alpha*I)^{-1}
        
        # Compute A^{-1} by solving A X = I
        # This is O(n^3) but n=200, so it's fine
        A_inv = np.linalg.inv(A)
        diag_inv = np.diag(A_inv)
        # Avoid division by zero
        diag_inv = np.maximum(np.abs(diag_inv), 1e-15)
        
        # LOOCV predictions: pred_i = y_i - (K @ coeffs)[i] / diag_inv[i]
        K_coeffs = K @ coeffs
        loocv_preds = y - K_coeffs / diag_inv
        loocv_errors = y - loocv_preds
        return np.mean(loocv_errors**2)
    except np.linalg.LinAlgError:
        return np.inf


def fit(X_train, y_train):
    t0 = time.time()
    deadline = t0 + 300.0  # Safety margin before 330s kill
    
    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_centered = y_train - y_mean
    
    # Normalize inputs to [0, 1]
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    x_range = x_max - x_min + 1e-12
    X_norm = (X_train - x_min) / x_range
    
    best_params = None
    best_cv_mse = np.inf
    
    # Phase 1: Coarse grid search over RBF length scales and regularization
    rbf_ls_grid = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    alpha_grid = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
    
    for ls_val in rbf_ls_grid:
        if time.time() > deadline - 10:
            break
        for alpha in alpha_grid:
            if time.time() > deadline - 10:
                break
            
            params = {
                'rbf_weight': 1.0, 'rbf_ls': np.full(d, ls_val),
                'poly_weight': 0.0, 'poly_degree': 2, 'poly_gamma': 1.0, 'poly_coef0': 1.0,
                'per_weight': 0.0, 'per_period': 2.0, 'per_ls': 1.0
            }
            
            K = _compute_kernel(X_norm, X_norm, params)
            cv_mse = _loocv_mse(K, y_centered, alpha)
            if cv_mse < best_cv_mse:
                best_cv_mse = cv_mse
                best_params = params.copy()
                best_params['alpha'] = alpha
    
    if best_params is None:
        best_params = {
            'rbf_weight': 1.0, 'rbf_ls': np.ones(d),
            'poly_weight': 0.0, 'poly_degree': 2, 'poly_gamma': 1.0, 'poly_coef0': 1.0,
            'per_weight': 0.0, 'per_period': 2.0, 'per_ls': 1.0,
            'alpha': 1e-2
        }
    
    # Phase 2: Anisotropic length scales if d > 1
    if d > 1 and time.time() < deadline - 20:
        base_ls = best_params['rbf_ls'].copy()
        for dim in range(d):
            if time.time() > deadline - 10:
                break
            for ls_mult in [0.3, 0.5, 0.7, 1.4, 2.0, 3.0]:
                if time.time() > deadline - 10:
                    break
                new_ls = base_ls.copy()
                new_ls[dim] *= ls_mult
                params = best_params.copy()
                params['rbf_ls'] = new_ls
                
                K = _compute_kernel(X_norm, X_norm, params)
                cv_mse = _loocv_mse(K, y_centered, best_params['alpha'])
                if cv_mse < best_cv_mse:
                    best_cv_mse = cv_mse
                    best_params = params.copy()
    
    # Phase 3: Add polynomial kernel
    poly_configs = [
        {'weight': 0.1, 'degree': 2, 'gamma': 1.0, 'coef0': 1.0},
        {'weight': 0.5, 'degree': 2, 'gamma': 1.0, 'coef0': 1.0},
        {'weight': 1.0, 'degree': 2, 'gamma': 1.0, 'coef0': 1.0},
        {'weight': 0.5, 'degree': 3, 'gamma': 0.5, 'coef0': 1.0},
        {'weight': 1.0, 'degree': 3, 'gamma': 0.5, 'coef0': 1.0},
        {'weight': 0.5, 'degree': 4, 'gamma': 0.5, 'coef0': 1.0},
    ]
    
    for poly_cfg in poly_configs:
        if time.time() > deadline - 10:
            break
        for ls_mult in [0.5, 1.0, 2.0]:
            if time.time() > deadline - 10:
                break
            for alpha_mult in [0.1, 1.0, 10.0]:
                if time.time() > deadline - 10:
                    break
                
                params = best_params.copy()
                params['rbf_ls'] = best_params['rbf_ls'].copy() * ls_mult
                alpha = best_params['alpha'] * alpha_mult
                params['alpha'] = alpha
                params['poly_weight'] = poly_cfg['weight']
                params['poly_degree'] = poly_cfg['degree']
                params['poly_gamma'] = poly_cfg['gamma']
                params['poly_coef0'] = poly_cfg['coef0']
                
                K = _compute_kernel(X_norm, X_norm, params)
                cv_mse = _loocv_mse(K, y_centered, alpha)
                if cv_mse < best_cv_mse:
                    best_cv_mse = cv_mse
                    best_params = params.copy()
    
    # Phase 4: Add periodic kernel
    per_configs = [
        {'weight': 0.1, 'period': 2.0, 'ls': 0.5},
        {'weight': 0.5, 'period': 2.0, 'ls': 0.5},
        {'weight': 0.1, 'period': 1.0, 'ls': 0.3},
        {'weight': 0.5, 'period': 1.0, 'ls': 0.3},
        {'weight': 0.1, 'period': 5.0, 'ls': 1.0},
        {'weight': 0.5, 'period': 5.0, 'ls': 1.0},
    ]
    
    for per_cfg in per_configs:
        if time.time() > deadline - 10:
            break
        
        params = best_params.copy()
        params['per_weight'] = per_cfg['weight']
        params['per_period'] = per_cfg['period']
        params['per_ls'] = per_cfg['ls']
        
        K = _compute_kernel(X_norm, X_norm, params)
        cv_mse = _loocv_mse(K, y_centered, best_params['alpha'])
        if cv_mse < best_cv_mse:
            best_cv_mse = cv_mse
            best_params = params.copy()
    
    # Phase 5: Fine-tune around best parameters
    if time.time() < deadline - 15:
        base_ls = best_params['rbf_ls'].copy()
        base_alpha = best_params['alpha']
        
        for ls_mult in [0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.5]:
            if time.time() > deadline - 10:
                break
            for alpha_mult in [0.5, 0.7, 1.3, 2.0, 3.0]:
                if time.time() > deadline - 10:
                    break
                
                params = best_params.copy()
                params['rbf_ls'] = base_ls * ls_mult
                alpha = base_alpha * alpha_mult
                params['alpha'] = alpha
                
                K = _compute_kernel(X_norm, X_norm, params)
                cv_mse = _loocv_mse(K, y_centered, alpha)
                if cv_mse < best_cv_mse:
                    best_cv_mse = cv_mse
                    best_params = params.copy()
    
    # Final fit with best parameters
    final_params = best_params
    K_final = _compute_kernel(X_norm, X_norm, final_params)
    coeffs_final = _kernel_ridge_solve(K_final, y_centered, final_params['alpha'])
    
    def predict(X):
        X_test_norm = (X - x_min) / x_range
        K_test = _compute_kernel(X_test_norm, X_norm, final_params)
        return K_test @ coeffs_final + y_mean
    
    return predict
# EVOLVE-BLOCK-END
