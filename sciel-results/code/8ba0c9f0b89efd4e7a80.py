"""Symbolic regression via composite kernel ridge regression with adaptive hyperparameter search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _rbf_kernel(X1, X2, bandwidth):
    """Compute RBF kernel matrix between X1 and X2."""
    sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
    sq_dists = np.maximum(sq_dists, 0)
    return np.exp(-sq_dists / (2 * bandwidth**2))


def _poly_kernel(X1, X2, degree, gamma, coef0):
    """Compute polynomial kernel matrix."""
    return (gamma * X1 @ X2.T + coef0) ** degree


def _periodic_kernel(X1, X2, length_scale, period):
    """Compute periodic kernel matrix."""
    diff = X1[:, :, None] - X2[:, None, :]
    sq_dists = np.sum(np.sin(np.pi * diff / period)**2, axis=1)
    return np.exp(-2 * sq_dists / length_scale**2)


def _composite_kernel(X1, X2, bandwidth, poly_degree, poly_gamma, poly_coef0, 
                       periodic_ls, periodic_period, w_rbf, w_poly, w_periodic):
    """Compute composite kernel matrix."""
    K_rbf = _rbf_kernel(X1, X2, bandwidth)
    K_poly = _poly_kernel(X1, X2, poly_degree, poly_gamma, poly_coef0)
    K_periodic = _periodic_kernel(X1, X2, periodic_ls, periodic_period)
    return w_rbf * K_rbf + w_poly * K_poly + w_periodic * K_periodic


def _kernel_ridge_solve(K_train, y, lam):
    """Solve kernel ridge regression: (K + lam*I) alpha = y."""
    n = K_train.shape[0]
    return np.linalg.solve(K_train + lam * np.eye(n), y)


def _kernel_ridge_predict(K_test_train, alpha):
    """Predict using kernel ridge regression."""
    return K_test_train @ alpha


def fit(X_train, y_train):
    """Fit using composite kernel ridge regression with hyperparameter search."""
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Normalize to [0, 1] range for stable kernel computation
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)
    X_range = X_max - X_min
    X_range[X_range < 1e-12] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Hyperparameter grid
    bandwidths = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    poly_degrees = [2, 3]
    poly_gammas = [0.5, 1.0, 2.0]
    poly_coef0s = [0.5, 1.0]
    periodic_lss = [0.5, 1.0]
    periodic_periods = [1.0, 2.0]
    weight_combos = [
        (1.0, 0.0, 0.0),   # RBF only
        (0.0, 1.0, 0.0),   # Poly only
        (0.0, 0.0, 1.0),   # Periodic only
        (0.7, 0.3, 0.0),   # RBF + Poly
        (0.5, 0.3, 0.2),   # RBF + Poly + Periodic
        (0.6, 0.2, 0.2),   # RBF + Poly + Periodic
    ]
    lam_values = [1e-4, 1e-3, 1e-2, 0.1, 1.0]
    
    # 2-fold CV for model selection
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)
    half = n // 2
    fold1_idx = idx[:half]
    fold2_idx = idx[half:]
    
    best_score = np.inf
    best_params = None
    best_alpha = None
    
    # Precompute K for train folds to save time
    K_fold1_train = None
    K_fold2_train = None
    K_fold1_test = None
    K_fold2_test = None
    
    for bw in bandwidths:
        if time.time() > deadline - 30:
            break
        for pdeg in poly_degrees:
            if time.time() > deadline - 30:
                break
            for pgamma in poly_gammas:
                if time.time() > deadline - 30:
                    break
                for pcoef in poly_coef0s:
                    if time.time() > deadline - 30:
                        break
                    for pls in periodic_lss:
                        if time.time() > deadline - 30:
                            break
                        for pper in periodic_periods:
                            if time.time() > deadline - 30:
                                break
                            for w_rbf, w_poly, w_periodic in weight_combos:
                                if time.time() > deadline - 15:
                                    break
                                
                                # Compute kernels for fold 1
                                K1_train = _composite_kernel(
                                    X_norm[fold1_idx], X_norm[fold1_idx],
                                    bw, pdeg, pgamma, pcoef, pls, pper,
                                    w_rbf, w_poly, w_periodic
                                )
                                K1_test = _composite_kernel(
                                    X_norm[fold2_idx], X_norm[fold1_idx],
                                    bw, pdeg, pgamma, pcoef, pls, pper,
                                    w_rbf, w_poly, w_periodic
                                )
                                
                                # Compute kernels for fold 2
                                K2_train = _composite_kernel(
                                    X_norm[fold2_idx], X_norm[fold2_idx],
                                    bw, pdeg, pgamma, pcoef, pls, pper,
                                    w_rbf, w_poly, w_periodic
                                )
                                K2_test = _composite_kernel(
                                    X_norm[fold1_idx], X_norm[fold2_idx],
                                    bw, pdeg, pgamma, pcoef, pls, pper,
                                    w_rbf, w_poly, w_periodic
                                )
                                
                                for lam in lam_values:
                                    if time.time() > deadline - 10:
                                        break
                                    try:
                                        # Fold 1
                                        alpha1 = _kernel_ridge_solve(K1_train, y[fold1_idx], lam)
                                        pred1 = _kernel_ridge_predict(K1_test, alpha1)
                                        mse1 = np.mean((pred1 - y[fold2_idx])**2)
                                        
                                        # Fold 2
                                        alpha2 = _kernel_ridge_solve(K2_train, y[fold2_idx], lam)
                                        pred2 = _kernel_ridge_predict(K2_test, alpha2)
                                        mse2 = np.mean((pred2 - y[fold1_idx])**2)
                                        
                                        score = (mse1 + mse2) / 2.0
                                        
                                        if score < best_score:
                                            best_score = score
                                            best_params = (bw, pdeg, pgamma, pcoef, pls, pper, 
                                                           w_rbf, w_poly, w_periodic, lam)
                                            
                                    except Exception:
                                        continue
    
    # If no good params found, use default RBF
    if best_params is None:
        best_params = (0.5, 2, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1e-3)
    
    bw, pdeg, pgamma, pcoef, pls, pper, w_rbf, w_poly, w_periodic, lam = best_params
    
    # Final solve on all training data
    K_full = _composite_kernel(
        X_norm, X_norm, bw, pdeg, pgamma, pcoef, pls, pper,
        w_rbf, w_poly, w_periodic
    )
    best_alpha = _kernel_ridge_solve(K_full, y, lam)
    
    # Store for prediction
    X_min_stored = X_min
    X_range_stored = X_range
    best_params_stored = best_params
    best_alpha_stored = best_alpha
    d_stored = d
    
    def predict(X_new):
        X_new = np.asarray(X_new, dtype=np.float64)
        X_new_norm = (X_new - X_min_stored) / X_range_stored
        X_new_norm = np.clip(X_new_norm, -0.5, 1.5)  # Clip to handle out-of-range points
        
        bw, pdeg, pgamma, pcoef, pls, pper, w_rbf, w_poly, w_periodic, lam = best_params_stored
        
        K_test = _composite_kernel(
            X_new_norm, X_norm, bw, pdeg, pgamma, pcoef, pls, pper,
            w_rbf, w_poly, w_periodic
        )
        return _kernel_ridge_predict(K_test, best_alpha_stored)
    
    return predict
# EVOLVE-BLOCK-END
