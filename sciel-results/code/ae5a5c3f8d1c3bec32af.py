"""Composite kernel ridge regression for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import solve
import warnings
warnings.filterwarnings('ignore')

def fit(X_train, y_train):
    n, d = X_train.shape
    
    # Normalize inputs using data-driven ranges with padding
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    padding = 0.1 * (X_max - X_min + 1e-8)
    X_min = X_min - padding
    X_max = X_max + padding
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X_norm = (X_train - X_min) / X_range
    
    # Pre-compute norms for efficient distance calculations
    X_norm_sq = np.sum(X_norm**2, axis=1)
    
    def compute_kernel(X1, X2, length_scale, poly_degree, poly_gamma, period, 
                        w_rbf=1.0, w_poly=0.5, w_per=0.5):
        """Compute composite kernel matrix: RBF + Polynomial + Periodic."""
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        
        # 1. RBF Kernel
        sq_dists = X_norm_sq[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
        sq_dists = np.maximum(sq_dists, 0)
        K_rbf = np.exp(-sq_dists / (2 * length_scale**2))
        
        # 2. Polynomial Kernel
        K_poly = (poly_gamma * (X1 @ X2.T) + 1.0) ** poly_degree
        
        # 3. Periodic Kernel (sum of 1D periodic kernels per dimension)
        K_per = np.zeros((n1, n2))
        for i in range(d):
            diff = X1[:, i:i+1] - X2[:, i:i+1].T
            term = np.sin(np.pi * diff / period) ** 2
            K_per += np.exp(-1.0 * term)
        K_per = K_per / d
        
        # Combine kernels
        K_total = w_rbf * K_rbf + w_poly * K_poly + w_per * K_per
        
        return K_total

    # Hyperparameter grid search
    # We search over: length_scale, poly_degree, poly_gamma, period, lambda_reg
    # Keep grid small enough to fit in time limit (300s total for all 10 equations)
    
    length_scales = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    poly_degrees = [2, 3, 4]
    poly_gammas = [0.5, 1.0, 2.0]
    periods = [1.0, 2.0, 4.0]
    lambdas = [1e-6, 1e-5, 1e-4, 1e-3]
    
    best_cv_score = np.inf
    best_params = {
        'length_scale': 0.5,
        'poly_degree': 3,
        'poly_gamma': 1.0,
        'period': 2.0,
        'lambda': 1e-5
    }
    
    # To keep computation manageable, we do a two-stage search:
    # Stage 1: Coarse grid
    # Stage 2: Fine grid around best
    
    # Stage 1: Coarse grid
    for ls in length_scales:
        for pd in poly_degrees:
            for pg in poly_gammas:
                for per in periods:
                    for lam in lambdas:
                        K_raw = compute_kernel(X_norm, X_norm, ls, pd, pg, per)
                        K_reg = K_raw + lam * np.eye(n)
                        
                        try:
                            # Solve for alpha
                            alpha = solve(K_reg, y_train, assume_a='pos')
                            y_hat = K_raw @ alpha
                            
                            # LOOCV via hat matrix
                            # H = K_raw @ inv(K_reg)
                            # To avoid full inverse, we can solve K_reg X = I
                            # But for n=200, full inverse is fast
                            inv_K_reg = np.linalg.inv(K_reg)
                            H = K_raw @ inv_K_reg
                            diag_H = np.diag(H)
                            
                            denom = 1.0 - diag_H
                            denom_clamped = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
                            
                            y_hat_loo = y_train - (y_train - y_hat) / denom_clamped
                            cv_mse = np.mean((y_hat_loo - y_train)**2)
                            
                            if cv_mse < best_cv_score:
                                best_cv_score = cv_mse
                                best_params = {
                                    'length_scale': ls,
                                    'poly_degree': pd,
                                    'poly_gamma': pg,
                                    'period': per,
                                    'lambda': lam
                                }
                        except:
                            continue
    
    # Stage 2: Fine grid around best parameters
    best_ls = best_params['length_scale']
    best_pd = best_params['poly_degree']
    best_pg = best_params['poly_gamma']
    best_per = best_params['period']
    best_lam = best_params['lambda']
    
    # Refine around best
    ls_fine = [best_ls * 0.5, best_ls * 0.75, best_ls, best_ls * 1.25, best_ls * 1.5, best_ls * 2.0]
    ls_fine = [max(0.01, x) for x in ls_fine]
    pg_fine = [best_pg * 0.5, best_pg * 0.75, best_pg, best_pg * 1.25, best_pg * 1.5, best_pg * 2.0]
    pg_fine = [max(0.01, x) for x in pg_fine]
    per_fine = [best_per * 0.5, best_per * 0.75, best_per, best_per * 1.25, best_per * 1.5, best_per * 2.0]
    per_fine = [max(0.1, x) for x in per_fine]
    lam_fine = [best_lam * 0.1, best_lam * 0.5, best_lam, best_lam * 2.0, best_lam * 5.0]
    
    for ls in ls_fine:
        for pg in pg_fine:
            for per in per_fine:
                for lam in lam_fine:
                    K_raw = compute_kernel(X_norm, X_norm, ls, best_pd, pg, per)
                    K_reg = K_raw + lam * np.eye(n)
                    
                    try:
                        alpha = solve(K_reg, y_train, assume_a='pos')
                        y_hat = K_raw @ alpha
                        
                        inv_K_reg = np.linalg.inv(K_reg)
                        H = K_raw @ inv_K_reg
                        diag_H = np.diag(H)
                        
                        denom = 1.0 - diag_H
                        denom_clamped = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
                        
                        y_hat_loo = y_train - (y_train - y_hat) / denom_clamped
                        cv_mse = np.mean((y_hat_loo - y_train)**2)
                        
                        if cv_mse < best_cv_score:
                            best_cv_score = cv_mse
                            best_params = {
                                'length_scale': ls,
                                'poly_degree': best_pd,
                                'poly_gamma': pg,
                                'period': per,
                                'lambda': lam
                            }
                    except:
                        continue
    
    # Final fit with best parameters
    K_raw = compute_kernel(X_norm, X_norm, 
                           best_params['length_scale'], 
                           best_params['poly_degree'], 
                           best_params['poly_gamma'], 
                           best_params['period'])
    K_reg = K_raw + best_params['lambda'] * np.eye(n)
    alpha = solve(K_reg, y_train, assume_a='pos')
    
    # Store training data for predict
    X_norm_train = X_norm.copy()
    X_min_final = X_min.copy()
    X_range_final = X_range.copy()
    d_final = d
    ls_final = best_params['length_scale']
    pd_final = best_params['poly_degree']
    pg_final = best_params['poly_gamma']
    per_final = best_params['period']
    
    def predict(X):
        X = np.asarray(X, dtype=float)
        X_norm_test = (X - X_min_final) / X_range_final
        
        n_test = X_norm_test.shape[0]
        
        # RBF
        X_test_sq = np.sum(X_norm_test**2, axis=1)
        sq_dists = X_test_sq[:, None] + X_norm_sq[None, :] - 2 * (X_norm_test @ X_norm_train.T)
        sq_dists = np.maximum(sq_dists, 0)
        K_rbf = np.exp(-sq_dists / (2 * ls_final**2))
        
        # Poly
        K_poly = (pg_final * (X_norm_test @ X_norm_train.T) + 1.0) ** pd_final
        
        # Periodic
        K_per = np.zeros((n_test, n))
        for i in range(d_final):
            diff = X_norm_test[:, i:i+1] - X_norm_train[:, i:i+1].T
            term = np.sin(np.pi * diff / per_final) ** 2
            K_per += np.exp(-1.0 * term)
        K_per = K_per / d_final
        
        K_test = 1.0 * K_rbf + 0.5 * K_poly + 0.5 * K_per
        
        return K_test @ alpha
    
    return predict
# EVOLVE-BLOCK-END
