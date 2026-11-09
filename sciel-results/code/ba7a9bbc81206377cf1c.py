"""Kernel ridge regression with composite kernel for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import solve

def fit(X_train, y_train):
    n, d = X_train.shape
    
    # Normalize inputs to [0, 1] range for better kernel scaling
    # We use the known ranges from the problem description if possible, 
    # but since we don't know which equation is which, we use data-driven min/max 
    # expanded slightly to cover test points.
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    # Expand range slightly to ensure test points (which are in the same uniform range) 
    # are not extrapolated too far if they land on the boundary.
    # The problem states uniform ranges. Using min/max of train is safe if train is representative.
    # Let's add a small padding to min/max to be safe.
    padding = 0.1 * (X_max - X_min + 1e-8)
    X_min = X_min - padding
    X_max = X_max + padding
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X_norm = (X_train - X_min) / X_range
    
    def compute_kernel(X1, X2, length_scale=1.0, poly_degree=3, poly_gamma=1.0, poly_coef0=1.0, period=1.0):
        """Compute composite kernel matrix: RBF + Polynomial + Periodic."""
        
        # 1. RBF Kernel
        # Efficient distance calculation
        sq_dists = (np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T)
        sq_dists = np.maximum(sq_dists, 0)
        K_rbf = np.exp(-sq_dists / (2 * length_scale**2))
        
        # 2. Polynomial Kernel
        # Standard polynomial: (gamma * <x, z> + coef0)^degree
        K_poly = (poly_gamma * (X1 @ X2.T) + poly_coef0) ** poly_degree
        
        # 3. Periodic Kernel
        # K_periodic(x, z) = exp(-gamma * sin^2(pi * (x - z) / period))
        # This helps capture oscillatory patterns if present.
        # We use a simple sine-based periodic kernel.
        # Note: For multi-dim, we sum the periodic components per dimension or use a global one.
        # A common approach is exp(-gamma * sum((sin(pi*(x-z)/period))^2) )
        # Or simpler: product of 1D periodic kernels? 
        # Let's use a sum of 1D periodic kernels for each dimension, scaled.
        K_per = np.zeros_like(K_rbf)
        for i in range(d):
            diff = X1[:, i:i+1] - X2[:, i:i+1].T
            # sin^2(pi * diff / period)
            term = np.sin(np.pi * diff / period) ** 2
            # Sum across dimensions? Or average?
            # Let's add the 1D periodic kernel for each dim, scaled down.
            K_per += np.exp(-1.0 * term)
        # Normalize K_per to have similar magnitude to K_rbf (which is in [0,1])
        # Max value of K_per is d (if all diffs are 0). Min is 0.
        # Let's scale it by 1/d to keep it in [0,1] range roughly.
        K_per = K_per / d

        # Combine kernels with fixed weights initially, or optimize weights?
        # Let's use fixed weights for now to keep grid search small.
        # RBF is good for local structure. Poly for global trends. Periodic for cycles.
        # Weights: RBF 1.0, Poly 0.5, Periodic 0.5
        K_total = 1.0 * K_rbf + 0.5 * K_poly + 0.5 * K_per
        
        return K_total

    def loocv_score(K, y, alpha):
        """
        Calculate LOOCV score for Kernel Ridge Regression.
        Score = 1/N * sum((y_i - y_hat_i_loo)^2)
        y_hat_i_loo = y_i - K_alpha_i / (1 + (K^{-1})_{ii}) ... wait, formula is:
        y_hat_i_loo = y_i - (K^{-1})_{ii}^{-1} * (y_i - y_hat_i) ? No.
        
        Correct Formula:
        Let K be the kernel matrix + lambda*I.
        alpha = K^{-1} y
        y_hat = K_train alpha
        y_hat_loo_i = y_i - (K^{-1})_{ii}^{-1} (y_i - y_hat_i)
        Actually, the efficient formula is:
        y_hat_loo_i = y_i - (K + lambda I)^{-1}_{ii}^{-1} * (y_i - y_hat_i)
        Wait, let's use the standard identity:
        y_hat_loo_i = y_i - \frac{y_i - y_hat_i}{1 - (K + \lambda I)^{-1}_{ii}}
        
        We need the diagonal of the inverse of (K + lambda I).
        """
        try:
            # We need diag of inv(K_reg)
            # K_reg is positive definite
            inv_K_reg = np.linalg.inv(K_reg)
            diag_inv = np.diag(inv_K_reg)
            # y_hat is K_train @ alpha, where K_train is the raw kernel part?
            # In KRR, y_hat = K_raw @ alpha.
            # The formula uses the inverse of the regularized matrix.
            # y_hat_loo_i = y_i - (y_i - y_hat_i) / (1 - diag_inv[i])
            # Let's verify.
            # y_hat_i = sum_j K_ij alpha_j
            # y_hat_loo_i = y_i - (K^{-1})_{ii}^{-1} (y_i - y_hat_i) is NOT correct.
            
            # Correct derivation:
            # alpha_loo = alpha - (K^{-1} e_i) (y_i - y_hat_i) / (1 + e_i^T K^{-1} e_i) ? No.
            # Standard result:
            # y_hat_loo_i = y_i - \frac{y_i - y_hat_i}{1 - h_{ii}}
            # where h_{ii} is the i-th diagonal element of the hat matrix H = K (K + \lambda I)^{-1} K^T ?
            # For KRR, H = K (K + \lambda I)^{-1}.
            # The diagonal of H is not simply related to diag of inv(K+lambda I) in a trivial way without computing H.
            # However, h_ii = 1 - (K + \lambda I)^{-1}_{ii} * \lambda ? No.
            
            # Let's just compute the hat matrix diagonal if n is small enough?
            # n=200. Inverting 200x200 is fast.
            # H = K_raw @ (K_raw + lambda I)^{-1}
            # diag_H = np.diag(H)
            # y_hat_loo_i = y_i - (y_i - y_hat_i) / (1 - diag_H[i])
            
            # To avoid forming full H, we can use:
            # diag_H = 1 - lambda * diag_inv_K_reg ? No.
            # Let's just compute H diagonal.
            # H = K_raw @ alpha_diag? No.
            # alpha = (K_raw + lambda I)^{-1} y
            # y_hat = K_raw alpha
            # y_hat_loo_i = y_i - (y_i - y_hat_i) / (1 - h_ii)
            
            # We need h_ii.
            # h_ii = sum_j K_ij alpha_j / y_i ? No.
            # h_ii = (K (K + \lambda I)^{-1})_{ii}
            
            # Let's compute the full matrix H if it's cheap.
            # 200x200 matrix mult is ~8M ops. Very fast.
            H = K_raw @ inv_K_reg  # This is the hat matrix for KRR
            diag_H = np.diag(H)
            
            # Avoid division by zero
            denom = 1.0 - diag_H
            denom[denom < 1e-8] = 1e-8
            
            y_hat_loo = y - (y - y_hat) / denom
            
            mse = np.mean((y_hat_loo - y)**2)
            return mse
        except:
            return np.inf

    # Hyperparameter grid
    # We will search over length_scale, poly_degree, poly_gamma, period
    # To keep it fast, we limit the grid size.
    
    best_cv_score = np.inf
    best_params = {
        'length_scale': 0.5,
        'poly_degree': 3,
        'poly_gamma': 1.0,
        'poly_coef0': 1.0,
        'period': 2.0
    }
    
    # Regularization parameter lambda
    # For noiseless data, we want small lambda, but not zero to avoid singularity.
    # We'll fix lambda to a small value, e.g., 1e-4 or 1e-6.
    # Or we can include it in the search. Let's fix it to 1e-5 for stability.
    lambda_reg = 1e-5
    
    # Grid values
    length_scales = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    poly_degrees = [2, 3, 4]
    poly_gammas = [0.5, 1.0, 2.0]
    periods = [1.0, 2.0, 4.0]
    
    # Pre-compute X_norm
    # We will iterate through a subset of the grid to save time.
    # Total combos: 6 * 3 * 3 * 3 = 162.
    # Each combo involves:
    # 1. Kernel matrix computation (200x200)
    # 2. Solve (200x200)
    # 3. LOOCV calculation (Matrix mult 200x200)
    # This should be well within time limits.
    
    for ls in length_scales:
        for pd in poly_degrees:
            for pg in poly_gammas:
                for per in periods:
                    # Compute Kernel
                    K_raw = compute_kernel(X_norm, X_norm, ls, pd, 1.0, 1.0, per)
                    # Note: I fixed poly_coef0 to 1.0 in the loop to reduce grid size, 
                    # or I can add it. Let's add poly_coef0 = 1.0 and 2.0.
                    # Actually, let's keep the grid manageable.
                    
                    K_reg = K_raw + lambda_reg * np.eye(n)
                    
                    try:
                        alpha = solve(K_reg, y_train, assume_a='pos')
                        y_hat = K_raw @ alpha
                        
                        # Calculate LOOCV
                        # We need inv(K_reg) for the hat matrix
                        inv_K_reg = np.linalg.inv(K_reg)
                        H = K_raw @ inv_K_reg
                        diag_H = np.diag(H)
                        
                        denom = 1.0 - diag_H
                        # If denom is close to 0, the point is highly influential, LOOCV is unstable.
                        # We handle this by clamping.
                        denom_clamped = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
                        
                        y_hat_loo = y_train - (y_train - y_hat) / denom_clamped
                        cv_mse = np.mean((y_hat_loo - y_train)**2)
                        
                        if cv_mse < best_cv_score:
                            best_cv_score = cv_mse
                            best_params = {
                                'length_scale': ls,
                                'poly_degree': pd,
                                'poly_gamma': pg,
                                'poly_coef0': 1.0,
                                'period': per
                            }
                    except:
                        continue

    # Final fit with best parameters
    K_raw = compute_kernel(X_norm, X_norm, 
                           best_params['length_scale'], 
                           best_params['poly_degree'], 
                           best_params['poly_gamma'], 
                           best_params['poly_coef0'], 
                           best_params['period'])
    K_reg = K_raw + lambda_reg * np.eye(n)
    alpha = solve(K_reg, y_train, assume_a='pos')
    
    def predict(X):
        X = np.asarray(X, dtype=float)
        X_norm_test = (X - X_min) / X_range
        K_test = compute_kernel(X_norm_test, X_norm, 
                                best_params['length_scale'], 
                                best_params['poly_degree'], 
                                best_params['poly_gamma'], 
                                best_params['poly_coef0'], 
                                best_params['period'])
        return K_test @ alpha
    
    return predict
# EVOLVE-BLOCK-END
