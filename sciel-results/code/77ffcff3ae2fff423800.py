import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = 110.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = min(x.size, y.size)
    if n == 0:
        return np.zeros(6, dtype=np.float64)
    
    x = x[:n]
    y = y[:n]
    
    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size
    
    if n == 0:
        return np.zeros(6, dtype=np.float64)
    
    # If n is very small, just do a linear fit
    if n < 20:
        coeff = np.zeros(6)
        if n >= 2:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        return coeff

    # --- Kernel-weighted denoising to estimate E[Y|X] ---
    # This reduces noise in the pseudo-labels before polynomial fitting
    def kernel_denoise(x_data, y_data, bandwidth=None):
        """Estimate E[Y|X] using Gaussian kernel weighting."""
        if bandwidth is None:
            # Adaptive bandwidth based on data spread
            bandwidth = 0.5 * (np.max(x_data) - np.min(x_data)) / np.sqrt(len(x_data))
            bandwidth = max(bandwidth, 1e-6)
        
        y_denoised = np.zeros_like(y_data)
        for i in range(len(x_data)):
            # Gaussian kernel weights
            dists = np.abs(x_data - x_data[i])
            weights = np.exp(-0.5 * (dists / bandwidth) ** 2)
            # Exclude self-weight to avoid trivial solution
            weights[i] = 0.0
            total_w = np.sum(weights)
            if total_w > 1e-12:
                y_denoised[i] = np.sum(weights * y_data) / total_w
            else:
                y_denoised[i] = y_data[i]
        return y_denoised

    # Try multiple bandwidths for denoising
    x_range = np.max(x) - np.min(x)
    if x_range < 1e-10:
        x_range = 1.0
    bandwidths = [x_range * f for f in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]]
    
    # Choose best bandwidth via simple CV on denoised data
    k_cv = 5
    indices_cv = np.arange(n)
    np.random.seed(123)
    np.random.shuffle(indices_cv)
    folds_cv = np.array_split(indices_cv, k_cv)
    
    best_bw = bandwidths[0]
    best_cv_score = float('inf')
    
    for bw in bandwidths:
        if time.time() > deadline:
            break
        cv_scores = []
        for i in range(k_cv):
            val_idx = folds_cv[i]
            train_idx = np.concatenate([folds_cv[j] for j in range(k_cv) if j != i])
            
            x_tr = x[train_idx]
            y_tr = y[train_idx]
            x_vl = x[val_idx]
            y_vl = y[val_idx]
            
            # Denoise training data
            y_tr_denoised = kernel_denoise(x_tr, y_tr, bandwidth=bw)
            
            # Quick polynomial fit on denoised data
            try:
                coeff_quick = np.polynomial.polynomial.polyfit(x_tr, y_tr_denoised, 3)
                y_pred = np.polynomial.polynomial.polyval(x_vl, coeff_quick)
                mse = np.mean((y_vl - y_pred) ** 2)
                cv_scores.append(mse)
            except:
                cv_scores.append(float('inf'))
        
        mean_cv = np.mean(cv_scores)
        if mean_cv < best_cv_score:
            best_cv_score = mean_cv
            best_bw = bw
    
    # Apply denoising with best bandwidth
    y_denoised = kernel_denoise(x, y, bandwidth=best_bw)
    
    # Define search space
    degrees = [1, 2, 3, 4, 5]
    lambdas = [1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4]
    
    # K-Fold Cross Validation
    k = 5
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    
    best_score = float('inf')
    best_degree = 1
    best_lambda = 0.0
    
    # Helper to build Vandermonde matrix
    def vandermonde(x, deg):
        return np.vander(x, N=deg+1, increasing=True)
    
    # Helper to solve ridge regression
    def solve_ridge(A, y, lam):
        if lam == 0:
            coeff, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            return coeff
        else:
            sqrt_lam = np.sqrt(lam)
            m, p = A.shape
            A_aug = np.vstack([A, sqrt_lam * np.eye(p)])
            y_aug = np.concatenate([y, np.zeros(p)])
            coeff, _, _, _ = np.linalg.lstsq(A_aug, y_aug, rcond=None)
            return coeff

    for deg in degrees:
        if time.time() > deadline:
            break
            
        for lam in lambdas:
            if time.time() > deadline:
                break
                
            cv_mses = []
            for i in range(k):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
                
                x_train = x[train_idx]
                y_train = y_denoised[train_idx]
                x_val = x[val_idx]
                y_val = y[val_idx]  # Use original y for evaluation
                
                # Build training matrix
                A_train = vandermonde(x_train, deg)
                
                # Solve
                coeff = solve_ridge(A_train, y_train, lam)
                
                # Evaluate on validation
                A_val = vandermonde(x_val, deg)
                y_pred = A_val @ coeff
                mse = np.mean((y_val - y_pred) ** 2)
                cv_mses.append(mse)
            
            mean_mse = np.mean(cv_mses)
            
            if mean_mse < best_score:
                best_score = mean_mse
                best_degree = deg
                best_lambda = lam

    # Refit on entire training set with best parameters
    A_full = vandermonde(x, best_degree)
    final_coeff = solve_ridge(A_full, y_denoised, best_lambda)
    
    # Ensure output is 6 coefficients in ascending powers
    result = np.zeros(6, dtype=np.float64)
    result[:best_degree+1] = final_coeff
    
    # Check for non-finite results
    if not np.all(np.isfinite(result)):
        # Fallback to linear fit
        result = np.zeros(6, dtype=np.float64)
        result[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        if not np.all(np.isfinite(result)):
            result = np.zeros(6, dtype=np.float64)
            
    return result
