import numpy as np
import time

def fit(X_train, y_train):
    t0 = time.time()
    deadline = t0 + 285  # Safety margin
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    x = X[:, 0].copy()
    
    # Handle trivial case
    if np.allclose(y, y[0]):
        return lambda X: np.full(len(X), y[0])
    
    # Sort for potential spline use later, but we'll use KRR mainly
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # We will use Kernel Ridge Regression (KRR) with a Gaussian kernel.
    # The key is to select the bandwidth (sigma) and regularization (alpha) well.
    # Since the input is 1D and ranges are known-ish (from the contract), we can normalize.
    
    # Normalize x to [0, 1] or [-1, 1] to make sigma selection more robust
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max == x_min:
        x_norm = np.zeros_like(x)
    else:
        x_norm = (x - x_min) / (x_max - x_min)
    
    # Center and scale y for numerical stability
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-12:
        return lambda X: np.full(len(X), y_mean)
    y_norm = (y - y_mean) / y_std
    
    # Candidate sigmas (in normalized space)
    # Range from very narrow to very broad
    sigmas = np.logspace(-2.5, 1.5, 15) # 0.003 to ~31
    
    # Candidate alphas (regularization)
    # log spaced
    alphas = np.logspace(-8, 8, 17) # 1e-8 to 1e8
    
    # Cross-Validation
    # Use 5-fold CV
    k_folds = 5
    indices = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    
    fold_size = n // k_folds
    folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(k_folds)]
    if n % k_folds != 0:
        # Append remainder to last fold
        folds[-1] = np.concatenate([folds[-1], indices[k_folds*fold_size:]])
    
    best_score = np.inf
    best_sigma = None
    best_alpha = None
    
    # Precompute pairwise distances for all x_norm
    # This is O(n^2) which is 200^2 = 40k, very fast.
    x_norm_col = x_norm[:, np.newaxis]
    x_norm_row = x_norm[np.newaxis, :]
    dist_sq = (x_norm_col - x_norm_row)**2
    
    # We will iterate over sigmas and alphas.
    # To be fast, we can precompute the base kernel matrix for each sigma?
    # K = exp(-dist_sq / (2*sigma^2))
    
    # Let's optimize the loop.
    # For each sigma, compute K.
    # Then for each alpha, solve (K + alpha*I) w = y_norm.
    # But CV requires refitting.
    
    # Alternative: Use LOOCV formula for KRR?
    # LOOCV error for KRR: e_i = (y_i - f_i) / (1 + (K + alpha I)^{-1}_{ii})
    # This is O(n^3) to invert, but we can do it once per (sigma, alpha) pair.
    # n=200, 15 sigmas * 17 alphas = 255 pairs.
    # 255 * 200^3 = 255 * 8e6 = 2e9 flops. Might be too slow in Python/NumPy if not optimized.
    # Let's stick to 5-fold CV but optimize the linear solve.
    
    # Actually, for each sigma, we can precompute the eigen-decomposition of K.
    # K = V diag(lambda) V^T
    # (K + alpha I)^{-1} = V diag(1/(lambda+alpha)) V^T
    # This allows fast solves for different alphas.
    
    for sigma in sigmas:
        # Check time
        if time.time() > deadline - 10:
            break
            
        # Compute Kernel
        # sigma is in normalized space
        # Standard Gaussian: exp(-||x-y||^2 / (2*sigma^2))
        K = np.exp(-dist_sq / (2 * sigma**2))
        
        # Eigen decomposition
        # K is symmetric
        try:
            eigvals, eigvecs = np.linalg.eigh(K)
        except np.linalg.LinAlgError:
            continue
        
        # Sort eigenvalues in descending order (eigh returns ascending)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Project y_norm onto eigenbasis
        y_proj = eigvecs.T @ y_norm
        
        # Now we can compute predictions for any alpha quickly
        # w = (K + alpha I)^{-1} y
        # In eigenbasis: w_proj = y_proj / (eigvals + alpha)
        # f = K w = sum_i (eigvals_i / (eigvals_i + alpha)) y_proj_i v_i
        # f = V diag( eigvals / (eigvals + alpha) ) V^T y
        
        # For CV, we need to leave out points.
        # This is harder with the eigen trick because leaving out points changes K.
        # However, since n is small, let's just do the 5-fold CV with direct solves but reuse the eigen decomp if possible?
        # No, leaving out points changes the matrix size.
        
        # Let's just do standard 5-fold CV with np.linalg.lstsq or solve.
        # It's only 5 folds * 17 alphas * 15 sigmas = 1275 solves.
        # Each solve is 195x195. Very fast.
        
        best_alpha_for_sigma = None
        best_cv_err_for_sigma = np.inf
        
        for alpha in alphas:
            cv_errors = []
            for i in range(k_folds):
                val_idx = folds[i]
                tr_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])
                
                x_tr = x_norm[tr_idx]
                y_tr = y_norm[tr_idx]
                x_val = x_norm[val_idx]
                y_val = y_norm[val_idx]
                
                # Compute kernel between train and val
                # K_tr_val = exp(-(x_tr[:, None] - x_val[None, :])**2 / (2*sigma^2))
                diff_tv = x_tr[:, np.newaxis] - x_val[np.newaxis, :]
                K_tv = np.exp(-diff_tv**2 / (2 * sigma**2))
                
                # K_tr_tr
                diff_tt = x_tr[:, np.newaxis] - x_tr[np.newaxis, :]
                K_tt = np.exp(-diff_tt**2 / (2 * sigma**2))
                
                # Solve (K_tt + alpha*I) w = y_tr
                try:
                    w = np.linalg.solve(K_tt + alpha * np.eye(len(tr_idx)), y_tr)
                    y_pred = K_tv @ w
                    err = np.mean((y_pred - y_val)**2)
                    cv_errors.append(err)
                except np.linalg.LinAlgError:
                    cv_errors.append(np.inf)
                    break
                
            mean_cv_err = np.mean(cv_errors)
            if mean_cv_err < best_cv_err_for_sigma:
                best_cv_err_for_sigma = mean_cv_err
                best_alpha_for_sigma = alpha
                
        if best_cv_err_for_sigma < best_score:
            best_score = best_cv_err_for_sigma
            best_sigma = sigma
            best_alpha = best_alpha_for_sigma
    
    # If no good parameters found, use defaults
    if best_sigma is None:
        best_sigma = 1.0
        best_alpha = 1.0
        
    # Final fit with best parameters
    # Use all data
    K_full = np.exp(-dist_sq / (2 * best_sigma**2))
    try:
        w_final = np.linalg.solve(K_full + best_alpha * np.eye(n), y_norm)
    except np.linalg.LinAlgError:
        # Fallback to poly
        w_poly, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), x_norm, x_norm**2]), y_norm, rcond=None)
        def predict_poly(X):
            x_p = (X[:, 0] - x_min) / (x_max - x_min)
            F = np.column_stack([np.ones(len(x_p)), x_p, x_p**2])
            return F @ w_poly * y_std + y_mean
        return predict_poly

    def predict(X):
        x_p = np.asarray(X, dtype=np.float64)[:, 0]
        # Normalize
        if x_max == x_min:
            x_norm_p = np.zeros_like(x_p)
        else:
            x_norm_p = (x_p - x_min) / (x_max - x_min)
        
        # Compute kernel between test and train
        diff_tp = x_norm_p[:, np.newaxis] - x_norm[np.newaxis, :]
        K_tp = np.exp(-diff_tp**2 / (2 * best_sigma**2))
        
        y_pred_norm = K_tp @ w_final
        y_pred = y_pred_norm * y_std + y_mean
        return y_pred

    return predict
