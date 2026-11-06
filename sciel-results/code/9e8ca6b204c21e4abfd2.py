"""Improved program: centered polynomial fit with dense regularization search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 115  # 35s safety margin before 150s kill
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Basic sanity: if inputs are degenerate, return zeros
    if len(x) < 6 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.zeros(6)
    
    n = len(x)
    
    # Center and scale x to improve conditioning
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_scaled = (x - x_mean) / x_std
    
    # Build Vandermonde matrix for scaled x
    V_scaled = np.vander(x_scaled, N=6, increasing=True)
    
    # Dense alpha search with emphasis on regularization
    # Since validation includes unseen regime (F=10), we want robustness
    alphas = np.logspace(-8, 4, 120)  # from 1e-8 to 1e4
    
    # Cross-validation: use multiple splits for more robust selection
    n_cv = min(5, max(2, n // 50))  # number of CV folds
    indices = np.arange(n)
    np.random.seed(42)
    indices = np.random.permutation(indices)
    
    best_coeff_scaled = None
    best_cv_score = np.inf
    
    for alpha in alphas:
        if time.time() > deadline - 15:
            break
        
        cv_scores = []
        for fold in range(n_cv):
            # Create fold split
            val_start = fold * (n // n_cv)
            val_end = (fold + 1) * (n // n_cv) if fold < n_cv - 1 else n
            val_idx = indices[val_start:val_end]
            trn_idx = np.concatenate([indices[:val_start], indices[val_end:]])
            
            if len(trn_idx) < 6:
                continue
            
            x_trn_scaled = x_scaled[trn_idx]
            y_trn = y[trn_idx]
            x_val_scaled = x_scaled[val_idx]
            y_val = y[val_idx]
            
            V_trn = np.vander(x_trn_scaled, N=6, increasing=True)
            V_val = np.vander(x_val_scaled, N=6, increasing=True)
            
            # Ridge fit
            A = V_trn.T @ V_trn + alpha * np.eye(6)
            b = V_trn.T @ y_trn
            try:
                c = np.linalg.solve(A, b)
            except:
                try:
                    c = np.linalg.lstsq(V_trn, y_trn, rcond=None)[0]
                except:
                    continue
            
            if not np.all(np.isfinite(c)):
                continue
            
            pred = V_val @ c
            mse = np.mean((pred - y_val) ** 2)
            cv_scores.append(mse)
        
        if len(cv_scores) > 0:
            mean_cv_score = np.mean(cv_scores)
            if mean_cv_score < best_cv_score:
                best_cv_score = mean_cv_score
                # Fit on all data with this alpha
                A_full = V_scaled.T @ V_scaled + alpha * np.eye(6)
                b_full = V_scaled.T @ y
                try:
                    best_coeff_scaled = np.linalg.solve(A_full, b_full)
                except:
                    try:
                        best_coeff_scaled = np.linalg.lstsq(V_scaled, y, rcond=None)[0]
                    except:
                        continue
    
    # If CV didn't find a good solution, fall back to unregularized LS on scaled data
    if best_coeff_scaled is None or not np.all(np.isfinite(best_coeff_scaled)):
        try:
            best_coeff_scaled = np.linalg.lstsq(V_scaled, y, rcond=None)[0]
        except:
            best_coeff_scaled = np.zeros(6)
    
    # Transform coefficients from scaled space back to original space
    # If p(x_scaled) = sum(c_i * x_scaled^i), and x_scaled = (x - x_mean)/x_std
    # Then p(x) = sum(c_i * ((x - x_mean)/x_std)^i)
    # = sum(c_i / x_std^i * (x - x_mean)^i)
    # Need to expand (x - x_mean)^i in terms of x
    # Use binomial theorem: (x - m)^i = sum_{k=0}^{i} binom(i,k) * x^k * (-m)^{i-k}
    
    coeff_original = np.zeros(6)
    for i in range(6):
        ci = best_coeff_scaled[i] / (x_std ** i)
        # Expand (x - x_mean)^i
        for k in range(i + 1):
            if k < 6:
                binom = 1
                for j in range(i - k):
                    binom *= (i - j) / (j + 1)
                coeff_original[k] += ci * binom * ((-x_mean) ** (i - k))
    
    # Final safety check
    if not np.all(np.isfinite(coeff_original)):
        # Fallback: direct LS on original x
        V_orig = np.vander(x, N=6, increasing=True)
        try:
            coeff_original = np.linalg.lstsq(V_orig, y, rcond=None)[0]
        except:
            coeff_original = np.zeros(6)
    
    # Truncate very large coefficients
    max_mag = np.max(np.abs(coeff_original))
    if max_mag > 1e6:
        coeff_original = np.clip(coeff_original, -1e6, 1e6)
    
    return coeff_original
# EVOLVE-BLOCK-END
