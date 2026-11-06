"""Strategy: Nadaraya-Watson denoising + degree-5 polynomial fit with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _nadaraya_watson(x, y, bandwidths=None, time_limit=None, start_time=None):
    """Apply Nadaraya-Watson kernel smoothing to denoise y given x."""
    n = len(x)
    if n < 10:
        return y.copy()
    
    if bandwidths is None:
        # Use multiple bandwidths for robustness
        x_range = np.max(x) - np.min(x)
        if x_range < 1e-12:
            x_range = 1.0
        bandwidths = [x_range * f for f in [0.01, 0.02, 0.05, 0.1, 0.2]]
    
    best_y_denoised = y.copy()
    best_score = np.inf
    
    for h in bandwidths:
        if time_limit is not None and start_time is not None:
            if time.time() > time_limit:
                break
        
        # Compute weights for each point
        y_denoised = np.zeros(n)
        for i in range(n):
            dist = np.abs(x - x[i])
            w = np.exp(-0.5 * (dist / h) ** 2)
            w_sum = np.sum(w)
            if w_sum < 1e-12:
                y_denoised[i] = y[i]
            else:
                y_denoised[i] = np.sum(w * y) / w_sum
        
        # Evaluate with a simple holdout
        n_val = max(10, n // 10)
        val_idx = np.arange(n - n_val, n)
        trn_idx = np.arange(0, n - n_val)
        
        # Re-fit on training portion only for fair evaluation
        y_denoised_trn = np.zeros(len(trn_idx))
        for i in range(len(trn_idx)):
            orig_i = trn_idx[i]
            dist = np.abs(x[trn_idx] - x[orig_i])
            w = np.exp(-0.5 * (dist / h) ** 2)
            w_sum = np.sum(w)
            if w_sum < 1e-12:
                y_denoised_trn[i] = y[orig_i]
            else:
                y_denoised_trn[i] = np.sum(w * y[trn_idx]) / w_sum
        
        # Use polynomial fit on denoised training to evaluate
        x_trn = x[trn_idx]
        x_val = x[val_idx]
        y_val = y[val_idx]
        
        x_mean = np.mean(x_trn)
        x_std = np.std(x_trn)
        if x_std < 1e-12:
            x_std = 1.0
        x_trn_scaled = (x_trn - x_mean) / x_std
        x_val_scaled = (x_val - x_mean) / x_std
        
        V_trn = np.vander(x_trn_scaled, N=6, increasing=True)
        V_val = np.vander(x_val_scaled, N=6, increasing=True)
        
        try:
            c = np.linalg.lstsq(V_trn, y_denoised_trn, rcond=None)[0]
            pred = V_val @ c
            mse = np.mean((pred - y_val) ** 2)
        except:
            mse = np.inf
        
        if mse < best_score:
            best_score = mse
            best_y_denoised = y_denoised
    
    return best_y_denoised


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = len(x)
    if n < 6:
        return np.zeros(6)

    # Remove any NaN/Inf
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 6:
        return np.zeros(6)
    x = x[mask]
    y = y[mask]
    n = len(x)

    # Apply Nadaraya-Watson denoising to remove fast-scale noise
    y_denoised = _nadaraya_watson(x, y, time_limit=deadline - 60, start_time=start_time)

    # Check time
    if time.time() > deadline - 50:
        # Not enough time for full search, do simple fit
        x_mean = np.mean(x)
        x_std = np.std(x)
        if x_std < 1e-12:
            x_std = 1.0
        x_scaled = (x - x_mean) / x_std
        V_scaled = np.vander(x_scaled, N=6, increasing=True)
        try:
            best_coeff_scaled = np.linalg.lstsq(V_scaled, y_denoised, rcond=None)[0]
        except:
            best_coeff_scaled = np.zeros(6)
    else:
        # Center and scale x to improve conditioning
        x_mean = np.mean(x)
        x_std = np.std(x)
        if x_std < 1e-12:
            x_std = 1.0
        x_scaled = (x - x_mean) / x_std
        
        # Build Vandermonde matrix for scaled x
        V_scaled = np.vander(x_scaled, N=6, increasing=True)
        
        # Dense alpha search with emphasis on regularization
        alphas = np.logspace(-6, 2, 40)  # Reduced range for speed
        
        # Cross-validation: use multiple splits for more robust selection
        n_cv = min(5, max(2, n // 50))
        indices = np.arange(n)
        np.random.seed(42)
        indices = np.random.permutation(indices)
        
        best_coeff_scaled = None
        best_cv_score = np.inf
        
        for alpha in alphas:
            if time.time() > deadline - 20:
                break
            
            cv_scores = []
            for fold in range(n_cv):
                val_start = fold * (n // n_cv)
                val_end = (fold + 1) * (n // n_cv) if fold < n_cv - 1 else n
                val_idx = indices[val_start:val_end]
                trn_idx = np.concatenate([indices[:val_start], indices[val_end:]])
                
                if len(trn_idx) < 6:
                    continue
                
                x_trn_scaled = x_scaled[trn_idx]
                y_trn = y_denoised[trn_idx]
                x_val_scaled = x_scaled[val_idx]
                y_val = y_denoised[val_idx]
                
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
                    b_full = V_scaled.T @ y_denoised
                    try:
                        best_coeff_scaled = np.linalg.solve(A_full, b_full)
                    except:
                        try:
                            best_coeff_scaled = np.linalg.lstsq(V_scaled, y_denoised, rcond=None)[0]
                        except:
                            continue
        
        # If CV didn't find a good solution, fall back to unregularized LS
        if best_coeff_scaled is None or not np.all(np.isfinite(best_coeff_scaled)):
            try:
                best_coeff_scaled = np.linalg.lstsq(V_scaled, y_denoised, rcond=None)[0]
            except:
                best_coeff_scaled = np.zeros(6)

    # Transform coefficients from scaled space back to original space
    coeff_original = np.zeros(6)
    for i in range(6):
        ci = best_coeff_scaled[i] / (x_std ** i)
        for k in range(i + 1):
            if k < 6:
                binom = 1
                for j in range(i - k):
                    binom *= (i - j) / (j + 1)
                coeff_original[k] += ci * binom * ((-x_mean) ** (i - k))
    
    # Final safety check
    if not np.all(np.isfinite(coeff_original)):
        V_orig = np.vander(x, N=6, increasing=True)
        try:
            coeff_original = np.linalg.lstsq(V_orig, y_denoised, rcond=None)[0]
        except:
            coeff_original = np.zeros(6)
    
    # Truncate very large coefficients
    max_mag = np.max(np.abs(coeff_original))
    if max_mag > 1e6:
        coeff_original = np.clip(coeff_original, -1e6, 1e6)
    
    return coeff_original
# EVOLVE-BLOCK-END
