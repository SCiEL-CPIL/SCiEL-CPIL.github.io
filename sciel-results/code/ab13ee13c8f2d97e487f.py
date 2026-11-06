import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    x = np.asarray(training_x, dtype=float)
    y = np.asarray(training_subgrid_tendency, dtype=float)
    
    # Filter out any NaNs or infinities just in case
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    n_samples = len(x)
    
    if n_samples < 10:
        # Fallback to linear if data is too sparse
        coeff = np.zeros(6)
        if n_samples >= 2:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        return coeff
    
    x_min, x_max = np.min(x), np.max(x)
    if x_max - x_min < 1e-10:
        # All x are same, fit constant
        coeff = np.zeros(6)
        coeff[0] = np.mean(y)
        return coeff

    # Strategy: 
    # 1. Use Nadaraya-Watson kernel smoothing to denoise the fast-scale fluctuations
    # 2. Fit a degree-5 polynomial to the smoothed data
    # 3. Perform iterative reweighting (Huber loss) to robustly refit the polynomial
    # 4. Select the best configuration based on training MSE (proxy for validation)
    
    # Candidate bandwidths for NW smoothing (fraction of data range)
    # We want a range of smoothness levels
    candidate_bw_fracs = [0.05, 0.1, 0.2, 0.3, 0.5]
    
    best_coeff = None
    best_mse = np.inf
    
    for bw_frac in candidate_bw_fracs:
        if time.time() > deadline - 5:
            break
            
        bandwidth = bw_frac * (x_max - x_min)
        
        # Nadaraya-Watson estimator
        # K(u) = exp(-0.5 * u^2) (Gaussian kernel)
        # y_hat(x_i) = sum_j K((x_i - x_j)/h) * y_j / sum_j K((x_i - x_j)/h)
        
        # Vectorized computation might be slow for large N, but N is likely manageable
        # Let's try to be efficient.
        # diff = x[:, None] - x[None, :]  # N x N matrix
        # This is O(N^2) memory and compute. If N is ~10k, this is 100M ops, might be slow.
        # Let's check N first.
        
        if n_samples > 5000:
            # For large N, use binning approximation for NW
            n_bins = max(50, min(500, int(np.sqrt(n_samples))))
            bin_edges = np.linspace(x_min, x_max, n_bins + 1)
            bin_indices = np.digitize(x, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            
            sum_x = np.bincount(bin_indices, weights=x, minlength=n_bins)
            count = np.bincount(bin_indices, minlength=n_bins)
            sum_y = np.bincount(bin_indices, weights=y, minlength=n_bins)
            
            valid_bins = count > 0
            if np.sum(valid_bins) < 6:
                continue
                
            mean_x = np.zeros(n_bins)
            mean_y = np.zeros(n_bins)
            mean_x[valid_bins] = sum_x[valid_bins] / count[valid_bins]
            mean_y[valid_bins] = sum_y[valid_bins] / count[valid_bins]
            
            x_smooth = mean_x[valid_bins]
            y_smooth = mean_y[valid_bins]
            
            # Apply a light Gaussian smoothing to the bin means to mimic NW?
            # Or just fit directly to bin means.
            # Let's stick to fitting bin means for speed.
            try:
                coeff = np.polynomial.polynomial.polyfit(x_smooth, y_smooth, 5)
            except Exception:
                continue
        else:
            # Direct NW for smaller N
            diff = x[:, None] - x[None, :]
            weights = np.exp(-0.5 * (diff / bandwidth) ** 2)
            # Handle division by zero if bandwidth is very small (shouldn't happen with fracs)
            denom = np.sum(weights, axis=1)
            denom[denom == 0] = 1e-10
            y_smooth = np.sum(weights * y[None, :], axis=1) / denom
            x_smooth = x  # We fit to x, y_smooth pairs
            
            # Subsample if even N is too large for polyfit?
            # polyfit is O(N) or O(N^2) depending on implementation.
            # Let's just try it.
            try:
                coeff = np.polynomial.polynomial.polyfit(x_smooth, y_smooth, 5)
            except Exception:
                continue
        
        # Robust Refit: Iterative Reweighted Least Squares (Huber)
        # 1. Compute residuals
        y_pred = np.polynomial.polynomial.polyval(x, coeff)
        residuals = y - y_pred
        
        # 2. Compute Huber weights
        # w_i = 1 if |r_i| <= delta, else delta / |r_i|
        # Choose delta based on MAD or standard deviation
        mad = np.median(np.abs(residuals - np.median(residuals)))
        if mad == 0:
            mad = np.std(residuals)
        if mad == 0:
            mad = 1e-6
            
        delta = 1.4826 * mad  # 1.4826 makes MAD consistent with std for normal dist
        
        # Weights
        abs_res = np.abs(residuals)
        weights = np.where(abs_res <= delta, 1.0, delta / abs_res)
        
        # 3. Refit with weights
        # polyfit doesn't support weights directly in np.polynomial.polynomial.polyfit?
        # Actually, it does not in older versions, but let's check.
        # np.polynomial.polynomial.polyfit(x, y, deg, w=None) -> w is supported!
        
        try:
            coeff_robust = np.polynomial.polynomial.polyfit(x, y, 5, w=weights)
        except Exception:
            coeff_robust = coeff
            
        # Evaluate both original and robust fits on training data
        y_pred_orig = np.polynomial.polynomial.polyval(x, coeff)
        mse_orig = np.mean((y - y_pred_orig) ** 2)
        
        y_pred_rob = np.polynomial.polynomial.polyval(x, coeff_robust)
        mse_rob = np.mean((y - y_pred_rob) ** 2)
        
        # Usually robust fit has higher training MSE but better generalization.
        # We need a selection criterion.
        # Since we don't have validation data, we can use cross-validation on a subset
        # or just pick the one with lower training MSE?
        # The task says "minimize normalized error on independent validation trajectories".
        # Training MSE is a biased estimator.
        # However, if the robust fit is significantly better at capturing the trend
        # and less sensitive to outliers, it might generalize better.
        
        # Let's try a simple 5-fold cross-validation on a random subset to pick the best coeff
        # among (coeff, coeff_robust) for this bandwidth.
        # If N is large, subsample for CV.
        cv_size = min(1000, n_samples)
        if cv_size < 10:
            cv_idx = np.arange(n_samples)
        else:
            cv_idx = np.random.choice(n_samples, cv_size, replace=False)
            
        x_cv = x[cv_idx]
        y_cv = y[cv_idx]
        
        # Split into train/test
        split_idx = int(0.8 * cv_size)
        x_tr = x_cv[:split_idx]
        y_tr = y_cv[:split_idx]
        x_te = x_cv[split_idx:]
        y_te = y_cv[split_idx:]
        
        # Fit both candidates on x_tr, y_tr
        try:
            # Candidate 1: Standard fit on train part
            coeff_cv1 = np.polynomial.polynomial.polyfit(x_tr, y_tr, 5)
            y_te_pred1 = np.polynomial.polynomial.polyval(x_te, coeff_cv1)
            mse_cv1 = np.mean((y_te - y_te_pred1) ** 2)
        except Exception:
            mse_cv1 = np.inf
            
        try:
            # Candidate 2: Robust fit on train part
            # Recompute weights for train part
            y_pred_tr = np.polynomial.polynomial.polyval(x_tr, coeff) # Use global coeff as init? No, fit from scratch.
            # Actually, we should fit the robust polynomial from scratch on x_tr, y_tr
            # 1. Initial fit
            coeff_init = np.polynomial.polynomial.polyfit(x_tr, y_tr, 5)
            y_pred_tr_init = np.polynomial.polynomial.polyval(x_tr, coeff_init)
            res_tr = y_tr - y_pred_tr_init
            mad_tr = np.median(np.abs(res_tr - np.median(res_tr)))
            if mad_tr == 0: mad_tr = 1e-6
            delta_tr = 1.4826 * mad_tr
            abs_res_tr = np.abs(res_tr)
            w_tr = np.where(abs_res_tr <= delta_tr, 1.0, delta_tr / abs_res_tr)
            
            coeff_cv2 = np.polynomial.polynomial.polyfit(x_tr, y_tr, 5, w=w_tr)
            y_te_pred2 = np.polynomial.polynomial.polyval(x_te, coeff_cv2)
            mse_cv2 = np.mean((y_te - y_te_pred2) ** 2)
        except Exception:
            mse_cv2 = np.inf
            
        # Pick the better CV candidate
        if mse_cv2 < mse_cv1:
            # Robust is better. Use coeff_robust (fitted on all data with weights)
            # Or should we refit robust on all data? We already have coeff_robust.
            # Let's assume coeff_robust is the best representative.
            final_coeff_candidate = coeff_robust
            final_mse_candidate = mse_rob # Proxy for logging
        else:
            final_coeff_candidate = coeff
            final_mse_candidate = mse_orig
            
        # Update global best
        # We should compare based on CV error if possible, but we only have training MSE.
        # Let's use the CV error from the subset as the selection metric.
        cv_score = mse_cv2 if mse_cv2 < mse_cv1 else mse_cv1
        
        if cv_score < best_mse:
            best_mse = cv_score
            best_coeff = final_coeff_candidate

    if best_coeff is None:
        # Fallback
        try:
            best_coeff = np.polynomial.polynomial.polyfit(x, y, 5)
        except Exception:
            best_coeff = np.zeros(6)
            if n_samples >= 2:
                best_coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            else:
                best_coeff[0] = np.mean(y) if len(y)>0 else 0.0

    return best_coeff
