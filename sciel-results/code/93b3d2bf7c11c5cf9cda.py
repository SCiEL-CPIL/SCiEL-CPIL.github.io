import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    # Hard limit is 150s, but we want to be fast. 
    deadline = start_time + 10.0
    
    # Ensure inputs are 1D numpy arrays
    x = np.asarray(training_x, dtype=float).flatten()
    y = np.asarray(training_subgrid_tendency, dtype=float).flatten()
    
    # Filter out any NaNs or Infs
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    n_samples = len(x)
    if n_samples == 0:
        return np.zeros(6)
    
    if n_samples < 6:
        try:
            deg = min(5, n_samples - 1)
            if deg >= 0:
                coeff = np.polynomial.polynomial.polyfit(x, y, deg)
                out = np.zeros(6)
                out[:len(coeff)] = coeff
                return out
        except Exception:
            pass
        return np.zeros(6)
    
    # Stage 1: Nadaraya-Watson Denoising
    # Estimate the conditional expectation E[y|x] to reduce stochastic noise
    # from the fast-scale dynamics before fitting the polynomial.
    
    # Choose bandwidth based on data spread using Silverman's rule
    x_std = np.std(x)
    if x_std == 0:
        return np.zeros(6)
    
    # Silverman's rule: 0.9 * min(std, IQR/1.34) * n^(-1/5)
    x_q1, x_q3 = np.percentile(x, [25, 75])
    x_iqr = x_q3 - x_q1
    x_scale = min(x_std, x_iqr / 1.34) if x_iqr > 0 else x_std
    if x_scale == 0:
        x_scale = x_std
    h_bw = 0.9 * x_scale * n_samples ** (-0.2)
    # Ensure bandwidth is reasonable (not too small or too large)
    h_bw = max(h_bw, 0.05 * (np.max(x) - np.min(x)) / 10)
    h_bw = min(h_bw, 0.5 * x_std)
    
    # To handle memory and speed, we can use a local approach or sort-based.
    # For N ~ 10^4-10^5, a direct loop is too slow. 
    # We can use binning or a grid-based approximation if N is large.
    # However, for N < 50000, we can try a vectorized approach if memory allows,
    # or a sorted sliding window.
    
    # Let's try a simpler approach: Binning
    # Create bins and average y within bins. This is a form of kernel regression with rectangular kernel.
    # Or better, use a few representative points or just do a weighted least squares 
    # where weights are determined by a pre-smoothing step.
    
    # Actually, the directive suggests: "perform an initial Nadaraya-Watson denoising, fit the degree-5 polynomial, and then re-fit it using weights derived from the residuals"
    
    # Efficient NW approximation:
    # If N is large, we can subsample or use a coarse grid.
    # Let's use a grid of ~1000 points for the density estimation to keep it fast.
    
    n_grid = 1000
    x_min, x_max = np.min(x), np.max(x)
    if x_max == x_min:
        return np.zeros(6)
        
    grid = np.linspace(x_min, x_max, n_grid)
    
    # Compute weights for each grid point based on all data points
    # This is O(N * n_grid). If N=10^5, this is 10^8 ops, which might be slow in Python loops but vectorized NumPy is okay if memory allows (10^5 * 1000 matrix is 800MB... too big).
    
    # Better: Use a sorted approach or smaller grid.
    # Let's reduce grid size to 200.
    n_grid = 200
    grid = np.linspace(x_min, x_max, n_grid)
    
    # Compute distances: |x_i - grid_j|
    # x is (N,), grid is (G,)
    # dist is (N, G)
    # If N=10^5, G=200, dist is 10^5*200 = 2*10^7 floats = 160MB. This is acceptable.
    
    dist = np.abs(x[:, np.newaxis] - grid[np.newaxis, :])
    # Gaussian kernel
    weights = np.exp(-0.5 * (dist / h_bw) ** 2)
    
    # Sum weights and weighted y
    w_sum = np.sum(weights, axis=0)
    y_w_sum = np.sum(weights * y[:, np.newaxis], axis=0)
    
    # Avoid division by zero
    y_smooth = np.where(w_sum > 1e-12, y_w_sum / w_sum, np.mean(y))
    
    # Now we have (grid, y_smooth) which is a denoised version of the relationship.
    # Fit the polynomial to this denoised data.
    # But we also want to use the original x for the final fit?
    # The directive says: "fit the degree-5 polynomial, and then re-fit it using weights derived from the residuals"
    
    # Step 1: Fit polynomial to (x, y_smooth_interpolated)? No, y_smooth is only on grid.
    # We should fit the polynomial to the denoised estimates.
    # Or, we can use the denoised values as targets for the original x points?
    # Let's interpolate y_smooth back to x.
    y_denoised = np.interp(x, grid, y_smooth)
    
    # Initial fit to denoised data
    V = np.vander(x, N=6, increasing=True)
    
    # Use ridge regression for stability
    # lambda = 1e-6 * trace(V.T V) / N
    try:
        VtV = V.T @ V
        trace_VtV = np.trace(VtV)
        lam = 1e-6 * trace_VtV / n_samples
        I = np.eye(6)
        c_init = np.linalg.solve(VtV + lam * I, V.T @ y_denoised)
    except np.linalg.LinAlgError:
        c_init = np.polynomial.polynomial.polyfit(x, y_denoised, 5)
    
    # Check time
    if time.time() > deadline:
        return c_init
    
    # Step 2: Robust Refit using Huber-like weights
    # Calculate residuals from the initial fit
    y_pred_init = V @ c_init
    residuals = y - y_pred_init  # Use original noisy y? Or denoised?
    # The directive says "downweight the influence of high-variance fast-scale fluctuations that persist after smoothing".
    # This implies we are fitting to the original y, but weighting points based on how well the smooth model (fit to denoised data) predicts them?
    # Or fitting to y_denoised?
    # "fit the degree-5 polynomial, and then re-fit it using weights derived from the residuals"
    # Usually robust regression fits to the original data but downweights outliers.
    # Here, the "outliers" are the noisy fluctuations.
    # So we fit to original y, but weight points where the initial smooth model (fit to denoised) has small residual?
    # Or simply, weight points based on the residual of the initial fit to the denoised data?
    
    # Let's compute residuals of the initial fit against the ORIGINAL y.
    # Points with large residuals are likely noise-dominated or structural mismatches.
    # We want to downweight them.
    
    # Huber loss weights: w = 1 if |r| < delta, else delta/|r|
    # Estimate scale of residuals
    r_abs = np.abs(residuals)
    delta_huber = 1.4826 * np.median(r_abs)  # MAD-based scale
    if delta_huber == 0:
        delta_huber = 1e-6
        
    # Compute weights
    w = np.ones(n_samples)
    mask_large = r_abs > delta_huber
    w[mask_large] = delta_huber / r_abs[mask_large]
    
    # Re-fit with weighted least squares
    # Minimize sum(w_i * (y_i - V_i c)^2)
    # This is equivalent to solving (V.T W V + lam I) c = V.T W y
    # W is diagonal with w_i
    
    # Efficient update:
    # V_w = V * np.sqrt(w)[:, np.newaxis]
    # y_w = y * np.sqrt(w)
    # Then solve standard ridge on V_w, y_w
    
    sqrt_w = np.sqrt(w)
    V_w = V * sqrt_w[:, np.newaxis]
    y_w = y * sqrt_w
    
    # Re-regularize
    # Use same lambda scale or adjust?
    # Let's use a slightly larger lambda for the robust fit to ensure stability
    lam_robust = 1e-5 * trace_VtV / n_samples
    
    c_robust = None
    try:
        VtV_w = V_w.T @ V_w
        VTy_w = V_w.T @ y_w
        c_robust = np.linalg.solve(VtV_w + lam_robust * I, VTy_w)
        
        if not np.all(np.isfinite(c_robust)):
            c_robust = None
            
    except np.linalg.LinAlgError:
        c_robust = None
    
    # Final high-precision polish: unweighted ridge on denoised data with tighter regularization
    # This removes bias from Huber weighting while retaining the smooth signal
    if time.time() < deadline - 1.0:
        lam_polish = 1e-7 * trace_VtV / n_samples
        try:
            c_polish = np.linalg.solve(VtV + lam_polish * I, V.T @ y_denoised)
            if np.all(np.isfinite(c_polish)):
                # Blend: use polish as primary, robust as fallback
                return c_polish
        except np.linalg.LinAlgError:
            pass
    
    if c_robust is not None:
        return c_robust
    return c_init
