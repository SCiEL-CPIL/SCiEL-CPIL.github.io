"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial using multi-bandwidth kernel-weighted
    denoising followed by weighted polynomial regression.
    
    Strategy:
    1. Create a dense evaluation grid spanning the data range.
    2. Use multiple bandwidths (geometrically spaced) to denoise via kernel
       regression, producing pseudo-labels at each grid point.
    3. Weight the pseudo-labels by local density (inverse of bandwidth
       uncertainty) and fit a degree-5 polynomial.
    4. Return coefficients in ascending powers of x.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin: finish by 110s, limit is 150s
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # Basic statistics
    x_mean = x.mean()
    x_std = x.std()
    if x_std < 1e-10:
        x_std = 1.0
    
    y_mean = y.mean()
    
    # Step 1: Create evaluation grid
    # Use a dense grid that covers the full range with extra points in high-density regions
    n_eval = min(3000, max(500, n // 2))
    
    # Mix of quantile-based and uniform points for good coverage
    q_points = np.quantile(x, np.linspace(0.001, 0.999, n_eval // 2))
    u_points = np.linspace(x.min(), x.max(), n_eval // 2)
    eval_x = np.concatenate([q_points, u_points])
    eval_x = np.sort(eval_x)
    
    # Step 2: Multi-bandwidth kernel denoising
    # Use several bandwidths and combine results
    x_range = x.max() - x.min()
    if x_range < 1e-10:
        x_range = 1.0
    
    # Geometrically spaced bandwidths
    bandwidths = np.array([0.02, 0.05, 0.10, 0.20, 0.40]) * x_range
    bandwidths = bandwidths[bandwidths > 1e-6]
    
    if len(bandwidths) == 0:
        bandwidths = np.array([x_range * 0.1])
    
    # For each bandwidth, compute kernel-weighted mean at eval points
    # Use vectorized computation where possible
    pseudo_y_combined = np.zeros(n_eval)
    weights_combined = np.zeros(n_eval)
    
    for bw in bandwidths:
        if time.time() > deadline - 20:
            break
        
        # Process in chunks to manage memory
        chunk_size = max(1, min(200, n_eval))
        
        for start in range(0, n_eval, chunk_size):
            if time.time() > deadline - 15:
                break
            
            end = min(start + chunk_size, n_eval)
            chunk_eval = eval_x[start:end]
            
            # Compute distances from all x to each eval point in chunk
            # Shape: (n, chunk_size)
            diff = x[:, np.newaxis] - chunk_eval[np.newaxis, :]
            diff_scaled = diff / bw
            
            # Gaussian kernel weights
            weights = np.exp(-0.5 * diff_scaled ** 2)
            
            # Weighted sum and sum of weights
            w_sums = weights.sum(axis=0)  # (chunk_size,)
            wy_sums = (weights * y[:, np.newaxis]).sum(axis=0)  # (chunk_size,)
            
            # Avoid division by zero
            valid = w_sums > 1e-10
            local_means = np.where(valid, wy_sums / np.where(valid, w_sums, 1.0), y_mean)
            
            # Weight by effective sample size (proxy for confidence)
            eff_n = np.where(valid, w_sums / bw * x_range, 0.0)
            
            pseudo_y_combined[start:end] += local_means
            weights_combined[start:end] += eff_n
    
    # Combine: weighted average of pseudo-labels
    valid_mask = weights_combined > 1e-10
    pseudo_y = np.where(valid_mask, pseudo_y_combined / np.where(valid_mask, weights_combined, 1.0), y_mean)
    
    # Step 3: Fit degree-5 polynomial with weights
    # Use the effective sample sizes as weights
    fit_weights = np.where(valid_mask, weights_combined, 0.0)
    # Normalize weights
    if fit_weights.sum() > 0:
        fit_weights = fit_weights / fit_weights.sum()
    else:
        fit_weights = np.ones(n_eval) / n_eval
    
    # Only use points with sufficient weight
    min_weight = fit_weights.max() * 0.01
    use_mask = fit_weights > min_weight
    
    if use_mask.sum() < 10:
        use_mask = np.ones(n_eval, dtype=bool)
    
    x_fit = eval_x[use_mask]
    y_fit = pseudo_y[use_mask]
    w_fit = fit_weights[use_mask]
    
    # Weighted least squares polynomial fit
    # Build Vandermonde matrix for degree 5
    deg = 5
    V = np.vander(x_fit, N=deg + 1, increasing=True)  # (n_fit, 6)
    
    try:
        # Weighted least squares: solve (V^T W V) coeff = V^T W y
        W_sqrt = np.sqrt(w_fit)
        V_w = V * W_sqrt[:, np.newaxis]
        y_w = y_fit * W_sqrt
        
        # Use lstsq for numerical stability
        coeff, residuals, rank, sv = np.linalg.lstsq(V_w, y_w, rcond=None)
        
        # Check if fit is reasonable
        pred = np.polyval(coeff[::-1], x_fit)  # Note: np.polyval uses descending order
        mse = np.mean((pred - y_fit) ** 2)
        y_var = np.var(y_fit)
        if y_var > 0 and mse / y_var > 0.9:
            # Fit is bad, try without weights
            coeff = np.polynomial.polynomial.polyfit(x_fit, y_fit, deg)
    except Exception:
        # Fallback to unweighted fit
        try:
            coeff = np.polynomial.polynomial.polyfit(x_fit, y_fit, deg)
        except Exception:
            # Final fallback: simple linear fit
            try:
                coeff = np.zeros(6)
                coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            except Exception:
                coeff = np.zeros(6)
                coeff[0] = y_mean
    
    # Ensure coefficients are finite
    if not np.all(np.isfinite(coeff)):
        coeff = np.where(np.isfinite(coeff), coeff, 0.0)
    
    return coeff
# EVOLVE-BLOCK-END
