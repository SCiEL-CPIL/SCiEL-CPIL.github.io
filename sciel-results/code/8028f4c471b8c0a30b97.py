"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved tendency.
    
    Strategy:
    1. Use Gaussian kernel smoothing to denoise the training data.
    2. Fit a degree-5 polynomial to the denoised pseudo-labels.
    3. Return coefficients in ascending powers of x.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin: finish by 100s, limit is 120s
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # Normalize x for better numerical conditioning
    x_min = x.min()
    x_max = x.max()
    x_range = x_max - x_min
    if x_range < 1e-12:
        x_norm = np.zeros_like(x)
    else:
        x_norm = (x - x_min) / x_range
    
    # Choose evaluation points for kernel smoothing
    # Use a grid of points spanning the data range
    n_eval = min(500, max(100, n // 10))
    x_eval = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, n_eval)
    x_eval_norm = (x_eval - x_min) / x_range if x_range > 1e-12 else np.zeros_like(x_eval)
    
    # Gaussian kernel bandwidth selection
    # Use a reasonable default bandwidth
    bandwidth = 0.1
    
    # Compute kernel-weighted averages at evaluation points
    # Vectorized computation for speed
    y_smoothed = np.zeros(n_eval)
    for i in range(n_eval):
        dist = np.abs(x_eval_norm[i] - x_norm)
        weights = np.exp(-0.5 * (dist / bandwidth) ** 2)
        weight_sum = np.sum(weights)
        if weight_sum > 1e-12:
            y_smoothed[i] = np.sum(weights * y) / weight_sum
        else:
            y_smoothed[i] = np.mean(y)
    
    # Check time
    if time.time() > deadline - 10:
        # Fallback to direct polynomial fit
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
            return coeff
        except:
            return np.zeros(6)
    
    # Fit degree-5 polynomial to the denoised data
    # Use the original x values for the fit
    try:
        coeff = np.polynomial.polynomial.polyfit(x_eval, y_smoothed, 5)
    except:
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
        except:
            coeff = np.zeros(6)
    
    # Ensure coefficients are finite
    if not np.all(np.isfinite(coeff)):
        coeff = np.zeros(6)
    
    return coeff
# EVOLVE-BLOCK-END
