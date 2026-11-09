"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
    
    # Strategy: 
    # 1. Create a dense grid of x values covering the range of training data.
    # 2. For each grid point, estimate the mean y using a local weighted least squares (LOESS) 
    #    or simple weighted mean with a Gaussian kernel. This effectively denoises the data 
    #    and captures the underlying conditional expectation E[Y|X].
    # 3. Fit a global degree-5 polynomial to these (x_grid, y_mean) pairs.
    # This is structurally different from direct polynomial regression on raw data because 
    # it first reduces the dimensionality/noise by estimating the conditional mean.
    
    # Define grid
    x_min, x_max = np.min(x), np.max(x)
    # Add a small buffer to cover the full range
    buffer = 0.05 * (x_max - x_min) if x_max > x_min else 0.1
    x_min -= buffer
    x_max += buffer
    
    num_grid = 200
    x_grid = np.linspace(x_min, x_max, num_grid)
    
    # Estimate conditional mean E[Y|X] at each grid point
    # We use a weighted average with a Gaussian kernel.
    # The bandwidth (sigma) is chosen based on the local density or a fixed fraction of range.
    # A fixed fraction of the range is robust.
    sigma = (x_max - x_min) / 10.0
    
    y_grid = np.zeros_like(x_grid)
    
    for i in range(num_grid):
        xi = x_grid[i]
        # Weights: Gaussian kernel
        # To avoid overflow/underflow in exp, we can compute distances first
        dist = x - xi
        # If sigma is very small, weights might be zero. 
        # Let's ensure sigma is at least some multiple of std(x) if possible, or just use range.
        # Using range/10 is a standard heuristic.
        
        weights = np.exp(-0.5 * (dist / sigma) ** 2)
        
        # Check if weights are effectively zero
        w_sum = np.sum(weights)
        if w_sum > 1e-12:
            y_grid[i] = np.sum(weights * y) / w_sum
        else:
            # Fallback to global mean if no data nearby (should not happen with buffer)
            y_grid[i] = np.mean(y)
            
        # Time check
        if time.time() - t_start > deadline - 5.0:
            # If we run out of time, break and fit with what we have
            # But we need to fit the polynomial. Let's just break the loop.
            # Actually, let's just break and use the partial y_grid? No, that's bad.
            # Better to just return a simple fit if time is tight.
            # But 200 iterations is very fast. This check is likely unnecessary but safe.
            pass

    # Fit a degree-5 polynomial to the denoised curve (x_grid, y_grid)
    # We use np.polyfit or np.polynomial.polynomial.polyfit
    # polyfit returns coefficients in descending powers. We need ascending.
    try:
        # Use polyfit with degree 5
        # x_grid and y_grid are the denoised data
        coeff_desc = np.polyfit(x_grid, y_grid, 5)
        # Convert to ascending order
        coeff_asc = coeff_desc[::-1]
        
        # Ensure all coefficients are finite
        if not np.all(np.isfinite(coeff_asc)):
             # Fallback to linear fit if numerical issues
             coeff_desc = np.polyfit(x_grid, y_grid, 1)
             coeff_asc = np.zeros(6)
             coeff_asc[0] = coeff_desc[1]
             coeff_asc[1] = coeff_desc[0]
             
    except Exception:
        # Fallback to simple linear fit on raw data if everything fails
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 1)
            out = np.zeros(6)
            out[:2] = coeff
            return out
        except Exception:
            return np.zeros(6, dtype=np.float64)
            
    return coeff_asc
# EVOLVE-BLOCK-END
