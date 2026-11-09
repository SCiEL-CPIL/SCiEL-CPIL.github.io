# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    # Handle edge cases
    if len(x) == 0:
        return np.zeros(6)
    
    if time.time() > deadline - 5:
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
            return coeff
        except Exception:
            return np.zeros(6)

    # Use a subset if data is very large for speed
    max_points = 5000
    if len(x) > max_points:
        indices = np.random.choice(len(x), max_points, replace=False)
        x_work = x[indices]
        y_work = y[indices]
    else:
        x_work = x
        y_work = y
        
    # --- Local Polynomial Regression (LOESS-like) to estimate E[Y|X] ---
    # We want to denoise the data by estimating the conditional mean.
    # We will do this on a grid of points covering the range of x.
    
    x_min, x_max = np.min(x), np.max(x)
    pad = 0.1 * (x_max - x_min) if x_max > x_min else 1.0
    x_min -= pad
    x_max += pad
    
    # Number of grid points for denoising
    n_grid = 200
    x_grid = np.linspace(x_min, x_max, n_grid)
    
    # Bandwidth for local regression
    # Use a fraction of the range or a robust scale estimate
    # Let's estimate the typical spacing
    x_sorted = np.sort(x_work)
    n = len(x_sorted)
    if n > 10:
        # Median distance between neighboring points in sorted order
        # This gives a local density estimate
        diffs = np.diff(x_sorted)
        bandwidth = np.median(diffs) * 3.0
        # Ensure bandwidth is not too small or too large
        if bandwidth < 1e-6:
            bandwidth = 0.01 * (x_max - x_min)
        if bandwidth > 0.2 * (x_max - x_min):
            bandwidth = 0.2 * (x_max - x_min)
    else:
        bandwidth = 0.1 * (x_max - x_min) if x_max > x_min else 1.0
        
    # Pre-compute weights for each grid point against all data points
    # To save time, we can vectorize the weight computation for all grid points at once
    # W[i, j] = exp(-0.5 * ((x_grid[i] - x_work[j]) / bandwidth)^2)
    
    # Check time
    if time.time() > deadline - 20:
        # Fallback to simple weighted polyfit or unweighted
        try:
            coeff = np.polynomial.polynomial.polyfit(x_work, y_work, 5)
            return coeff
        except Exception:
            return np.zeros(6)

    # Compute weights matrix
    # Shape: (n_grid, n_work)
    # This might be large if n_work is 5000 and n_grid is 200 -> 1M entries, which is fine.
    x_diff = x_grid[:, np.newaxis] - x_work[np.newaxis, :]
    weights = np.exp(-0.5 * (x_diff / bandwidth) ** 2)
    
    # Normalize weights so they sum to 1 for each grid point (avoid division by zero)
    weight_sums = np.sum(weights, axis=1)
    weight_sums[weight_sums == 0] = 1.0
    weights /= weight_sums[:, np.newaxis]
    
    # Denoised y values: y_grid[i] = sum_j W[i, j] * y_work[j]
    y_denoised = weights @ y_work
    
    # Filter out grid points with very low total weight (outliers in x space with no nearby data)
    # This prevents the polynomial fit from being skewed by extrapolation in empty regions
    valid_mask = weight_sums > 1e-4
    if np.sum(valid_mask) < 10:
        # If too few valid points, just use all
        valid_mask = np.ones(n_grid, dtype=bool)
        
    x_fit = x_grid[valid_mask]
    y_fit = y_denoised[valid_mask]
    
    # Check time
    if time.time() > deadline - 10:
        try:
            coeff = np.polynomial.polynomial.polyfit(x_fit, y_fit, 5)
            return coeff
        except Exception:
            return np.zeros(6)

    # Fit degree-5 polynomial to the denoised data
    # We can use weighted least squares to give more weight to regions with more data
    # The weight for each fit point is the sum of the original weights (data density)
    fit_weights = weight_sums[valid_mask]
    
    try:
        # Use np.polyfit with w parameter? No, np.polynomial.polynomial.polyfit doesn't support weights directly in the same way.
        # We can use lstsq with weights.
        # Construct Vandermonde matrix for degree 5
        V = np.vander(x_fit, 6, increasing=True).T  # 6 x N
        # Weighted least squares: min || W^(1/2) (V c - y) ||^2
        # Equivalent to solving (V^T W V) c = V^T W y
        W_diag = fit_weights
        # Ensure weights are positive
        W_diag[W_diag <= 0] = 1e-10
        
        # Compute A = V^T W V and b = V^T W y
        VW = V * W_diag[np.newaxis, :]  # Scale columns of V by sqrt(weights)? No, V^T W V
        # A_ij = sum_k V_ik W_k V_jk
        A = V.T @ (V * W_diag[np.newaxis, :])
        b = V.T @ (y_fit * W_diag)
        
        # Solve A c = b
        # Add small regularization for stability
        A += 1e-10 * np.eye(6)
        coeff = np.linalg.solve(A, b)
        
        return coeff
    except Exception:
        # Fallback to unweighted polyfit on denoised data
        try:
            coeff = np.polynomial.polynomial.polyfit(x_fit, y_fit, 5)
            return coeff
        except Exception:
            # Final fallback to raw data
            try:
                coeff = np.polynomial.polynomial.polyfit(x_work, y_work, 5)
                return coeff
            except Exception:
                return np.zeros(6)

# EVOLVE-BLOCK-END
