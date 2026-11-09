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
        # Increase bandwidth factor to 7.0 for even smoother denoising
        bandwidth = np.median(diffs) * 7.0
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
        # Use Total Least Squares (TLS) approach for better generalization
        # This accounts for noise in both x and y
        # Construct augmented matrix [x_fit, y_fit] with weights
        W_diag = fit_weights
        W_diag[W_diag <= 0] = 1e-10
        
        # Weighted TLS: minimize sum W_k * ||(x_k, y_k) - (x_k, V c)||^2
        # This is equivalent to solving a generalized eigenvalue problem
        # For degree-5 polynomial: y = c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4 + c5*x^5
        # Rearranging: c0 + c1*x + ... + c5*x^5 - y = 0
        # So we're looking for coefficients [c0, c1, ..., c5, -1] such that
        # sum_i a_i * z_i = 0 where z = [1, x, x^2, x^3, x^4, x^5, y]
        
        # Construct weighted design matrix
        Z = np.vander(x_fit, 6, increasing=True)  # N x 6
        Z = np.hstack([Z, y_fit[:, np.newaxis]])  # N x 7
        
        # Weight each row by sqrt(W_diag)
        Z_weighted = Z * np.sqrt(W_diag)[:, np.newaxis]
        
        # SVD of weighted design matrix
        U, s, Vh = np.linalg.svd(Z_weighted, full_matrices=False)
        
        # The solution is the last right singular vector (corresponding to smallest singular value)
        # But we need to normalize so that the last component (coefficient of y) is -1
        v_last = Vh[-1, :]
        
        # Normalize so that v_last[-1] = -1
        if abs(v_last[-1]) > 1e-12:
            v_last = v_last / v_last[-1]
            coeff = v_last[:-1]
        else:
            # Fallback to standard WLS if TLS degenerates
            V = np.vander(x_fit, 6, increasing=True).T
            A = V.T @ (V * W_diag[np.newaxis, :])
            b = V.T @ (y_fit * W_diag)
            reg = 1e-6 * np.trace(A) / 6.0
            if reg < 1e-10:
                reg = 1e-10
            A += reg * np.eye(6)
            U2, s2, Vh2 = np.linalg.svd(A)
            coeff = Vh2.T @ (np.where(s2 > 1e-10, (U2.T @ b) / s2, 0.0))
        
        return coeff
    except Exception:
        # Fallback to weighted polyfit on denoised data
        try:
            V = np.vander(x_fit, 6, increasing=True).T
            W_diag = fit_weights
            W_diag[W_diag <= 0] = 1e-10
            A = V.T @ (V * W_diag[np.newaxis, :])
            b = V.T @ (y_fit * W_diag)
            reg = 1e-6 * np.trace(A) / 6.0
            if reg < 1e-10:
                reg = 1e-10
            A += reg * np.eye(6)
            U2, s2, Vh2 = np.linalg.svd(A)
            coeff = Vh2.T @ (np.where(s2 > 1e-10, (U2.T @ b) / s2, 0.0))
            return coeff
        except Exception:
            # Final fallback to raw data
            try:
                coeff = np.polynomial.polynomial.polyfit(x_work, y_work, 5)
                return coeff
            except Exception:
                return np.zeros(6)

# EVOLVE-BLOCK-END
