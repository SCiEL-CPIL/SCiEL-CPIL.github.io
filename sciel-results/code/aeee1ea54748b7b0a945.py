"""Strategy: Kernel-weighted global polynomial fit to emulate local adaptive closure."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n < 6:
        return np.zeros(6)
        
    # 1. Determine scale for the kernel
    # Use a robust scale estimate (MAD) to handle outliers in the state space
    median_x = np.median(x)
    mad = np.median(np.abs(x - median_x))
    if mad == 0:
        scale = np.std(x)
    else:
        scale = mad * 1.4826  # Consistent with std for normal distribution
        
    # Fallback if scale is still zero or tiny
    if scale < 1e-10:
        scale = 1.0

    # 2. Construct the design matrix for degree 5 polynomial
    # V[:, j] = x**j
    V = np.vander(x, N=6, increasing=True)
    
    # 3. Compute weights for each training point i based on all other points j
    # w_ij = exp(-0.5 * ((x_i - x_j) / bandwidth)^2)
    # To save memory/time, we compute weights in chunks or use vectorized ops if n is small enough.
    # Given L96 data, n is likely manageable (e.g., < 100k). 
    # If n is very large, we might need subsampling for the weight matrix computation, 
    # but let's try direct vectorization first with a time check.
    
    if n > 50000:
        # Subsample for weight calculation if huge, but use full set for final fit? 
        # Actually, for LWLR, the weights are local. 
        # Efficient way: For each i, we only care about neighbors.
        # But we need a single global fit. 
        # The objective is min sum_i w_i * || V x - y ||^2 ? No, that's just weighted global LS.
        
        # Wait, the strategy is "Kernel-weighted local polynomial regression".
        # Standard LWLR fits a separate polynomial in a neighborhood of each point.
        # The task requires a SINGLE SHARED polynomial.
        # Therefore, the strategy must be:
        # "Fit a global polynomial that minimizes the weighted error, where the weights 
        #  emphasize local consistency."
        
        # Actually, a more powerful interpretation of "local adaptive fitting strategy" 
        # constrained to a global polynomial is to use a **Weighted Least Squares** 
        # where weights are determined by local density or residual magnitude?
        
        # Let's look at Inspiration 3: "local adaptive fitting strategy that uses a 
        # kernel-weighted least-squares approach (essentially a local polynomial regression...)".
        # If I fit a local polynomial at every point, I get a function f(x). 
        # Then I project f(x) onto the global degree-5 polynomial basis.
        
        # Plan:
        # 1. For a subset of points (or all if fast), fit a local degree-5 poly using 
        #    Gaussian weights centered at that point.
        # 2. Evaluate this local fit at the point itself (or nearby) to get "smoothed" targets.
        # 3. Fit the global degree-5 poly to these smoothed targets.
        
        # This is effectively a **Smoothing Spline** or **LOESS** projection.
        
        # Let's implement a simplified version:
        # 1. Estimate the local derivative/curvature? No, just local fit.
        # 2. To keep it fast, we can approximate the "local fit value" at x_i by 
        #    taking a weighted average of y_j with weights based on x distance. 
        #    This is a Nadaraya-Watson estimator.
        #    y_smooth_i = sum_j K(x_i, x_j) y_j / sum_j K(x_i, x_j)
        # 3. Fit global poly to y_smooth.
        
        # This reduces noise significantly and captures the mean trend locally 
        # without the instability of high-degree global fits on noisy data.
        
        # Let's try Nadaraya-Watson smoothing first, then global polyfit.
        
        # Choose bandwidth h. 
        # Rule of thumb: h = scale * n^(-1/5) for 1D? Or similar.
        # Let's use a bandwidth that covers a local neighborhood.
        h = scale * (n ** -0.2) 
        # Ensure h is not too small
        if h < 1e-6:
            h = 1e-6

        # Compute smoothed y
        # Vectorized: diff = x_i - x_j
        # This is O(N^2) memory. If N=100k, that's 10^10 floats = 80GB. Too big.
        # We need to chunk this.
        
        y_smooth = np.zeros_like(y)
        chunk_size = 1000
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            x_chunk = x[start:end]
            # diff shape: (chunk_size, n)
            diff = x_chunk[:, np.newaxis] - x[np.newaxis, :]
            # Weights: exp(-0.5 * (diff/h)^2)
            w = np.exp(-0.5 * (diff / h) ** 2)
            # Denominator
            w_sum = np.sum(w, axis=1)
            # Avoid division by zero
            w_sum[w_sum == 0] = 1e-10
            # Numerator
            w_y = np.sum(w * y[np.newaxis, :], axis=1)
            y_smooth[start:end] = w_y / w_sum
            
            if time.time() > deadline - 20:
                break

        # Fit global degree 5 polynomial to smoothed y
        # Use polyfit which is robust
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y_smooth, 5)
        except Exception:
            coeff = np.polynomial.polynomial.polyfit(x, y, 1) # Fallback to linear
        
    else:
        # Smaller N, can do full matrix if needed, but chunking is safe
        y_smooth = np.zeros_like(y)
        chunk_size = 1000
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            x_chunk = x[start:end]
            diff = x_chunk[:, np.newaxis] - x[np.newaxis, :]
            h = scale * (n ** -0.2)
            if h < 1e-6: h = 1e-6
            w = np.exp(-0.5 * (diff / h) ** 2)
            w_sum = np.sum(w, axis=1)
            w_sum[w_sum == 0] = 1e-10
            w_y = np.sum(w * y[np.newaxis, :], axis=1)
            y_smooth[start:end] = w_y / w_sum
            if time.time() > deadline - 20:
                break

        try:
            coeff = np.polynomial.polynomial.polyfit(x, y_smooth, 5)
        except Exception:
            coeff = np.polynomial.polynomial.polyfit(x, y, 1)

    # Ensure finite
    if not np.all(np.isfinite(coeff)):
        coeff = np.zeros(6)
        
    return coeff
# EVOLVE-BLOCK-END
