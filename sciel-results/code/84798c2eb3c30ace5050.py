"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    if len(x) == 0:
        return np.zeros(6)

    # Fallback: simple degree-5 polyfit if time is tight
    if time.time() > deadline - 10:
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
            return coeff
        except Exception:
            return np.zeros(6)

    # Stage 1: Kernel-weighted denoising to estimate E[Y|X]
    # Use a Gaussian kernel with adaptive bandwidth
    n = len(x)
    
    # Compute a reasonable bandwidth based on data spread
    x_range = np.ptp(x)
    if x_range < 1e-10:
        bandwidth = 1.0
    else:
        # Silverman's rule of thumb adapted for 1D
        bandwidth = 1.06 * np.std(x) * (n ** (-1.0 / 5.0))
        if bandwidth < 1e-6:
            bandwidth = x_range * 0.1
        bandwidth = max(bandwidth, x_range * 0.05)

    # Use a subset of points for efficiency if n is large
    max_knn = 2000
    if n > max_knn:
        # Randomly subsample for kernel evaluation
        idx = np.random.RandomState(42).choice(n, size=max_knn, replace=False)
        x_sub = x[idx]
        y_sub = y[idx]
    else:
        x_sub = x
        y_sub = y

    # Evaluate kernel weights at each training point using subsampled points
    # For efficiency, we'll do this in chunks
    chunk_size = 1000
    y_smoothed = np.zeros(n)
    
    for i in range(0, n, chunk_size):
        if time.time() > deadline - 15:
            # Fallback to simple polyfit
            try:
                coeff = np.polynomial.polynomial.polyfit(x, y, 5)
                return coeff
            except Exception:
                return np.zeros(6)
        
        i_end = min(i + chunk_size, n)
        x_chunk = x[i:i_end]
        
        # Compute kernel weights: K(x_chunk, x_sub) / sum
        # Reshape for broadcasting: (chunk_size, 1) and (1, len(x_sub))
        diff = (x_chunk[:, np.newaxis] - x_sub[np.newaxis, :]) / bandwidth
        weights = np.exp(-0.5 * diff ** 2)
        
        # Normalize weights
        weight_sums = np.sum(weights, axis=1, keepdims=True)
        weight_sums = np.where(weight_sums < 1e-12, 1.0, weight_sums)
        weights = weights / weight_sums
        
        # Compute weighted average
        y_smoothed[i:i_end] = weights @ y_sub

    # Stage 2: Fit degree-5 polynomial to smoothed pseudo-labels
    try:
        coeff = np.polynomial.polynomial.polyfit(x, y_smoothed, 5)
    except Exception:
        # Fallback to direct fit
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
        except Exception:
            coeff = np.zeros(6)

    # Ensure coefficients are finite
    if not np.all(np.isfinite(coeff)):
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
        except Exception:
            coeff = np.zeros(6)

    return coeff
# EVOLVE-BLOCK-END
