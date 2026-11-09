"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # If n is very small, just do a simple fit
    if n < 10:
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, min(5, n-1))
            if len(coeff) < 6:
                full_coeff = np.zeros(6)
                full_coeff[:len(coeff)] = coeff
                return full_coeff
            return coeff
        except Exception:
            return np.zeros(6)

    # 1. Kernel Smoothing to denoise the subgrid tendency
    # Estimate bandwidth
    # Use a fraction of the standard deviation or range
    std_x = np.std(x)
    if std_x == 0:
        std_x = 1.0
    
    # Heuristic bandwidth: 
    # We want to smooth out fast-scale fluctuations but keep the slow-scale trend.
    # The slow state x varies on a scale of O(1).
    # Let's try a bandwidth that is small enough to resolve the polynomial features.
    # Silverman's rule for density: 1.06 * std * n^(-1/5)
    # For regression, often a larger bandwidth is used to get a smooth mean function.
    # Let's use a few percent of the range or std.
    h = 0.1 * std_x
    # Ensure h is not too small
    if h < 1e-3:
        h = 1e-3

    # Compute Gaussian kernel weights
    # To be efficient, we can vectorize if n is not huge, but O(N^2) might be slow for large N.
    # Let's check N. If N is large (e.g., > 10000), O(N^2) is 10^8 ops, which is borderline in 100s in Python/NumPy.
    # However, typical trajectory lengths might be smaller. Let's assume N is manageable.
    # If N is very large, we might need a grid-based approach or binning.
    
    if n > 5000:
        # Use a grid-based smoothing for efficiency
        # Bin the data
        bins = np.linspace(np.min(x) - h, np.max(x) + h, 1000)
        # This is a simplified approximation. 
        # For now, let's stick to direct calculation if n is reasonable, else fallback.
        # Actually, let's just try the direct calculation with a warning or limit.
        # If it's too slow, it will hit the deadline.
        pass

    # Direct kernel smoothing
    # K(x_i, x_j) = exp(-0.5 * ((x_i - x_j) / h)^2)
    # This is O(N^2).
    
    # Optimization: Only compute for pairs within a certain distance?
    # Or just do it. NumPy is fast.
    
    try:
        # Vectorized computation
        # diff = x[:, None] - x[None, :]
        # This creates an N x N matrix. If N=10000, that's 100M floats = 800MB. Might be too much memory.
        # Let's do it in chunks or use a simpler method if N is large.
        
        if n > 2000:
            # Use a simpler smoothing: local weighted average using a sorted array and sliding window?
            # Or just reduce the effective N by subsampling? No, we need to use all data.
            # Let's use a grid-based interpolation approach.
            # 1. Bin x into bins.
            # 2. Calculate mean y in each bin.
            # 3. Interpolate to get smooth y?
            # Better: Use np.interp or a simple moving average on sorted data.
            
            # Sort x and y
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            y_sorted = y[sort_idx]
            
            # Use a window size based on h
            # Approximate number of points in window
            # This is an approximation of kernel smoothing.
            window_size = max(5, int(0.1 * n)) # Adaptive window
            
            # Simple moving average with Gaussian-like weights?
            # Let's just use a simple uniform moving average for robustness and speed, 
            # or a weighted one if we can afford it.
            
            # Let's try a weighted moving average on the sorted data.
            # This is O(N log N) for sort + O(N * W) for smoothing.
            
            y_smoothed = np.zeros(n)
            # We need to map back to original indices, or just fit on sorted?
            # Fitting on sorted x, y_smoothed is fine.
            
            # Gaussian weights for the window
            # Window radius in index space is tricky because x is not uniform.
            # Let's stick to the kernel method but with memory checks.
            
            # If N is large, let's use a grid.
            n_bins = 500
            bin_edges = np.linspace(np.min(x), np.max(x), n_bins + 1)
            bin_indices = np.digitize(x, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            
            bin_sums = np.zeros(n_bins)
            bin_counts = np.zeros(n_bins)
            np.add.at(bin_sums, bin_indices, y)
            np.add.at(bin_counts, bin_indices, 1)
            
            valid_bins = bin_counts > 0
            bin_means = np.zeros(n_bins)
            bin_means[valid_bins] = bin_sums[valid_bins] / bin_counts[valid_bins]
            
            # Interpolate to x
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            # Interpolate y_smoothed at x locations
            y_smoothed = np.interp(x, bin_centers, bin_means, left=bin_means[0], right=bin_means[-1])
            
        else:
            # Direct kernel smoothing
            # To save memory, compute in chunks? Or just do it if N < 2000.
            # For N=2000, 4M floats = 32MB. Fine.
            
            # We can use a trick: only compute distances for nearby points?
            # For simplicity and given the 100s limit, let's just do the full matrix if N is small.
            
            # Actually, even for N=2000, a full N x N matrix is fine.
            
            # Calculate distances
            # x_diff = x[:, np.newaxis] - x[np.newaxis, :]
            # This is memory heavy.
            
            # Let's use a simpler approach: 
            # For each i, find j such that |x_i - x_j| < 5*h.
            # This is sparse.
            
            y_smoothed = np.zeros(n)
            for i in range(n):
                # Find neighbors
                # This is O(N^2) in worst case if we don't sort.
                # Let's sort first to make neighbor finding O(N log N) total?
                # But we need to map back.
                
                # Let's just use the grid method for all N > 1000 to be safe and fast.
                pass # Handled by the if n > 2000 block above? 
                # Wait, the logic above was inside an if. Let's restructure.
            
            # Let's restart the smoothing logic cleanly.
            
    except Exception:
        y_smoothed = y # Fallback to raw data

    # Let's implement the grid-based smoothing robustly for all cases, as it's O(N) and fast.
    # It approximates the kernel mean well if bins are small enough.
    
    n_bins = 1000
    x_min, x_max = np.min(x), np.max(x)
    if x_max == x_min:
        x_max = x_min + 1e-6
        
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    bin_indices = np.digitize(x, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    bin_sums = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    np.add.at(bin_sums, bin_indices, y)
    np.add.at(bin_counts, bin_indices, 1)
    
    valid_bins = bin_counts > 0
    bin_means = np.zeros(n_bins)
    bin_means[valid_bins] = bin_sums[valid_bins] / bin_counts[valid_bins]
    
    # Fill empty bins with interpolation
    if np.any(~valid_bins):
        # Interpolate missing bins
        valid_idx = np.where(valid_bins)[0]
        if len(valid_idx) > 1:
            bin_means = np.interp(np.arange(n_bins), valid_idx, bin_means[valid_idx])
        else:
            bin_means[:] = bin_means[valid_bins][0]
            
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    y_smoothed = np.interp(x, bin_centers, bin_means, left=bin_means[0], right=bin_means[-1])
    
    # 2. Polynomial Fit
    # Fit degree 5 polynomial to (x, y_smoothed)
    try:
        # Use polyfit
        coeff = np.polynomial.polynomial.polyfit(x, y_smoothed, 5)
        
        # Check for NaNs or Infs
        if not np.all(np.isfinite(coeff)):
            # Fallback to lower degree or regularized
            coeff = np.polynomial.polynomial.polyfit(x, y, 3)
            if len(coeff) < 6:
                full_coeff = np.zeros(6)
                full_coeff[:len(coeff)] = coeff
                coeff = full_coeff
            if not np.all(np.isfinite(coeff)):
                return np.zeros(6)
                
    except Exception:
        # Fallback to simple linear fit
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 1)
            if len(coeff) < 6:
                full_coeff = np.zeros(6)
                full_coeff[:len(coeff)] = coeff
                coeff = full_coeff
        except Exception:
            coeff = np.zeros(6)
            
    return coeff
# EVOLVE-BLOCK-END
