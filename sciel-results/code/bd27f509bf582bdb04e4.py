"""Strategy: Binned-mean denoising followed by degree-5 polynomial fit."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n < 10:
        return np.zeros(6)
    
    # Remove any NaNs or Infs
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    
    if n < 10:
        return np.zeros(6)

    # Strategy: Bin the data to estimate E[y|x] locally, then fit polynomial.
    # This acts as a denoising step (low-pass filter) to remove fast-scale fluctuations.
    
    # Choose number of bins. 
    # Too few: underfitting the shape of E[y|x].
    # Too many: noise remains.
    # Heuristic: sqrt(N) to N^(2/5) is common for histogram density, 
    # but for regression smoothing, we want enough points per bin for stable mean.
    # Let's aim for at least 20-50 points per bin if possible.
    
    # Determine range of x
    x_min, x_max = np.min(x), np.max(x)
    if x_max == x_min:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.mean(y)])
        
    # Adaptive bin count
    # If N is large, we can use more bins.
    # Let's try K = max(50, min(200, int(np.sqrt(n))))
    # But we need to ensure we don't timeout. Binning is O(N log N) for sort.
    
    # Sort x for binning
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Define bins
    # Use a reasonable number of bins to balance bias/variance.
    # 100 bins is usually a good start for this type of problem.
    n_bins = 100
    
    # Create bin edges
    # Ensure bins cover the data range
    edges = np.linspace(x_min, x_max, n_bins + 1)
    
    # Assign points to bins
    # np.digitize returns 1..n_bins for points in bins, 0 for < min, n_bins+1 for > max
    # Since we used linspace from min to max, all points should be in 1..n_bins
    # Note: points exactly on edges might be tricky, but digitize handles it consistently.
    bin_indices = np.digitize(x_sorted, edges) - 1
    
    # Filter out indices 0 or n_bins (shouldn't happen much with min/max edges, but safe)
    valid_bins = (bin_indices >= 0) & (bin_indices < n_bins)
    x_valid = x_sorted[valid_bins]
    y_valid = y_sorted[valid_bins]
    bin_indices_valid = bin_indices[valid_bins]
    
    # Compute bin centers and means
    bin_centers = np.zeros(n_bins)
    bin_means = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    
    # Use np.bincount for efficiency
    # We need sum(y) and count for each bin
    sum_y = np.bincount(bin_indices_valid, weights=y_valid, minlength=n_bins)
    counts = np.bincount(bin_indices_valid, minlength=n_bins)
    
    # Identify non-empty bins
    non_empty = counts > 0
    
    # Compute mean y for non-empty bins
    bin_means[non_empty] = sum_y[non_empty] / counts[non_empty]
    
    # Compute mean x for non-empty bins (better than center of edge)
    sum_x = np.bincount(bin_indices_valid, weights=x_valid, minlength=n_bins)
    bin_centers[non_empty] = sum_x[non_empty] / counts[non_empty]
    
    # Extract valid bin data for fitting
    fit_x = bin_centers[non_empty]
    fit_y = bin_means[non_empty]
    
    # If we don't have enough points to fit degree 5, fall back to raw data or lower degree
    if len(fit_x) < 6:
         # Fall back to fitting raw data with degree 1 or 0 if binning failed
         # But usually 100 bins will have many non-empty bins.
         # If not, just fit raw data with degree 1.
         try:
             coeff = np.polynomial.polynomial.polyfit(x, y, 1)
             return np.pad(coeff, (0, 5-len(coeff)))
         except Exception:
             return np.zeros(6)

    # Fit degree 5 polynomial to the binned means
    # np.polynomial.polynomial.polyfit returns coefficients in ascending order: c0, c1, ..., c5
    try:
        # Check time
        if time.time() > deadline - 5.0:
             # Fallback to simple linear fit if running out of time
             coeff = np.polynomial.polynomial.polyfit(x, y, 1)
             return np.pad(coeff, (0, 5-len(coeff)))
             
        coeff = np.polynomial.polynomial.polyfit(fit_x, fit_y, 5)
        
        # Sanity check: coefficients should be finite
        if not np.all(np.isfinite(coeff)):
             # Fallback
             coeff = np.polynomial.polynomial.polyfit(x, y, 1)
             return np.pad(coeff, (0, 5-len(coeff)))
             
        return coeff

    except Exception as e:
        # If polyfit fails (e.g. rank deficiency), fall back to linear
        try:
             coeff = np.polynomial.polynomial.polyfit(x, y, 1)
             return np.pad(coeff, (0, 5-len(coeff)))
        except Exception:
             return np.zeros(6)
# EVOLVE-BLOCK-END
