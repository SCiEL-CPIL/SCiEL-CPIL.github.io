"""Rewritten program: Fit degree-5 polynomial via smoothed spline proxy."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import UnivariateSpline

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved subgrid tendency.
    
    Args:
        training_x: flattened array of slow-state values
        training_subgrid_tendency: flattened array of corresponding subgrid tendencies
    
    Returns:
        6 finite real coefficients in ascending powers of x
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = len(x)
    if n == 0:
        return np.zeros(6)

    # Remove any NaN/Inf values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)

    if n == 0:
        return np.zeros(6)

    if n < 6:
        if n >= 2:
            try:
                coeffs = np.polyfit(x, y, 1)
                out = np.zeros(6)
                out[0] = coeffs[1]
                out[1] = coeffs[0]
                return out
            except:
                return np.zeros(6)
        return np.zeros(6)

    # Standardize x for better conditioning
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        return np.zeros(6)
    
    x_norm = (x - x_mean) / x_std

    # Prepare data for spline: sort and average duplicates
    sort_idx = np.argsort(x_norm)
    x_sorted = x_norm[sort_idx]
    y_sorted = y[sort_idx]
    
    unique_x, unique_indices, counts = np.unique(x_sorted, return_index=True, return_counts=True)
    if len(unique_x) < len(x_sorted):
        y_sum = np.zeros_like(unique_x)
        np.add.at(y_sum, unique_indices, y_sorted)
        y_unique = y_sum / counts
    else:
        y_unique = y_sorted
        unique_x = x_sorted

    if len(unique_x) < 6:
        try:
            coeffs = np.polyfit(x, y, 5)
            out = np.zeros(6)
            for i in range(6):
                out[5-i] = coeffs[i]
            if np.all(np.isfinite(out)):
                return out
        except:
            pass
        return np.zeros(6)

    # Optimize smoothing parameter s for UnivariateSpline
    # Use a subset for CV if large
    cv_size = min(1000, len(unique_x))
    rng = np.random.RandomState(42)
    cv_indices = rng.choice(len(unique_x), cv_size, replace=False)
    x_cv = unique_x[cv_indices]
    y_cv = y_unique[cv_indices]
    
    # Sort CV data
    cv_sort_idx = np.argsort(x_cv)
    x_cv = x_cv[cv_sort_idx]
    y_cv = y_cv[cv_sort_idx]
    
    max_s = np.sum((y_cv - np.mean(y_cv))**2)
    if max_s < 1e-12:
        max_s = 1.0
        
    s_candidates = np.logspace(np.log10(1e-4 * max_s), np.log10(max_s), 20)
    
    best_s = max_s # Default to very smooth
    best_mse = np.inf
    
    for s_val in s_candidates:
        if time.time() - start_time > deadline - 10:
            break
        try:
            spl = UnivariateSpline(x_cv, y_cv, s=s_val, k=3)
            y_pred = spl(x_cv)
            mse = np.mean((y_pred - y_cv)**2)
            if mse < best_mse:
                best_mse = mse
                best_s = s_val
        except:
            continue

    # Fit final spline on all unique data
    try:
        spl_final = UnivariateSpline(unique_x, y_unique, s=best_s, k=3)
    except:
        # Fallback to direct polyfit
        try:
            coeffs = np.polyfit(x, y, 5)
            out = np.zeros(6)
            for i in range(6):
                out[5-i] = coeffs[i]
            if np.all(np.isfinite(out)):
                return out
        except:
            return np.zeros(6)

    # Evaluate spline on a grid in normalized space
    x_min_grid = np.min(unique_x)
    x_max_grid = np.max(unique_x)
    x_grid_norm = np.linspace(x_min_grid, x_max_grid, 500)
    y_grid_norm = spl_final(x_grid_norm)
    
    # Fit degree-5 polynomial in normalized space
    V_norm = np.vander(x_grid_norm, 6, increasing=True)
    try:
        coeffs_norm, residuals, rank, s = np.linalg.lstsq(V_norm, y_grid_norm, rcond=None)
    except:
        coeffs_norm = np.zeros(6)

    # Transform to original space by evaluating and refitting
    x_grid_orig = x_grid_norm * x_std + x_mean
    y_grid_orig = y_grid_norm
    
    V_orig = np.vander(x_grid_orig, 6, increasing=True)
    try:
        coeff_orig, residuals, rank, s = np.linalg.lstsq(V_orig, y_grid_orig, rcond=None)
    except:
        coeff_orig = np.zeros(6)

    if not np.all(np.isfinite(coeff_orig)):
        try:
            c_lin = np.polyfit(x, y, 1)
            coeff_orig = np.zeros(6)
            coeff_orig[0] = c_lin[1]
            coeff_orig[1] = c_lin[0]
        except:
            coeff_orig = np.zeros(6)

    return coeff_orig
# EVOLVE-BLOCK-END
