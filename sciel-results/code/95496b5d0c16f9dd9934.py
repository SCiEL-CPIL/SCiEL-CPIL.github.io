# EVOLVE-BLOCK-START
import numpy as np
from numpy.polynomial import polynomial as P


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved tendency.
    Uses a two-step approach:
    1. Smooth the training data using a Gaussian kernel to estimate the underlying mean function.
    2. Fit a degree-5 polynomial to the smoothed data.
    This separates noise reduction from functional approximation, potentially improving generalization
    compared to direct least-squares fitting on raw noisy data.
    """
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Handle potential NaN/Inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 6:
        return np.zeros(6)
    
    try:
        # 1. Kernel Smoothing
        # Sort data for potential efficiency, though vectorized kernel is fine
        # Determine bandwidth
        # Silverman's rule of thumb for 1D
        std_x = np.std(x)
        n = len(x)
        if std_x < 1e-12:
            return np.zeros(6)
        
        # Heuristic bandwidth: balance bias and variance
        # Smaller h -> more variance, larger h -> more bias
        # We want to capture the structure of Lorenz-96 which has multiple scales
        # Let's try a few bandwidths or a standard one.
        # h = 1.06 * std_x * n^(-1/5)
        h = 1.06 * std_x * (n ** (-0.2))
        
        # Ensure h is not too small
        h = max(h, 1e-3 * (np.max(x) - np.min(x)))
        
        # Create a grid of points for smoothing
        # Use a fine grid to represent the smoothed function
        x_min, x_max = np.min(x), np.max(x)
        # Extend slightly to handle edges
        pad = 0.1 * (x_max - x_min)
        x_grid = np.linspace(x_min - pad, x_max + pad, 500)
        
        # Compute kernel smoothed y
        # K(u) = exp(-0.5 * u^2)
        # To avoid memory explosion with large n and large grid, we can chunk or vectorize carefully
        # n is likely ~1000-10000. Grid is 500. 500*10000 = 5e6, which is fine.
        
        # Vectorized kernel evaluation
        # diff shape: (n, 1) - (1, grid_size) -> (n, grid_size)
        diff = (x[:, np.newaxis] - x_grid[np.newaxis, :]) / h
        weights = np.exp(-0.5 * diff**2)
        
        # Sum of weights and weighted y
        # weights shape: (n, grid_size)
        # y shape: (n,)
        w_sum = np.sum(weights, axis=0)
        w_y_sum = np.sum(weights * y[:, np.newaxis], axis=0)
        
        # Avoid division by zero
        w_y_sum = np.divide(w_y_sum, w_sum, out=np.zeros_like(w_y_sum), where=w_sum>1e-12)
        
        # 2. Fit polynomial to smoothed data
        # Filter out points where weight was effectively zero (edges)
        valid_mask = w_sum > 1e-12
        x_fit = x_grid[valid_mask]
        y_fit = w_y_sum[valid_mask]
        
        if len(x_fit) < 6:
             # Fallback to raw fit
             c = P.polyfit(x, y, 5)
             return np.asarray(c, dtype=np.float64)
        
        # Fit degree 5 polynomial
        c = P.polyfit(x_fit, y_fit, 5)
        c = np.asarray(c, dtype=np.float64)
        
        if not np.all(np.isfinite(c)):
             # Fallback to raw fit
             c = P.polyfit(x, y, 5)
             return np.asarray(c, dtype=np.float64)
             
        return c
    except Exception:
        return np.zeros(6)
# EVOLVE-BLOCK-END
