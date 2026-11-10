# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _union_length(intervals):
    """
    Calculate the total length of the union of a list of intervals.
    intervals: list of (start, end) tuples.
    """
    if not intervals:
        return 0.0
    # Sort by start time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    
    total_length = 0.0
    current_start, current_end = sorted_intervals[0]
    
    for start, end in sorted_intervals[1:]:
        if start <= current_end:
            # Overlapping or touching
            current_end = max(current_end, end)
        else:
            # Disjoint
            total_length += current_end - current_start
            current_start, current_end = start, end
            
    total_length += current_end - current_start
    return total_length

def _cost_1d(X):
    """
    Calculate the time-averaged length of the union of sheared intervals.
    X: array of 8 base positions.
    """
    N = 8
    indices = np.arange(N)
    # Sample c values
    c_samples = np.linspace(0, 1, 100)
    total_cost = 0.0
    for c in c_samples:
        starts = X + c * indices / N
        ends = starts + 1.0 / N
        # Create list of intervals
        intervals = list(zip(starts, ends))
        total_cost += _union_length(intervals)
    return total_cost / len(c_samples)

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    
    # Initial guess: all zeros to maximize initial overlap at c=0
    X0 = np.zeros(N)
    
    # Optimize X
    # Use Powell's method which doesn't require gradients and is robust for small dimensions
    try:
        res = minimize(_cost_1d, X0, method='Powell', options={'maxiter': 20000, 'xatol': 1e-9, 'fatol': 1e-11})
        if res.success:
            X_opt = res.x
        else:
            # Fallback to Nelder-Mead if Powell fails
            res2 = minimize(_cost_1d, X0, method='Nelder-Mead', options={'maxiter': 20000, 'xatol': 1e-9, 'fatol': 1e-11})
            X_opt = res2.x
    except Exception:
        # If optimization fails completely, fall back to a simple heuristic
        X_opt = np.linspace(0, 1, N)

    # Since the problem is symmetric, Y_opt should be the same
    Y_opt = X_opt

    # Construct the 64 positions
    # i = k // 8, j = k % 8
    i_indices = np.repeat(np.arange(N), N)
    j_indices = np.tile(np.arange(N), N)
    
    base_x = X_opt[i_indices]
    base_y = Y_opt[j_indices]
    
    positions = np.stack([base_x, base_y], axis=1)
    return positions
# EVOLVE-BLOCK-END
