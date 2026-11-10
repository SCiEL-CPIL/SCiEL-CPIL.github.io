# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

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
            if end > current_end:
                current_end = end
        else:
            # Disjoint
            total_length += current_end - current_start
            current_start, current_end = start, end
            
    total_length += current_end - current_start
    return total_length

def _cost_function(X):
    """
    Calculate the approximate volume contribution for symmetric X=Y.
    V = integral_0^1 L(c)^2 dc
    """
    N = 8
    # Sample c values for integration
    # Use a moderate number of samples for speed during optimization
    c_samples = np.linspace(0, 1, 50)
    total_cost = 0.0
    
    indices = np.arange(N)
    
    for c in c_samples:
        # Calculate x-intervals
        # x_k = X[i], shear = c * i / N
        starts = X + c * indices / N
        ends = starts + 1.0 / N
        
        # Compute union length
        intervals = list(zip(starts, ends))
        L = _union_length(intervals)
        
        # Integrate L^2
        total_cost += L * L
        
    # Approximate integral using simple Riemann sum
    # Delta c is 1/(50-1) approx, but linspace includes endpoints.
    # More accurate: trapezoidal rule or just sum * delta.
    # With 50 points, delta = 1/49? No, spacing is 1/49.
    # Let's use np.trapz for better accuracy if we return the array, 
    # but for speed in DE, simple sum * (1/num_samples) is often enough 
    # for ranking, but let's be a bit more precise.
    return total_cost / len(c_samples)

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    
    # Bounds for optimization
    # The tubes have width 1/8. To keep them compact, positions shouldn't be too far apart.
    # Range [-1, 2] is safe.
    bounds = [(-1.0, 2.0) for _ in range(N)]
    
    # Use differential evolution for global optimization
    # This handles the non-smoothness of the union length function well.
    try:
        result = differential_evolution(
            _cost_function, 
            bounds, 
            seed=42, 
            maxiter=100, 
            tol=1e-6,
            popsize=15,
            polish=True
        )
        X_opt = result.x
    except Exception as e:
        # Fallback to a simple heuristic if optimization fails
        # Try to pack intervals tightly
        X_opt = np.zeros(N)
        
    # Construct the 64 positions
    # i = k // 8, j = k % 8
    i_indices = np.repeat(np.arange(N), N)
    j_indices = np.tile(np.arange(N), N)
    
    # Symmetric assumption: Y = X
    base_x = X_opt[i_indices]
    base_y = X_opt[j_indices]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
# EVOLVE-BLOCK-END
