# EVOLVE-BLOCK-START
import numpy as np

def _union_length_fast(intervals):
    """
    Calculate the total length of the union of a list of intervals.
    intervals: array of shape (N, 2) with [start, end].
    """
    # Sort by start
    sorted_intervals = intervals[np.argsort(intervals[:, 0])]
    starts = sorted_intervals[:, 0]
    ends = sorted_intervals[:, 1]
    
    # Merge intervals
    # We can do this vectorized? Not easily because of the running max.
    # But N=8, so a loop is fine and fast.
    current_start = starts[0]
    current_end = ends[0]
    total_length = 0.0
    
    for i in range(1, len(starts)):
        if starts[i] <= current_end:
            if ends[i] > current_end:
                current_end = ends[i]
        else:
            total_length += current_end - current_start
            current_start = starts[i]
            current_end = ends[i]
            
    total_length += current_end - current_start
    return total_length

def _cost_1d(X):
    """
    Calculate the time-averaged length of the union of sheared intervals.
    X: array of 8 base positions.
    """
    N = 8
    indices = np.arange(N)
    # Use more samples for better accuracy, it's cheap
    c_samples = np.linspace(0, 1, 200)
    total_cost = 0.0
    
    # Precompute factors
    shear_factors = indices / N
    width = 1.0 / N
    
    for c in c_samples:
        # Vectorized computation of starts and ends
        starts = X + c * shear_factors
        ends = starts + width
        intervals = np.column_stack((starts, ends))
        total_cost += _union_length_fast(intervals)
        
    return total_cost / len(c_samples)

def _coordinate_descent(X_init, max_iter=1000, tol=1e-8):
    """
    Perform coordinate descent to minimize _cost_1d.
    """
    N = 8
    X = X_init.copy()
    current_cost = _cost_1d(X)
    step_size = 0.01
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            # Try to optimize X[i]
            # Simple golden section or ternary search?
            # Or just finite difference gradient?
            # Let's use a simple line search along axis i.
            # We assume the function is unimodal in each coordinate locally.
            # We will try a few steps.
            
            best_x = X[i]
            best_cost = current_cost
            
            # Search in a window
            # Adaptive window
            for delta in [0.1, 0.05, 0.01, 0.001]:
                for sign in [-1, 1]:
                    X_new = X.copy()
                    X_new[i] = best_x + sign * delta
                    c_new = _cost_1d(X_new)
                    if c_new < best_cost:
                        best_cost = c_new
                        best_x = X_new[i]
                        improved = True
            
            if best_x != X[i]:
                X[i] = best_x
                current_cost = best_cost
                step_size *= 0.9 # Shrink step size if we found an improvement?
                # Actually, if we improved, we should continue searching in that direction or shrink.
                # If we didn't improve, we should shrink.
            else:
                step_size *= 0.5
                if step_size < tol:
                    break
                    
        if not improved:
            break
            
    return X, current_cost

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    
    # Multi-start strategy
    best_X = None
    best_cost = np.inf
    
    # Start 1: Zeros
    X_init = np.zeros(N)
    X_opt, cost = _coordinate_descent(X_init)
    if cost < best_cost:
        best_cost = cost
        best_X = X_opt
        
    # Start 2: Linspace
    X_init = np.linspace(0, 1, N)
    X_opt, cost = _coordinate_descent(X_init)
    if cost < best_cost:
        best_cost = cost
        best_X = X_opt
        
    # Start 3: Negative Linspace (Tiling at c=0)
    X_init = -np.linspace(0, 1, N)[:-1] # -0, -1/7... wait, N=8
    X_init = -np.arange(N) / N
    X_opt, cost = _coordinate_descent(X_init)
    if cost < best_cost:
        best_cost = cost
        best_X = X_opt
        
    # Start 4: Random
    rng = np.random.default_rng(42)
    for _ in range(10):
        X_init = rng.uniform(-1, 1, N)
        X_opt, cost = _coordinate_descent(X_init, max_iter=500)
        if cost < best_cost:
            best_cost = cost
            best_X = X_opt
            
    # Refine the best one with tighter tolerance
    best_X, _ = _coordinate_descent(best_X, max_iter=2000, tol=1e-10)
    
    Y_opt = best_X
    
    # Construct the 64 positions
    i_indices = np.repeat(np.arange(N), N)
    j_indices = np.tile(np.arange(N), N)
    
    base_x = best_X[i_indices]
    base_y = Y_opt[j_indices]
    
    positions = np.stack([base_x, base_y], axis=1)
    return positions
# EVOLVE-BLOCK-END
