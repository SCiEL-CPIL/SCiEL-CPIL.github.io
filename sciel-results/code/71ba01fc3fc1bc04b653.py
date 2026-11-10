import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing task.
    
    Args:
        demands: list or array of item sizes (floats).
        capacity: bin capacity (float).
        
    Returns:
        numpy array of shape (n, n) with heuristic values.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    n = len(demands)
    d = np.array(demands, dtype=np.float64)
    
    # Calculate pair sums
    pair_sums = d[:, None] + d[None, :]
    
    # Calculate waste (remaining space in bin after placing i and j)
    waste = capacity - pair_sums
    
    # Initialize heuristic matrix
    # We use 0 for non-fitting items, which will be clamped to 1e-9 by the evaluator
    H = np.zeros((n, n), dtype=np.float64)
    
    # Mask for items that fit together
    fits = pair_sums <= capacity
    
    if np.any(fits):
        # Add a small epsilon to avoid division by zero for perfect fits
        eps = 1e-6
        valid_waste = waste[fits] + eps
        
        # Use 1/waste instead of 1/waste^2
        # This provides a smoother gradient, allowing pheromone to play a larger role
        # and potentially escaping local optima created by overly greedy pair selection.
        H[fits] = 1.0 / valid_waste
        
        # Ensure all values are finite and positive
        H[~np.isfinite(H)] = 1e-9
        H[H <= 0] = 1e-9

    # Time guard check
    if time.time() > deadline:
        # If we are running out of time (unlikely for this simple calculation), return what we have
        pass
        
    return H
