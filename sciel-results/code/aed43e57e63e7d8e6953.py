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
    
    # Mask for items that fit together
    fits = pair_sums <= capacity
    
    # Count how many items each item can pair with (compatibility degree)
    # For item j, how many items i can it fit with?
    compat_count = np.sum(fits, axis=0)  # For each item j, how many i can fit with j
    
    # Initialize heuristic matrix
    H = np.zeros((n, n), dtype=np.float64)
    
    if np.any(fits):
        # Base heuristic: 1/waste (smooth, allows pheromone to play a role)
        eps = 1e-6
        base_heuristic = np.zeros((n, n), dtype=np.float64)
        base_heuristic[fits] = 1.0 / (waste[fits] + eps)
        
        # Rarity bonus: items with fewer compatible partners should be prioritized
        # When placing j after i, if j is hard to place (low compat_count[j]), boost the heuristic
        rarity_bonus = 1.0 / (compat_count + 1.0)  # Higher for items with fewer options
        
        # Combine: base heuristic weighted by rarity of the next item j
        # H[i,j] = base_heuristic[i,j] * rarity_bonus[j]
        H[fits] = base_heuristic[fits] * rarity_bonus[None, :]
        
        # Ensure all values are finite and positive
        H[~np.isfinite(H)] = 1e-9
        H[H <= 0] = 1e-9

    # Time guard check
    if time.time() > deadline:
        pass
        
    return H
