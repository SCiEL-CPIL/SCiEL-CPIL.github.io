import numpy as np

def heuristics(demands, capacity):
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Create pairwise sum matrix
    # demands[:, None] is (n, 1), demands[None, :] is (1, n)
    pair_sum = demands[:, None] + demands[None, :]
    
    # Calculate waste for each pair
    waste = capacity - pair_sum
    
    # Initialize heuristic matrix with a very small value for invalid pairs
    eta = np.full((n, n), 1e-9, dtype=float)
    
    # Identify valid pairs where sum <= capacity
    valid_mask = pair_sum <= capacity
    
    # For valid pairs, calculate 1 / (waste + 1)
    # We add 1 to avoid division by zero and to scale the values reasonably.
    # If waste is 0, eta = 1.0. If waste is large, eta is small.
    eta[valid_mask] = 1.0 / (waste[valid_mask] + 1.0)
    
    return eta
