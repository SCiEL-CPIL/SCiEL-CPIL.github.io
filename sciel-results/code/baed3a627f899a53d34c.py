import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing problem.
    Uses a high power of the pair fill ratio to strongly encourage
    combinations that fill the bin tightly.
    """
    start_time = time.time()
    # Safety margin: return simple heuristic if taking too long
    # Though this calculation is O(N^2) and very fast for N=120
    deadline = 300.0 
    margin = 10.0
    
    n = len(demands)
    if n == 0:
        return np.array([])
        
    # Vectorized sum of pairs
    # demands is 1D array
    sums = demands[:, None] + demands[None, :]
    
    # Valid mask: sum <= capacity
    valid = sums <= capacity
    
    # Initialize heuristic matrix to 0
    # Invalid entries will remain 0, which the evaluator converts to 1e-9
    heur = np.zeros((n, n), dtype=float)
    
    # Calculate fill ratio for valid pairs
    # Avoid division by zero (capacity is 150, so safe)
    ratios = np.zeros((n, n), dtype=float)
    ratios[valid] = sums[valid] / capacity
    
    # Apply power transformation
    # Higher power favors tighter fits more strongly
    power = 10.0
    heur[valid] = np.power(ratios[valid], power)
    
    # Check time
    if time.time() > deadline - margin:
        # Fallback: return a simpler heuristic or the current one
        # Since we are already done, just return
        pass
        
    return heur
