import numpy as np

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ant colony optimization.
    
    The heuristic is based on the 'complementarity' of item pairs.
    Pairs that sum to a value close to the capacity (leaving minimal waste)
    are assigned higher attractiveness.
    
    h_ij = 1 / (waste_ij + 1) if demands[i] + demands[j] <= capacity
    h_ij = 1e-9 otherwise
    
    where waste_ij = capacity - (demands[i] + demands[j])
    """
    n = len(demands)
    d = np.asarray(demands, dtype=float)
    
    # Calculate the sum of every pair of items
    # pair_sums[i, j] = d[i] + d[j]
    pair_sums = d[:, None] + d[None, :]
    
    # Calculate the waste for each pair
    # waste[i, j] = capacity - pair_sums[i, j]
    waste = capacity - pair_sums
    
    # Identify pairs that fit in the bin
    fits = pair_sums <= capacity
    
    # Initialize the heuristic matrix with the minimum value
    h = np.full((n, n), 1e-9, dtype=float)
    
    # For pairs that fit, calculate the complementarity score
    # We use 1 / (waste + 1) to avoid division by zero and to strongly
    # prefer pairs with zero waste (perfect fit).
    # The +1 ensures the denominator is at least 1.
    
    # We only calculate for the 'fits' mask to avoid unnecessary operations
    # and to ensure we don't compute values for non-fitting pairs.
    h[fits] = 1.0 / (waste[fits] + 1.0)
    
    # Ensure all values are finite and positive
    # The calculation above should guarantee this, but just in case
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h
