import numpy as np

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing algorithm.
    
    The heuristic eta[i, j] represents the attractiveness of placing item j
    immediately after item i in the same bin.
    
    Strategy:
    We use a 'Best Fit' inspired heuristic. We want to fill bins as tightly as
    possible. Therefore, a pair (i, j) is attractive if:
    1. They fit in the bin (w_i + w_j <= capacity).
    2. The remaining space (gap) after placing them is small.
    3. The item j is large (consuming more space is generally better to avoid
       having large items stranded later).
    
    Formula: eta[i, j] = w_j / (gap + epsilon)
    where gap = capacity - w_i - w_j.
    """
    W = np.array(demands, dtype=float)
    n = len(W)
    
    # Calculate sums of all pairs
    pair_sums = W[:, None] + W[None, :]
    
    # Calculate gaps
    gaps = capacity - pair_sums
    
    # Identify valid pairs
    fits = gaps >= 0
    
    # Initialize heuristic matrix with a very small value for invalid pairs
    H = np.full((n, n), 1e-9)
    
    if np.any(fits):
        # Add epsilon to avoid division by zero for exact fits
        # Exact fits should have the highest heuristic value
        eps = 1e-9
        safe_gaps = np.where(fits, gaps + eps, 1.0)
        
        # Numerator: size of the item being placed (j)
        # W[None, :] creates a row vector of W, which broadcasts to (n, n)
        # H[i, j] will use W[j] as the numerator
        numerator = W[None, :]
        
        # Calculate heuristic for valid pairs
        H_valid = numerator / safe_gaps
        
        # Set invalid pairs to 1e-9
        H = np.where(fits, H_valid, 1e-9)
        
    # Final safety checks: ensure finite and positive
    H = np.where(np.isfinite(H), H, 1e-9)
    H = np.where(H > 0, H, 1e-9)
    
    return H
