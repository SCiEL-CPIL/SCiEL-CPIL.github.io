import numpy as np
import time

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i
    in the same bin. Larger is more attractive.
    
    Strategy:
    1. Prefer pairs that fit (sum <= capacity).
    2. Among fitting pairs, strongly prefer tight fits (minimal waste).
    3. Use a very sharp decay based on waste to prioritize near-perfect fits.
    4. Invalid pairs (sum > capacity) get 1e-9.
    5. Ensure all values are positive and finite.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=float)
    
    # Compute pair sums
    pair_sums = d[:, None] + d[None, :]
    
    # Valid mask: pair sum <= capacity
    valid = pair_sums <= capacity
    
    # Initialize heuristic matrix with small positive value for invalid entries
    eta = np.full((n, n), 1e-9, dtype=float)
    
    if not np.any(valid):
        return eta
    
    # For valid entries, compute waste
    waste = capacity - pair_sums
    waste_valid = waste[valid]
    
    # Use a much sharper decay to prioritize tight fits.
    # The previous version used 1 / (1 + waste/10)^2.
    # Let's try 1 / (1 + waste/3)^3 for a much stronger preference for low waste.
    # waste=0 -> 1.0
    # waste=3 -> 1/8 = 0.125
    # waste=6 -> 1/27 ≈ 0.037
    # waste=10 -> 1/125 = 0.008
    # This makes the ant strongly prefer items that fill the bin almost completely.
    
    values = 1.0 / (1.0 + waste_valid / 3.0) ** 3
    
    eta[valid] = values
    
    # Ensure no non-positive or non-finite values
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    return eta
