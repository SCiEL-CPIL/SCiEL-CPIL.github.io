"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing algorithm.
    
    The heuristic eta[i, j] represents the attractiveness of placing item j
    immediately after item i in the same bin.
    
    Strategy:
    We design the heuristic to guide ants toward bin configurations that:
    1. Fit within capacity.
    2. Maximize bin utilization (large combined demand).
    3. Avoid creating dead gaps (remaining space too small for any other item).
    4. Prefer pairing items that are hard to place elsewhere (large items).
    
    Formula:
    For fitting pairs (d_i + d_j <= capacity):
        remaining = capacity - (d_i + d_j)
        min_demand = min(demands)
        
        If remaining < min_demand (dead gap):
            eta[i,j] = 1e-9  (heavily penalize)
        Else:
            # Reward tight fits and large items
            # The term (d_i + d_j) rewards large combined demand
            # The term 1/(1 + remaining) rewards small waste
            # We also add a small bonus for items that are individually large
            eta[i,j] = (d_i + d_j) / (1.0 + remaining) * (1.0 + 0.1 * (d_i + d_j) / capacity)
    
    For non-fitting pairs: eta[i,j] = 1e-9
    """
    start_time = time.time()
    deadline = 325.0  # Safety margin before 330s limit
    
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Compute pairwise sums
    pair_sums = demands[:, None] + demands[None, :]
    
    # Determine which pairs fit in the bin
    fits = pair_sums <= capacity
    
    # Compute remaining space for fitting pairs
    remaining = capacity - pair_sums
    
    # Minimum demand among all items
    min_demand = np.min(demands)
    
    # Initialize heuristic matrix with penalty value
    eta = np.full((n, n), 1e-9, dtype=float)
    
    # Apply the heuristic for fitting pairs
    if np.any(fits):
        # Identify pairs that fit AND don't create a dead gap
        # A dead gap is when remaining space < min_demand
        valid_pairs = fits & (remaining >= min_demand)
        
        # For valid pairs, compute heuristic
        if np.any(valid_pairs):
            # Base heuristic: reward large combined demand and small waste
            # (d_i + d_j) / (1 + remaining)
            base = pair_sums[valid_pairs] / (1.0 + remaining[valid_pairs])
            
            # Additional bonus: prefer pairing large items together
            # This helps consolidate large items which are hardest to place
            large_bonus = 1.0 + 0.15 * (pair_sums[valid_pairs] / capacity)
            
            eta[valid_pairs] = base * large_bonus
        
        # Check time
        if time.time() > deadline:
            return eta
    
    return eta
# EVOLVE-BLOCK-END
