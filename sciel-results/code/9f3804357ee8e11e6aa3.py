"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    # Convert to numpy array for vectorized operations
    d = np.array(demands)
    
    # Calculate pair sums
    pair_sums = d[:, None] + d[None, :]
    
    # Calculate gaps (how much space is left)
    # Only valid if pair_sums <= capacity
    gaps = capacity - pair_sums
    
    # Initialize heuristic matrix
    # We want to favor pairs that fit and leave a small gap (Best Fit heuristic)
    # h = 1 / (1 + gap) for valid pairs
    # h = 1e-9 for invalid pairs
    
    h = np.full((n, n), 1e-9)
    
    # Mask for valid pairs
    valid_mask = pair_sums <= capacity
    
    # Calculate heuristic for valid pairs
    # Add 1 to denominator to avoid division by zero and keep values bounded
    h[valid_mask] = 1.0 / (1.0 + gaps[valid_mask])
    
    return h
# EVOLVE-BLOCK-END
