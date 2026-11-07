"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    n = len(demands)
    pair = demands[:, None] + demands[None, :]
    fits = pair <= capacity
    
    # Calculate remaining space
    remaining = np.where(fits, capacity - pair, capacity)
    
    # Heuristic: Reward tight packing scaled by the size of the pair.
    # Larger pairs that fit tightly are more attractive to place early.
    # This helps reduce fragmentation and utilize bin capacity efficiently.
    # Add 1.0 to denominator to avoid division by zero and soften the peak.
    eta = np.where(fits, pair / (remaining + 1.0), 1e-9)
    
    # Ensure finite values
    eta = np.nan_to_num(eta, nan=1e-9, posinf=1e9, neginf=1e-9)
    
    return eta
# EVOLVE-BLOCK-END
