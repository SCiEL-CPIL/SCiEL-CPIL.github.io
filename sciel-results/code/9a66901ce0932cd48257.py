"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    """
    Computes a heuristic matrix for the ant colony optimizer.
    
    Strategy:
    1. Tight packing reward: 1/(remaining+1) favors pairs that fill the bin tightly.
    2. Dead-gap penalty: penalize remaining space that is too small to fit any item,
       preventing fragmentation.
    3. Pair-sum normalization for smooth gradient.
    
    eta[i, j] = tight_reward * gap_penalty + 0.5 * (pair_sum/capacity) if pair fits
    eta[i, j] = 1e-9 otherwise
    """
    d = np.asarray(demands, dtype=np.float64)
    n = len(d)
    
    # Calculate pair sums
    pair_sums = d[:, None] + d[None, :]
    
    # Determine which pairs fit in the bin
    fits = pair_sums <= capacity
    
    # Calculate remaining space for fitting pairs
    remaining = capacity - pair_sums
    
    # Initialize eta
    eta = np.full((n, n), 1e-9, dtype=np.float64)
    
    if np.any(fits):
        # For fitting pairs: combine tight-packing inverse with normalized pair sum
        tight_reward = 1.0 / (remaining[fits] + 1.0)
        normalized_sum = pair_sums[fits] / capacity
        
        # Dead-gap penalty: if remaining space is less than the smallest item,
        # it creates a useless gap. Penalize such gaps.
        min_item = d.min()
        rem_fits = remaining[fits]
        
        # Penalty: 1.0 if remaining >= min_item (usable gap), else decay
        # Use smooth penalty: min(1.0, remaining / min_item)
        gap_penalty = np.clip(rem_fits / min_item, 0.0, 1.0)
        
        # Blend: tight packing * gap penalty + normalized sum
        eta[fits] = tight_reward * gap_penalty + 0.5 * normalized_sum
    
    # Ensure all values are finite
    if not np.all(np.isfinite(eta)):
        eta = np.nan_to_num(eta, nan=1e-9, posinf=1e-9, neginf=1e-9)
    
    # Ensure no negative or zero values (non-positive becomes 1e-9)
    eta[eta <= 0] = 1e-9
    
    return eta
# EVOLVE-BLOCK-END
