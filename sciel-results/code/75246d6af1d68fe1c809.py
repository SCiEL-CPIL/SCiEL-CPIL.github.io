"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    # Sort demands and track original indices
    sorted_idx = np.argsort(d)
    sorted_d = d[sorted_idx]
    
    # Two-pointer sweep to find ideal complement for each item in sorted order
    # For each item at position i in sorted order, find the position j such that
    # sorted_d[i] + sorted_d[j] is closest to capacity (but <= capacity)
    ideal_partner = np.zeros(n, dtype=np.int64)
    j = n - 1
    for i in range(n):
        while j > i and sorted_d[i] + sorted_d[j] > capacity:
            j -= 1
        if j >= i:
            ideal_partner[i] = j
        else:
            ideal_partner[i] = 0  # no valid partner, will be handled below
    
    # Create a matrix of "ideality" scores based on how close each pair is to the ideal complement
    # For each item i, its ideal partner in sorted order is at position ideal_partner[i]
    # We want to give high scores to pairs (i, j) where j is close to ideal_partner[i] in sorted order
    
    # Build the heuristic matrix
    result = np.full((n, n), 1e-9, dtype=np.float64)
    
    # For each pair (i, j) that fits, compute attractiveness
    pair_sum = d[:, None] + d[None, :]
    fits = pair_sum <= capacity
    
    # Base score: reward tight fits
    waste = capacity - pair_sum
    base_score = np.where(fits, 1.0 / (waste + 1.0), 0.0)
    
    # Apply power to sharpen
    power = 6.0
    base_score = np.power(base_score, power)
    
    # Now add a global structural bonus: for each item i, boost its ideal partner
    # We need to map from original indices to sorted positions
    sorted_pos = np.zeros(n, dtype=np.int64)
    for pos, orig in enumerate(sorted_idx):
        sorted_pos[orig] = pos
    
    # For each item i, find its ideal partner in original index space
    ideal_partner_orig = np.zeros(n, dtype=np.int64)
    for i in range(n):
        pos_i = sorted_pos[i]
        partner_pos = ideal_partner[pos_i]
        ideal_partner_orig[i] = sorted_idx[partner_pos]
    
    # Add bonus for ideal partners
    for i in range(n):
        j = ideal_partner_orig[i]
        if fits[i, j]:
            result[i, j] = 1000.0  # Very high bonus for ideal complement
    
    # Combine base score with ideal partner bonus
    result = np.where(fits, base_score, 1e-9)
    
    # Ensure non-finite values are replaced
    result = np.where(np.isfinite(result), result, 1e-9)
    
    # Ensure all non-positive become 1e-9
    result = np.where(result > 0, result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
