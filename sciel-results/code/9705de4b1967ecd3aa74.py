"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from math import gcd
import time

def construct():
    N = 512
    size = 511  # numbers 1..511
    
    # Precompute conflict matrix: conflict[i][j] = True if gcd(i+1, j+1) > 1
    conflict = np.zeros((size, size), dtype=bool)
    for i in range(size):
        a = i + 1
        for j in range(i + 1, size):
            b = j + 1
            if gcd(a, b) > 1:
                conflict[i][j] = True
                conflict[j][i] = True
    
    # Greedy initialization: pick from 511 down to 1
    chosen_mask = np.zeros(size, dtype=bool)
    for i in range(size - 1, -1, -1):  # a=511 down to 1
        if not np.any(conflict[i] & chosen_mask):
            chosen_mask[i] = True
    
    # Local search: try swaps to improve
    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < 100:
        improved = False
        chosen_indices = np.where(chosen_mask)[0]
        for idx_a in chosen_indices:
            a = idx_a + 1
            # Temporarily remove idx_a
            temp_mask = chosen_mask.copy()
            temp_mask[idx_a] = False
            
            # Find best candidate b to add
            candidate_indices = np.where(~temp_mask)[0]
            best_b = -1
            best_gain = 0.0
            for idx_b in candidate_indices:
                b = idx_b + 1
                # Check if b is coprime to all remaining chosen
                if np.any(conflict[idx_b] & temp_mask):
                    continue
                gain = 1.0 / (N - b) - 1.0 / (N - a)
                if gain > best_gain:
                    best_gain = gain
                    best_b = idx_b
            
            if best_b >= 0 and best_gain > 1e-12:
                chosen_mask[idx_a] = False
                chosen_mask[best_b] = True
                improved = True
                break  # Restart after one swap
    
    return [int(i + 1) for i in np.where(chosen_mask)[0]]
# EVOLVE-BLOCK-END
