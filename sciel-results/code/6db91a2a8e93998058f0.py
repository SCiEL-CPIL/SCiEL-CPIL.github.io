"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import random

N = 100

def construct():
    # New approach: Binary local search with fallback to constant sequence.
    # We model h as binary {0, 1} to enforce bang-bang structure.
    # We use local search to minimize the maximum overlap.
    # If the binary solution is worse than the constant 0.5 solution, we return the constant one.
    
    def get_max_overlap(h):
        max_ov = 0.0
        for k in range(1, N):
            shifted = np.roll(h, -k)
            ov = np.sum(h * (1.0 - shifted))
            if ov > max_ov:
                max_ov = ov
        return max_ov

    # Constant sequence baseline
    h_const = np.full(N, 0.5)
    best_val = get_max_overlap(h_const)
    best_h = h_const.copy()

    # Binary local search
    # Initialize with random binary sequence with N/2 ones
    h_bin = np.zeros(N)
    ones_indices = random.sample(range(N), N // 2)
    h_bin[ones_indices] = 1.0
    
    current_val = get_max_overlap(h_bin)
    
    # Local search: swap a 1 and a 0
    for _ in range(2000):
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        
        # Ensure i is 1 and j is 0 (or vice versa) to maintain sum
        if h_bin[i] == h_bin[j]:
            continue
            
        # Swap
        h_bin[i], h_bin[j] = h_bin[j], h_bin[i]
        
        new_val = get_max_overlap(h_bin)
        
        if new_val < current_val:
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_h = h_bin.copy()
        else:
            # Revert
            h_bin[i], h_bin[j] = h_bin[j], h_bin[i]

    return best_h

# EVOLVE-BLOCK-END
