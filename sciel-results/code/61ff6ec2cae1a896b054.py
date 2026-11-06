"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Range of n is [1000, 1000000]
    # We need to check n+1 to n+24
    # Max value to check: 1000000 + 24 = 1000024
    max_val = 1000024
    
    # Precompute masks for each number using NumPy
    masks = np.zeros(max_val + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        bit = 1 << i
        masks[p::p] |= bit
    
    n_start = 1000
    n_end = 1000000
    num_windows = n_end - n_start + 1
    
    if time.time() > deadline:
        return 1000
    
    # Build indices for all windows at once
    rows = np.arange(n_start, n_end + 1, dtype=np.int32)[:, None]
    cols = np.arange(1, 25, dtype=np.int32)[None, :]
    indices = rows + cols  # shape (num_windows, 24)
    
    if time.time() > deadline:
        return 1000
    
    window_masks = masks[indices]  # shape (num_windows, 24)
    
    if time.time() > deadline:
        return 1000
    
    # Compute union mask for each window using bitwise OR reduction
    union_masks = np.bitwise_or.reduce(window_masks, axis=1)
    
    if time.time() > deadline:
        return 1000
    
    # Count bits using a lookup table (union_masks are at most 9 bits, so < 512)
    lookup = np.array([bin(i).count('1') for i in range(512)], dtype=np.int32)
    counts = lookup[union_masks]
    
    if time.time() > deadline:
        return 1000
    
    best_idx = np.argmin(counts)
    best_n = n_start + int(best_idx)
    
    return best_n
# EVOLVE-BLOCK-END
