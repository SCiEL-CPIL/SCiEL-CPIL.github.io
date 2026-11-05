"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # We need to compute the "small-prime part" for each number m
    # i.e., the largest divisor of m composed only of primes < 24
    max_n = 1000000
    max_val = max_n + 24
    
    # Precompute small-prime part for each number up to max_val
    # small_part[m] = product of p^a for p in small_primes where p^a || m
    
    # Use a fast sieve-like approach with NumPy vectorization
    small_part = np.ones(max_val + 1, dtype=np.int64)
    
    for p in small_primes:
        p_power = p
        while p_power <= max_val:
            # Multiply all indices that are multiples of p_power by p
            # This effectively adds one factor of p to the small-part
            small_part[p_power::p_power] *= p
            p_power *= p
            
    # Now small_part[m] is the complete factor of m composed of primes < 24
    
    # Extract the relevant slice for the sliding window
    # n ranges from 1000 to 1000000
    # For n, the window is small_part[n+1] to small_part[n+24]
    # Let S = small_part[1001 : 1000025]
    # Then for n=1000, window is S[0:24]
    # For n=1000000, window is S[999000:999024]
    
    S = small_part[1001:1000025]
    
    # Create a sliding window view
    # Number of windows is len(S) - 24 + 1
    # len(S) = 1000024
    # Windows = 1000001
    # But we only need n up to 1000000, which corresponds to index 999000 in S
    # So we take the first 999001 windows
    
    windows = np.lib.stride_tricks.sliding_window_view(S, 24)
    
    # We only need the first 999001 windows (for n=1000 to 1000000)
    # windows has shape (1000001, 24)
    # Let's slice to be safe, though it's exactly the right size
    # n=1000 -> idx 0
    # n=1000000 -> idx 999000
    # So we take windows[0:999001]
    
    target_windows = windows[0:999001]
    
    # To count distinct values in each row (window):
    # 1. Sort each row
    # 2. Count the number of changes + 1
    
    # Sorting 1M x 24 array might be memory intensive but doable.
    # 1M * 24 * 8 bytes = 192 MB.
    # Sorted copy adds another 192 MB.
    # Total ~400 MB.
    
    sorted_windows = np.sort(target_windows, axis=1)
    diffs = np.diff(sorted_windows, axis=1)
    distinct_counts = (diffs != 0).sum(axis=1) + 1
    
    min_idx = np.argmin(distinct_counts)
    best_n = 1000 + min_idx
    
    return best_n

# EVOLVE-BLOCK-END
