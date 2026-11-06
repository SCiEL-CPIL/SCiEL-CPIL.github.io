"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import Counter

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    lower = 1000
    upper = 1000000
    max_val = upper + 24
    nvals = max_val - lower + 1

    # Precompute small-prime part for each number in [lower, max_val] using NumPy
    sp_parts = np.ones(nvals, dtype=np.int64)
    arr = np.arange(lower, max_val + 1, dtype=np.int64)

    for p in _SMALL_PRIMES:
        mask = (arr % p == 0)
        if not np.any(mask):
            continue
        # For each multiple, compute the p-part
        # We need to find the highest power of p dividing each multiple
        # Use repeated division
        vals = arr[mask].copy()
        while np.any(vals % p == 0):
            vals[vals % p == 0] //= p
            sp_parts[mask] *= p

    # Convert to list for faster Counter operations in sliding window
    sp_list = sp_parts.tolist()

    # Sliding window of size 24 over sp_parts[n+1-lower .. n+24-lower]
    # For n in [lower, upper], window covers indices (n+1-lower) to (n+24-lower)
    # Let offset = n - lower, then window is sp_parts[offset+1 .. offset+24]
    
    best_n = lower
    best_count = float('inf')

    window_size = 24
    # Initial window for n=lower: indices 1 to 24
    window = Counter()
    for i in range(1, 25):  # indices 1..24 inclusive
        window[sp_list[i]] += 1
    distinct = len(window)
    if distinct < best_count:
        best_count = distinct
        best_n = lower

    # Slide: for n from lower+1 to upper
    for offset in range(1, upper - lower + 1):
        if time.time() > deadline:
            break
        # Remove index offset (which was sp_parts[offset], i.e., the element leaving)
        # Previous window for n=lower+offset-1 covered indices offset..offset+23
        # New window for n=lower+offset covers indices offset+1..offset+24
        # So we remove sp_parts[offset] and add sp_parts[offset+24]
        
        # Remove
        val_out = sp_list[offset]
        window[val_out] -= 1
        if window[val_out] == 0:
            del window[val_out]
        
        # Add
        val_in = sp_list[offset + 24]
        window[val_in] = window.get(val_in, 0) + 1
        
        distinct = len(window)
        if distinct < best_count:
            best_count = distinct
            best_n = lower + offset
            if best_count <= 2:
                pass  # keep scanning in case we find even lower

    return best_n
# EVOLVE-BLOCK-END
