"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import Counter

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def construct():
    start_time = time.time()
    deadline = start_time + 88  # safety margin before 90s kill

    lower = 1000
    upper = 1000000
    max_val = upper + 24
    nvals = max_val - lower + 1

    # Precompute small-prime part for each number in [lower, max_val]
    sp_parts = np.ones(nvals, dtype=np.int64)
    arr = np.arange(lower, max_val + 1, dtype=np.int64)

    for p in _SMALL_PRIMES:
        # Vectorized: repeatedly divide by p and accumulate
        # For each power p^k, find indices divisible by p^k
        power = p
        while power <= max_val:
            mask = (arr % power == 0)
            if np.any(mask):
                sp_parts[mask] *= p
            power *= p

    # Now search for n in [lower, upper] such that the 24 values
    # sp_parts[n+1-lower], ..., sp_parts[n+24-lower] have minimum distinct count
    sp_list = sp_parts.tolist()

    best_n = lower
    best_count = 999

    # Use a frequency dictionary for the sliding window
    window = {}
    
    # Initial window for n=lower: sp_list[1] through sp_list[24]
    for i in range(1, 25):
        v = sp_list[i]
        window[v] = window.get(v, 0) + 1
    distinct = len(window)
    best_count = distinct
    if best_count == 1:
        return best_n

    # Slide the window with periodic time checks
    n_steps = upper - lower
    check_interval = 50000
    for offset in range(1, n_steps + 1):
        if offset % check_interval == 0 and time.time() > deadline:
            break
        
        # Remove sp_list[offset] (leaving the window)
        val_out = sp_list[offset]
        c = window[val_out] - 1
        if c == 0:
            del window[val_out]
        else:
            window[val_out] = c
        
        # Add sp_list[offset + 24] (entering the window)
        val_in = sp_list[offset + 24]
        window[val_in] = window.get(val_in, 0) + 1
        
        distinct = len(window)
        if distinct < best_count:
            best_count = distinct
            best_n = lower + offset
            if best_count == 1:
                return best_n

    return best_n
# EVOLVE-BLOCK-END
