"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin vs 90s kill

    lower = 1000
    upper = 1000000
    window_size = 24
    max_val = upper + window_size

    nvals = max_val - lower + 1

    # ---- Step 1: Compute the 23-smooth part of each integer in [lower, max_val] ----
    # Use a segmented sieve approach: for each small prime, find all multiples and divide
    sp_part = np.ones(nvals, dtype=np.int64)
    remaining = np.arange(lower, max_val + 1, dtype=np.int64)

    for p in _SMALL_PRIMES:
        # Find all indices where remaining is divisible by p
        # Use vectorized operations: find positions where remaining % p == 0
        # Then divide those positions by p and multiply sp_part by p
        # Repeat until no more divisions possible
        
        # Efficient approach: use the fact that if remaining[i] % p == 0,
        # we can divide. Use a while loop with numpy operations.
        while True:
            mask = (remaining % p == 0)
            count = np.count_nonzero(mask)
            if count == 0:
                break
            sp_part[mask] *= p
            remaining[mask] //= p

    if time.time() > deadline:
        return lower

    # ---- Step 2: Map smooth-part values to compact integer IDs ----
    uniq_vals, ids = np.unique(sp_part, return_inverse=True)
    n_unique = len(uniq_vals)
    ids = ids.astype(np.int32)

    if time.time() > deadline:
        return lower

    # ---- Step 3: Sliding window with O(1) frequency updates ----
    # For n in [lower, upper], window covers n+1 to n+24
    # In array terms (0-indexed from lower), this is positions (n-lower+1) to (n-lower+24)
    # So for n = lower + i, positions are i+1 to i+24
    
    max_start = min(nvals - window_size - 1, upper - lower)

    # Use Python lists for faster access in the tight loop
    freq = [0] * n_unique
    distinct_count = 0
    ids_list = ids.tolist()  # Convert to list for faster Python access

    # Initialize first window: i=0, positions 1 through 24
    for j in range(1, window_size + 1):
        idx = ids_list[j]
        if freq[idx] == 0:
            distinct_count += 1
        freq[idx] += 1

    best_n = lower
    best_count = distinct_count

    # Slide the window
    for i in range(1, max_start + 1):
        # Remove position i
        out_idx = ids_list[i]
        freq[out_idx] -= 1
        if freq[out_idx] == 0:
            distinct_count -= 1
        # Add position i + window_size
        in_idx = ids_list[i + window_size]
        if freq[in_idx] == 0:
            distinct_count += 1
        freq[in_idx] += 1

        if distinct_count < best_count:
            best_count = distinct_count
            best_n = lower + i
            if best_count <= 2:
                break

        if i % 100000 == 0:
            if time.time() > deadline:
                break

    return best_n
# EVOLVE-BLOCK-END
