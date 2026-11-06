"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    lower = 1000
    upper = 1000000
    max_val = upper + 24
    N = max_val - lower + 1

    # Compute smooth part (factor composed of primes < 24) for each number
    # in [lower, max_val]
    # Method: for each number, divide out all prime factors >= 24
    # Equivalently: multiply together all p^k for p in {2,3,5,7,11,13,17,19,23}
    
    smooth = np.arange(lower, max_val + 1, dtype=np.int64)
    
    # For each small prime p, extract the p-adic part
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    result = np.ones(N, dtype=np.int64)
    
    for p in small_primes:
        # Extract p-power from each number
        temp = smooth.copy()
        factor = np.ones(N, dtype=np.int64)
        while True:
            mask = (temp % p) == 0
            if not np.any(mask):
                break
            temp[mask] //= p
            factor[mask] *= p
        result *= factor
    
    smooth = result
    # Now smooth[i] is the smooth part of (lower + i)
    
    # Encode smooth parts as compact integer IDs using a dictionary
    unique_vals = np.unique(smooth)
    val_to_id = {int(v): i for i, v in enumerate(unique_vals)}
    num_ids = len(unique_vals)
    
    # Convert to compact IDs
    smooth_ids = np.array([val_to_id[int(v)] for v in smooth], dtype=np.int32)
    
    # Sliding window: for each n in [lower, upper], look at smooth_ids[n+1-lower] to smooth_ids[n+24-lower]
    # Count distinct values in that window
    
    best_n = lower
    best_count = 25

    freq = np.zeros(num_ids, dtype=np.int32)
    
    # Initialize window for n = lower: indices 1 to 24 in smooth array
    window = smooth_ids[1:25]
    freq[window] += 1
    distinct = np.count_nonzero(freq)
    
    if distinct < best_count:
        best_count = distinct
        best_n = lower

    # Slide the window
    smooth_arr = smooth_ids
    l = lower
    u = upper
    time_check_interval = 10000
    
    for n in range(l + 1, u + 1):
        if (n - l) % time_check_interval == 0 and time.time() > deadline:
            break

        out_idx = n - l
        in_idx = n + 24 - l

        v_out = smooth_arr[out_idx]
        freq[v_out] -= 1
        if freq[v_out] == 0:
            distinct -= 1

        v_in = smooth_arr[in_idx]
        if freq[v_in] == 0:
            distinct += 1
        freq[v_in] += 1

        if distinct < best_count:
            best_count = distinct
            best_n = n
            # If we found the theoretical minimum (1 distinct value), we can stop early
            if best_count == 1:
                break

    return best_n
# EVOLVE-BLOCK-END
