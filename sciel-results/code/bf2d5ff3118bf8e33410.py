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
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # remaining will be divided down to the non-smooth part
    # smooth_parts will accumulate the smooth part
    remaining = np.arange(lower, max_val + 1, dtype=np.int64)
    smooth_parts = np.ones(N, dtype=np.int64)
    
    for p in small_primes:
        mask = (remaining % p) == 0
        while np.any(mask):
            remaining[mask] //= p
            smooth_parts[mask] *= p
            mask = (remaining % p) == 0
            
    # Now smooth_parts[i] is the smooth part of (lower + i)
    
    # We will use the smooth part value as an index into a frequency array.
    # Max value is max_val.
    freq = np.zeros(max_val + 1, dtype=np.int32)
    
    best_n = lower
    best_count = 25
    
    # Initialize window for n = lower
    # Window covers indices 1 to 24 in smooth_parts (corresponding to lower+1 to lower+24)
    window_vals = smooth_parts[1:25]
    for v in window_vals:
        freq[v] += 1
        
    distinct = np.count_nonzero(freq)
    
    if distinct < best_count:
        best_count = distinct
        best_n = lower

    # Slide the window
    # n goes from lower+1 to upper
    # For a given n, window is [n+1, n+24]
    # In array indices (0-based, relative to lower):
    # n = lower + k
    # Window values are at indices k+1 to k+24
    # Previous window (n-1) was at indices k to k+23
    
    l = lower
    u = upper
    time_check_interval = 10000
    
    for n in range(l + 1, u + 1):
        if (n - l) % time_check_interval == 0 and time.time() > deadline:
            break

        k = n - l
        # Element leaving: index k (corresponds to n)
        v_out = smooth_parts[k]
        freq[v_out] -= 1
        if freq[v_out] == 0:
            distinct -= 1

        # Element entering: index k + 24 (corresponds to n + 24)
        v_in = smooth_parts[k + 24]
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
