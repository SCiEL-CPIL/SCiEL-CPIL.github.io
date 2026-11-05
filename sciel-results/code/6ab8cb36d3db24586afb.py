import time
import math
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    MAX_VAL = N_MAX + WINDOW

    # Precompute extracted[m] for m in [1, MAX_VAL]
    # extracted[m] = product of p^k for p < 24 where p^k || m
    extracted = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    for p in small_primes:
        p_power = p
        # For each power of p, multiply into extracted for multiples
        while p_power <= MAX_VAL:
            for m in range(p_power, MAX_VAL + 1, p_power):
                if m % (p_power * p) != 0:  # only if p_power is the exact power
                    extracted[m] *= p
                    break
            p_power *= p
        
        # Simpler approach: for each multiple of p, divide out all p factors
        # Let's use a cleaner method
    # Reset and use a cleaner precomputation
    extracted = np.ones(MAX_VAL + 1, dtype=np.int64)
    for p in small_primes:
        for m in range(p, MAX_VAL + 1, p):
            val = m
            while val % p == 0:
                extracted[m] *= p
                val //= p

    # The window for n is extracted[n+1 : n+1+WINDOW]
    # We want distinct counts for n in [N_MIN, N_MAX]
    
    # Use chunked vectorized approach with larger chunks for better throughput
    CHUNK_SIZE = 100000  # number of windows to process per chunk
    best_n = N_MIN
    best_count = WINDOW + 1
    
    total_windows = N_MAX - N_MIN + 1
    
    # Import once
    from numpy.lib.stride_tricks import as_strided
    
    # Process in chunks
    for chunk_start in range(0, total_windows, CHUNK_SIZE):
        if time.time() - start_time > DEADLINE:
            break
        
        chunk_end = min(chunk_start + CHUNK_SIZE, total_windows)
        n_start = N_MIN + chunk_start
        n_end = N_MIN + chunk_end  # exclusive
        
        arr_start = n_start + 1
        arr_end = n_end + WINDOW
        
        chunk_len = chunk_end - chunk_start
        flat_arr = extracted[arr_start : arr_end]
        
        if flat_arr.size >= chunk_len + WINDOW - 1:
            row_stride = flat_arr.strides[0]
            windows_2d = as_strided(
                flat_arr,
                shape=(chunk_len, WINDOW),
                strides=(row_stride, row_stride)
            )
            
            # Count distinct values per row using sort + diff
            sorted_windows = np.sort(windows_2d, axis=1)
            changes = np.diff(sorted_windows, axis=1)
            distinct_counts = 1 + np.sum(changes != 0, axis=1)
            
            # Find all positions with count < best_count
            candidates = np.where(distinct_counts < best_count)[0]
            if len(candidates) > 0:
                local_min_idx = candidates[0]  # first occurrence of new best
                local_min_val = distinct_counts[local_min_idx]
                if local_min_val < best_count:
                    best_count = int(local_min_val)
                    best_n = n_start + int(local_min_idx)
                    if best_count == 1:
                        break
        else:
            for j in range(chunk_len):
                n = n_start + j
                vals = extracted[n+1 : n+1+WINDOW]
                dc = len(set(vals.tolist()))
                if dc < best_count:
                    best_count = dc
                    best_n = n
                    if best_count == 1:
                        break

    # If we have time left, do a focused search around the best found
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 10 and best_count > 2:
        # Search in a window around best_n
        search_radius = min(50000, N_MAX - best_n, best_n - N_MIN)
        for offset in range(-search_radius, search_radius + 1, 100):
            if time.time() - start_time > DEADLINE:
                break
            n_test = best_n + offset
            if n_test < N_MIN or n_test > N_MAX:
                continue
            vals = extracted[n_test+1 : n_test+1+WINDOW]
            dc = len(set(vals.tolist()))
            if dc < best_count:
                best_count = dc
                best_n = n_test
                if best_count == 1:
                    break

    return best_n
