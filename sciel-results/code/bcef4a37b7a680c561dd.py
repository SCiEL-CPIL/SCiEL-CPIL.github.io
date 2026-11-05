import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill

    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24

    # Max value we need: 1000000 + 24 = 1000024
    MAX_VAL = N_MAX + WINDOW

    # Precompute extracted[m] for m in [1, MAX_VAL]
    # extracted[m] = product of p^k for p < 24, p^k || m
    extracted = np.ones(MAX_VAL + 1, dtype=np.int64)

    for p in small_primes:
        # Sieve-style: for each power p^k, multiply extracted by p at each multiple
        power = p
        while power <= MAX_VAL:
            extracted[power::power] *= p
            power *= p

    # Build the full array of extracted values we need
    # Window for n uses indices n+1 .. n+WINDOW
    # So we need extracted[N_MIN+1 .. N_MAX+WINDOW]
    vals = extracted[N_MIN + 1 : N_MAX + WINDOW + 1]
    # vals[i] corresponds to extracted[N_MIN + 1 + i]
    # Window for n = N_MIN + j uses vals[j : j+WINDOW]

    n_windows = N_MAX - N_MIN + 1
    best_count = WINDOW + 1
    best_n = N_MIN

    # Use a more efficient approach: sort each window and count distinct
    # We'll process in larger chunks and use a different strategy
    
    # First, let's try a smarter approach: 
    # For each position, track the "signature" of the window
    # We can use the fact that we're looking for windows with few distinct values
    
    # Let's use a different vectorization: for each window, count distinct
    # by sorting and counting changes
    
    CHUNK = 100000
    for chunk_start in range(0, n_windows, CHUNK):
        if time.time() - start_time > DEADLINE:
            break
        chunk_end = min(chunk_start + CHUNK, n_windows)
        n_chunk = chunk_end - chunk_start
        
        # Extract the windows as a 2D array
        chunk_vals = vals[chunk_start : chunk_start + n_chunk + WINDOW - 1]
        
        # Create windows array using stride tricks
        windows = np.lib.stride_tricks.as_strided(
            chunk_vals,
            shape=(n_chunk, WINDOW),
            strides=(chunk_vals.strides[0], chunk_vals.strides[0])
        ).copy()
        
        # Sort each window along axis 1
        sorted_windows = np.sort(windows, axis=1)
        
        # Count distinct: 1 + number of positions where sorted_windows[:,j] != sorted_windows[:,j-1]
        distinct_counts = np.ones(n_chunk, dtype=np.int32)
        for j in range(1, WINDOW):
            distinct_counts += (sorted_windows[:, j] != sorted_windows[:, j-1]).astype(np.int32)
        
        min_idx = int(np.argmin(distinct_counts))
        min_val = int(distinct_counts[min_idx])
        
        if min_val < best_count:
            best_count = min_val
            best_n = N_MIN + chunk_start + min_idx
            if best_count <= 3:
                return best_n

    return best_n
