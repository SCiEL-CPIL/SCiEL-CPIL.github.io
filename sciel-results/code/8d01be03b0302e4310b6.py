import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    MAX_VAL = N_MAX + WINDOW
    
    # Precompute extracted[m] for m in [1, MAX_VAL]
    extracted = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    for p in small_primes:
        multiples = np.arange(p, MAX_VAL + 1, p)
        temp = multiples.copy()
        p_parts = np.ones(len(multiples), dtype=np.int64)
        while np.any(temp % p == 0):
            mask = temp % p == 0
            temp[mask] //= p
            p_parts[mask] *= p
        extracted[multiples] *= p_parts
    
    # Map to IDs
    unique_vals, ids = np.unique(extracted, return_inverse=True)
    
    # Sliding windows
    num_windows = N_MAX - N_MIN + 1
    window_data = ids[N_MIN+1 : N_MAX+WINDOW+1]
    
    from numpy.lib.stride_tricks import as_strided
    stride = window_data.strides[0]
    window_2d = as_strided(window_data, shape=(num_windows, WINDOW), strides=(stride, stride))
    
    # Vectorized distinct count using first-occurrence method
    is_new = np.ones((num_windows, WINDOW), dtype=bool)
    is_new[:, 0] = True
    for j in range(1, WINDOW):
        equal = (window_2d[:, j:j+1] == window_2d[:, 0:j])
        is_new[:, j] = ~np.any(equal, axis=1)
    
    distinct_counts = is_new.sum(axis=1)
    
    best_idx = np.argmin(distinct_counts)
    
    return N_MIN + best_idx
