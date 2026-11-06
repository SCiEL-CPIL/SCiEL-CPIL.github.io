# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin: evaluator kills at 90s
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    MAX_N = 1000000
    MIN_N = 1000
    WINDOW_SIZE = 24
    
    # We need small parts for numbers up to MAX_N + WINDOW_SIZE
    M = MAX_N + WINDOW_SIZE + 10
    
    # Precompute log of smooth parts
    # L[i] = sum_{p < 24} v_p(i) * ln(p)
    L = np.zeros(M + 1, dtype=np.float64)
    
    # Precompute indices once
    idx = np.arange(M + 1)
    
    for p in primes:
        log_p = np.log(p)
        power = p
        while power <= M:
            # Find all multiples of power
            mask = idx % power == 0
            L[mask] += log_p
            power *= p
            
    # Convert back to integer smooth parts
    # Using exp and round to recover integer
    sp = np.exp(L).astype(np.int64)
    
    # Map smooth part values to compact indices
    # We only care about distinct values in the window, so we can map the whole range
    # Using a dictionary for unique mapping is O(M)
    unique_vals = {}
    sp_idx = np.zeros(M + 1, dtype=np.int32)
    
    # Vectorized mapping is hard with dict, so we loop or use pandas if available (not guaranteed)
    # M is ~1e6, python loop is ~0.1-0.5s. Acceptable.
    for i in range(M + 1):
        v = int(sp[i])
        if v not in unique_vals:
            unique_vals[v] = len(unique_vals)
        sp_idx[i] = unique_vals[v]
    
    K = len(unique_vals)
    
    # Sliding window with frequency counter
    counts = np.zeros(K, dtype=np.int32)
    distinct = 0
    
    # Initialize counts for the first window: sp[MIN_N+1] .. sp[MIN_N+24]
    for i in range(1, WINDOW_SIZE + 1):
        idx_val = int(sp_idx[MIN_N + i])
        if counts[idx_val] == 0:
            distinct += 1
        counts[idx_val] += 1
        
    best_n = MIN_N
    best_distinct = distinct
    
    # Slide the window
    for n in range(MIN_N, MAX_N + 1):
        if distinct < best_distinct:
            best_distinct = distinct
            best_n = n
            if best_distinct <= 2:  # Very unlikely to go much lower
                break
        
        # Check time every 5000 iterations
        if (n - MIN_N) % 5000 == 0:
            if time.time() > deadline:
                break
        
        if n < MAX_N:
            # Remove sp[n+1]
            idx_out = int(sp_idx[n + 1])
            counts[idx_out] -= 1
            if counts[idx_out] == 0:
                distinct -= 1
            # Add sp[n+1+WINDOW_SIZE]
            idx_in = int(sp_idx[n + 1 + WINDOW_SIZE])
            if counts[idx_in] == 0:
                distinct += 1
            counts[idx_in] += 1
            
    return int(best_n)
# EVOLVE-BLOCK-END
