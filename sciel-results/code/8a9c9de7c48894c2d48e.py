import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N_MIN = 10000
    N_MAX = 1000000
    
    # We need largest prime factor up to N_MAX + 1
    limit = N_MAX + 1
    
    # Initialize array for largest prime factors
    lpf = np.zeros(limit + 1, dtype=np.int32)
    
    # Sieve to find largest prime factors
    # Iterating p in increasing order ensures that for any composite m,
    # lpf[m] is updated by its prime factors in increasing order.
    # The last update comes from the largest prime factor.
    for p in range(2, limit + 1):
        if lpf[p] == 0:
            # p is prime
            # Set lpf for all multiples of p to p
            # This is O(limit/p)
            lpf[p::p] = p
            
    if time.time() > deadline:
        return N_MIN

    # Now scan for the best n
    best_n = N_MIN
    best_val = float('inf')
    
    # We can vectorize this scan partially or just loop.
    # Loop is fast enough for 1e6 iterations.
    # lpf[n] and lpf[n+1] are int32.
    
    # Extract slice for n from N_MIN to N_MAX
    # n ranges from N_MIN to N_MAX
    # We need max(lpf[n], lpf[n+1])
    
    # Let's use numpy for the scan as well
    # n_indices = np.arange(N_MIN, N_MAX + 1)
    # vals = np.maximum(lpf[n_indices], lpf[n_indices + 1])
    # min_idx = np.argmin(vals)
    # best_n = n_indices[min_idx]
    
    # This is very fast.
    n_indices = np.arange(N_MIN, N_MAX + 1)
    vals = np.maximum(lpf[n_indices], lpf[n_indices + 1])
    min_idx = np.argmin(vals)
    best_n = int(n_indices[min_idx])
    
    return best_n
