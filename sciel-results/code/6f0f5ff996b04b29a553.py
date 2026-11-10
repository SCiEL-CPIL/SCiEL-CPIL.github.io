"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 60s limit
    
    N_MIN = 10000
    N_MAX = 1000000
    size = N_MAX + 2  # To check n+1 up to 1000001
    
    # Precompute largest prime factor for all numbers up to size
    # Using NumPy for speed
    lpf = np.zeros(size, dtype=np.int32)
    
    # Sieve for largest prime factor
    # Iterate p from 2 to size-1
    # If lpf[p] is 0, p is prime
    # Update lpf[j] = p for j in range(p, size, p)
    for p in range(2, size):
        if lpf[p] == 0:
            lpf[p::p] = p
            
    # Check deadline after sieve
    if time.time() > deadline:
        # Fallback: return a reasonable default
        return N_MIN
        
    # Now find min max(lpf[n], lpf[n+1]) for n in [10000, 1000000]
    # We can vectorize the scoring too.
    
    # Create array of n values
    n_range = np.arange(N_MIN, N_MAX + 1)
    
    # Calculate scores: max(lpf[n], lpf[n+1])
    scores = np.maximum(lpf[n_range], lpf[n_range + 1])
    
    # Find the index of the minimum score
    min_score_idx = np.argmin(scores)
    best_n = n_range[min_score_idx]
    
    return int(best_n)
# EVOLVE-BLOCK-END
