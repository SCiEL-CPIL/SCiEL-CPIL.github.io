"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def construct():
    """
    Find n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    
    Strategy:
    1. Precompute the largest prime factor (LPF) for all numbers up to 1,000,001
       using a vectorized sieve-like approach with NumPy.
    2. Scan all valid n in [10000, 1000000] using vectorized operations.
    3. For each n, compute max(LPF[n], LPF[n+1]).
    4. Track the n with the minimum such value.
    5. Return that n.
    
    Deadline guard: If precomputation or scan takes too long, return best found so far.
    """
    start_time = time.time()
    deadline = start_time + 55  # 5 second safety margin before 60s limit
    
    # We need LPF for numbers up to 1,000,001
    MAX_VAL = 1000001
    
    # Initialize LPF array
    lpf = np.zeros(MAX_VAL + 1, dtype=np.int64)
    
    # Sieve to compute largest prime factor using NumPy vectorization
    # For each i, if lpf[i] == 0, it's prime; mark all multiples
    for i in range(2, MAX_VAL + 1):
        if lpf[i] == 0:
            # i is prime; mark all multiples of i
            lpf[i::i] = i
        # Check time periodically
        if i % 50000 == 0 and time.time() > deadline:
            return 10000
    
    # Now scan for best n using vectorized operations
    # n ranges from 10000 to 1000000
    # We need max(lpf[n], lpf[n+1]) for each n
    n_start = 10000
    n_end = 1000000
    
    # Get LPF values for n and n+1
    lpf_n = lpf[n_start:n_end+1]       # lpf[10000] to lpf[1000000]
    lpf_n1 = lpf[n_start+1:n_end+2]    # lpf[10001] to lpf[1000001]
    
    # Compute max for each pair
    max_lpf = np.maximum(lpf_n, lpf_n1)
    
    # Find the index with minimum value
    min_idx = np.argmin(max_lpf)
    best_n = n_start + min_idx
    best_lpf_val = max_lpf[min_idx]
    
    return int(best_n)

# EVOLVE-BLOCK-END
