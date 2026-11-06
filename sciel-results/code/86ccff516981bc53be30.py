"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Find n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    
    Strategy: Use a vectorized NumPy sieve to compute largest prime factor (LPF)
    for every integer up to 1,000,001. Then scan all candidates n in [10000, 1000000]
    and track the minimum of max(lpf[n], lpf[n+1]).
    """
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline with 5s safety margin
    
    N_MIN = 10000
    N_MAX = 1000000
    N_MAX_PLUS_1 = N_MAX + 1
    SIZE = N_MAX_PLUS_1 + 1  # 1000002
    
    # Vectorized LPF sieve using NumPy
    lpf = np.zeros(SIZE, dtype=np.int64)
    
    for p in range(2, SIZE):
        if time.time() > deadline:
            break
        if lpf[p] == 0:  # p is prime
            lpf[p::p] = p
    
    # Check if sieve completed
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        # Fallback: use partial results or default
        # If sieve didn't complete, do a quick scan with trial division on a subset
        best_n = N_MIN
        best_val = float('inf')
        for n in range(N_MIN, min(N_MIN + 100000, N_MAX + 1)):
            if time.time() > deadline:
                break
            # Simple trial division for LPF
            def lpf_trial(x):
                if x <= 1:
                    return 1
                max_pf = 1
                d = 2
                while d * d <= x:
                    while x % d == 0:
                        max_pf = max(max_pf, d)
                        x //= d
                    d += 1
                return max(max_pf, x)
            val = max(lpf_trial(n), lpf_trial(n + 1))
            if val < best_val:
                best_val = val
                best_n = n
        return best_n
    
    # Vectorized scan
    n_arr = np.arange(N_MIN, N_MAX + 1, dtype=np.int64)
    max_lpf = np.maximum(lpf[n_arr], lpf[n_arr + 1])
    min_idx = np.argmin(max_lpf)
    best_n = int(n_arr[min_idx])
    
    return best_n
# EVOLVE-BLOCK-END
