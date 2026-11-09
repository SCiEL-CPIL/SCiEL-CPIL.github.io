import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin (90s kill limit)
    
    N = 1000000
    limit = N + 1
    
    lpf = np.zeros(limit + 1, dtype=np.int64)
    
    # NumPy sieve for Largest Prime Factor
    # We iterate through primes (lpf[i] == 0) and mark multiples.
    # Since we iterate i from 2 to limit, the last prime to divide a number
    # is its largest prime factor.
    sieve_complete = True
    for i in range(2, limit + 1):
        # Check time less frequently to reduce overhead
        if i % 10000 == 0 and time.time() > deadline:
            sieve_complete = False
            break
        if lpf[i] == 0:
            lpf[i::i] = i
            
    if not sieve_complete:
        # Fallback: linear scan with partial sieve
        # This is unlikely to be triggered given the speed of the NumPy sieve
        best_val = float('inf')
        best_n = 10000
        for n in range(10000, N + 1):
            a = int(lpf[n])
            b = int(lpf[n+1])
            if a == 0 or b == 0:
                continue
            val = a if a > b else b
            if val < best_val:
                best_val = val
                best_n = n
        return best_n
    
    # Vectorized scan for minimum max prime factor
    # lpf[10000:N+1] gives lpf(n) for n in [10000, 1000000]
    # lpf[10001:N+2] gives lpf(n+1) for n in [10000, 1000000]
    vals = np.maximum(lpf[10000:N+1], lpf[10001:N+2])
    min_idx = int(np.argmin(vals))
    best_n = 10000 + min_idx
    
    return best_n
