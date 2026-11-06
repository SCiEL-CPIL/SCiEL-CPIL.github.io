import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    
    # Range for S calculation needs to cover N_MAX + WINDOW
    MAX_VAL = N_MAX + WINDOW + 10
    
    # Primes < 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(primes)
    
    # We want S[x] = product of prime factors of x that are < 24.
    # Let L[x] = product of prime factors of x that are >= 24.
    # Then S[x] = x // L[x].
    # We can compute L[x] by starting with L[x] = x and dividing out all small primes.
    
    L = np.arange(1, MAX_VAL + 1, dtype=np.int64)
    
    # For each small prime p, divide L[m] by p as many times as possible
    for p in primes:
        while True:
            mask = (L % p == 0)
            if not np.any(mask):
                break
            L[mask] //= p
            
    # Now L[x] contains the part of x composed of primes >= 24.
    # The part composed of primes < 24 is S[x] = x // L[x].
    X = np.arange(1, MAX_VAL + 1, dtype=np.int64)
    S = X // L
    
    # We need to find n in [N_MIN, N_MAX] minimizing distinct count in S[n+1..n+WINDOW]
    
    best_n = N_MIN
    best_count = float('inf')
    
    # Sliding window with dictionary-based distinct count
    # Window for n: S[n+1] ... S[n+WINDOW]
    
    # Initial window for n = N_MIN
    counts = {}
    for i in range(N_MIN + 1, N_MIN + WINDOW + 1):
        v = S[i]
        counts[v] = counts.get(v, 0) + 1
        
    current_distinct = len(counts)
    best_count = current_distinct
    best_n = N_MIN
    
    if best_count == 1:
        return best_n
    
    # Slide window
    for n in range(N_MIN + 1, N_MAX + 1):
        # Remove S[n] (was at position n in window for n-1)
        val_out = S[n]
        counts[val_out] -= 1
        if counts[val_out] == 0:
            del counts[val_out]
            
        # Add S[n + WINDOW]
        val_in = S[n + WINDOW]
        counts[val_in] = counts.get(val_in, 0) + 1
        
        current_distinct = len(counts)
        
        if current_distinct < best_count:
            best_count = current_distinct
            best_n = n
            if best_count == 1:
                return best_n
                
        # Time check every 10000 iterations to reduce overhead
        if n % 10000 == 0 and time.time() > deadline:
            break
            
    return best_n
