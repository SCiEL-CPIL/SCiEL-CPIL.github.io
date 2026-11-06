"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin: evaluator kills at 90s
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    MAX_N = 1000000
    MIN_N = 1000
    WINDOW_SIZE = 24
    
    # We need small parts for numbers up to MAX_N + WINDOW_SIZE
    M = MAX_N + WINDOW_SIZE + 10
    
    # Precompute small parts for all numbers up to M
    # sp[x] will hold the product of prime factors < 24 of x
    sp = [1] * (M + 1)
    rem = list(range(M + 1))
    
    # Sieve-like computation for small parts
    for p in primes:
        for j in range(p, M + 1, p):
            while rem[j] % p == 0:
                sp[j] *= p
                rem[j] //= p
    
    # Map smooth part values to compact indices
    unique_vals = {}
    sp_idx = [0] * (M + 1)
    for i in range(M + 1):
        v = sp[i]
        if v not in unique_vals:
            unique_vals[v] = len(unique_vals)
        sp_idx[i] = unique_vals[v]
    
    K = len(unique_vals)
    counts = [0] * K
    distinct = 0
    
    # Initialize counts for the first window
    for i in range(1, WINDOW_SIZE + 1):
        idx = sp_idx[MIN_N + i]
        if counts[idx] == 0:
            distinct += 1
        counts[idx] += 1
        
    best_n = MIN_N
    best_distinct = distinct
    
    # Slide the window
    for n in range(MIN_N, MAX_N + 1):
        if distinct < best_distinct:
            best_distinct = distinct
            best_n = n
            
        if time.time() > deadline:
            break
        
        if n < MAX_N:
            # Remove the leftmost element: sp[n+1]
            idx_out = sp_idx[n + 1]
            counts[idx_out] -= 1
            if counts[idx_out] == 0:
                distinct -= 1
            # Add the new rightmost element: sp[n+25]
            idx_in = sp_idx[n + 1 + WINDOW_SIZE]
            if counts[idx_in] == 0:
                distinct += 1
            counts[idx_in] += 1
            
    return best_n
# EVOLVE-BLOCK-END
