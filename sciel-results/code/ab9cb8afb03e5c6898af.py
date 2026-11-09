"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N = 1000000
    
    # Sieve to compute largest prime factor for each number up to N+1
    lpf = list(range(N + 2))  # lpf[i] = largest prime factor of i
    
    # For each prime p, mark all multiples with lpf = p
    # We process primes in increasing order, so later (larger) primes overwrite
    for p in range(2, N + 2):
        if lpf[p] == p:  # p is prime
            for multiple in range(p, N + 2, p):
                lpf[multiple] = p
    
    # Now scan for n in [10000, N] minimizing max(lpf[n], lpf[n+1])
    best_n = 10000
    best_val = max(lpf[10000], lpf[10001])
    
    for n in range(10000, N + 1):
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
