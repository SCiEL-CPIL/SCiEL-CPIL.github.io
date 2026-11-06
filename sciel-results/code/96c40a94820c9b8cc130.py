"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    N_MAX = 1000000
    N_MIN = 10000
    
    # Build smallest prime factor sieve up to N_MAX + 1
    size = N_MAX + 2
    spf = list(range(size))
    for i in range(2, int(math.isqrt(size)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, size, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute largest prime factor for all numbers using DP
    lpf = [1] * size
    for x in range(2, size):
        p = spf[x]
        y = x // p
        while y % p == 0:
            y //= p
        lpf[x] = max(p, lpf[y])
    
    best_n = N_MIN
    best_val = float('inf')
    
    # Search the full range with fast array lookups
    for n in range(N_MIN, N_MAX + 1):
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
            # Early termination: if we find a very smooth pair
            if best_val <= 2:
                break
    
    # Check deadline
    if time.time() > deadline:
        return best_n
    
    return best_n
# EVOLVE-BLOCK-END
