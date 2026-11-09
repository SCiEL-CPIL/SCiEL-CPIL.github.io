import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = 50.0 # 50 seconds safety margin

    # 1. Determine max value for sieve
    max_val = max(hi for _, hi in P['gaps'])
    
    # 2. Precompute Least Prime Factor (LPF)
    # lpf[i] will store the least prime factor of i
    lpf = list(range(max_val + 1))
    # Sieve of Eratosthenes variant for LPF
    # We only need to iterate up to sqrt(max_val)
    for i in range(2, isqrt(max_val) + 1):
        if lpf[i] == i: # i is prime
            # Mark multiples of i
            # Start from i*i as smaller multiples have smaller prime factors
            for j in range(i * i, max_val + 1, i):
                if lpf[j] == j:
                    lpf[j] = i

    # 3. Find witnesses for each gap
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap = hi - lo
        # Search for m in (lo, hi) such that lpf[m] >= gap
        # We can iterate m from lo+1 to hi-1
        # If we find one, we stop for this gap
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                result[i] = m
                break

    return result
