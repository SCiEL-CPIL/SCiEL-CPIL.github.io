"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s deadline with safety margin
    
    N_MIN = 10000
    N_MAX = 1000000
    
    # Build smallest prime factor table up to N_MAX
    limit = N_MAX
    spf = list(range(limit + 1))
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute largest prime factor for all numbers up to N_MAX
    lpf = [0] * (limit + 1)
    lpf[0] = 0
    lpf[1] = 1
    for i in range(2, limit + 1):
        n = i
        max_pf = 1
        while n > 1:
            p = spf[n]
            max_pf = p
            while n % p == 0:
                n //= p
        lpf[i] = max_pf
    
    # Scan for n that minimizes max(lpf[n], lpf[n+1])
    best_n = N_MIN
    best_val = max(lpf[N_MIN], lpf[N_MIN + 1])
    
    for n in range(N_MIN, N_MAX):
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
        # Early termination: if we found 2, can't do better
        if best_val == 2:
            break
    
    return best_n
# EVOLVE-BLOCK-END
