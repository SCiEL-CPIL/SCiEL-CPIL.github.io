import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    N = 1000001  # we need up to 1000000+1
    
    # Sieve for largest prime factor
    lpf = list(range(N + 1))  # lpf[i] will hold largest prime factor of i
    # Actually, let's use a different approach
    # lpf[i] = largest prime factor of i
    # Initialize: for i >= 2, lpf[i] = i (assume prime)
    # Then for each prime p, for multiples, set lpf[m] = p
    # But we want LARGEST, so we need to be careful
    
    # Better: standard sieve
    # lpf[i] = 1 initially means "not yet set"
    # For p from 2 to N:
    #   if lpf[p] == 1: p is prime
    #     for m in range(p, N+1, p):
    #       lpf[m] = p
    # This gives largest prime factor because larger p overwrites
    
    lpf = [1] * (N + 1)
    for p in range(2, N + 1):
        if lpf[p] == 1:  # p is prime
            for m in range(p, N + 1, p):
                lpf[m] = p
    
    # Now find n in [10000, 1000000] that minimizes max(lpf[n], lpf[n+1])
    best_n = 10000
    best_val = max(lpf[10000], lpf[10001])
    
    for n in range(10001, 1000001):
        if time.time() > deadline:
            break
        val = max(lpf[n], lpf[n+1])
        if val < best_val:
            best_val = val
            best_n = n
    
    return best_n
