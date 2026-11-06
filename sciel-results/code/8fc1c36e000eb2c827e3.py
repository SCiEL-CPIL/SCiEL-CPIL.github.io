import time
import math

def construct():
    """
    Finds the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    """
    start_time = time.time()
    deadline = start_time + 60.0  # Keep a safety margin before the 90s kill
    
    limit = 1000000
    
    # 1. Compute Smallest Prime Factor (SPF) for all numbers up to limit
    spf = list(range(limit + 1))
    if limit >= 1:
        spf[0] = 0
        spf[1] = 1
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # 2. Compute phi(n) for all n up to limit
    phi = list(range(limit + 1))
    for i in range(2, limit + 1):
        if spf[i] == i:  # i is prime
            for j in range(i, limit + 1, i):
                phi[j] -= phi[j] // i

    # 3. Find the smallest n such that phi(n) % 120 == 0
    for n in range(1, limit + 1):
        if phi[n] % 120 == 0:
            return n
            
    # Fallback (should not happen if limit is sufficient)
    return 14400
