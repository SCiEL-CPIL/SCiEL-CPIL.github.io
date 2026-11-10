"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    deadline = time.time() + 85  # safety margin before 90s limit
    N = 1000001  # we need up to 1000000, and n+1 up to 1000001
    
    # SPF sieve using a more efficient approach
    spf = list(range(N + 1))
    # Only need to check up to sqrt(N)
    limit = int(math.isqrt(N))
    for i in range(2, limit + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each number
    # P(n) = product of p^e for all prime factors p of n where e >= 2
    pow_part = [1] * (N + 1)
    for n in range(2, N + 1):
        x = n
        pp = 1
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                pp *= p ** e
        pow_part[n] = pp
    
    # Find n in [10000, 1000000] maximizing pow_part[n] * pow_part[n+1]
    best_n = 10000
    best_val = pow_part[10000] * pow_part[10001]
    
    for n in range(10000, 1000000):
        val = pow_part[n] * pow_part[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
