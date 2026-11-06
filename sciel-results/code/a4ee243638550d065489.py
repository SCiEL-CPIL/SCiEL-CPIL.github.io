"""Search for n maximizing powerful_part(n)*powerful_part(n+1)."""
import time
import math
from functools import lru_cache

# EVOLVE-BLOCK-START

def _compute_powerful_parts(limit):
    """Sieve to compute powerful_part for every number up to limit.
    powerful_part(n) = product of p^e for each prime p where exponent e >= 2 in n's factorization."""
    pp = [1] * (limit + 1)
    # For each prime p, for each power p^e with e>=2, multiply into pp for multiples of p^e
    # that are divisible by p^e but we need the full p^e contribution.
    # Better: factor each number via smallest-prime-factor sieve, then extract powerful part.
    
    # Smallest prime factor sieve
    spf = list(range(limit + 1))
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:  # prime
            step = i
            start = i * i
            for j in range(start, limit + 1, step):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each number
    for n in range(2, limit + 1):
        m = n
        result = 1
        while m > 1:
            p = spf[m]
            cnt = 0
            while m % p == 0:
                m //= p
                cnt += 1
            if cnt >= 2:
                result *= p ** cnt
        pp[n] = result
    return pp


def construct():
    deadline = time.time() + 75  # safety margin before 90s kill
    limit = 1000000
    lower = 10000
    
    # Compute powerful parts for all numbers up to limit
    pp = _compute_powerful_parts(limit)
    
    # Scan for best consecutive pair
    best_n = lower
    best_val = 0
    for n in range(lower, limit):
        v = pp[n] * pp[n + 1]
        if v > best_val:
            best_val = v
            best_n = n
    
    return best_n

# EVOLVE-BLOCK-END
