"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from functools import lru_cache

# EVOLVE-BLOCK-START
def _largest_prime_factor(n):
    """Return the largest prime factor of n, with P(1)=1."""
    if n <= 1:
        return 1
    result = 1
    # trial division by 2
    while n % 2 == 0:
        result = 2
        n //= 2
    # trial division by odd numbers up to sqrt(n)
    f = 3
    while f * f <= n:
        while n % f == 0:
            result = f
            n //= f
        f += 2
    if n > 1:
        result = n
    return result

def construct():
    target = 123457
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # We want to find a, b > 0 with a+b = target, minimizing max(P(a), P(b)).
    # Strategy: try to make both a and b have only small prime factors.
    # This is a "smooth number" problem.
    # 
    # Approach: 
    # 1. Generate all "smooth" numbers (with small prime factors) up to target.
    # 2. For each smooth number a, check if b = target - a is also smooth.
    # 3. Track the best (a,b) pair minimizing max(P(a), P(b)).
    #
    # But generating all smooth numbers up to 123457 might be feasible.
    # Let's use a different approach: iterate a from 1 to target//2, compute P(a) and P(target-a),
    # track the minimum of max(P(a), P(target-a)).
    #
    # To speed this up, we can precompute P(n) for all n up to target using a sieve-like method.
    
    n_max = target
    # Sieve for largest prime factor
    lp = [0] * (n_max + 1)
    lp[1] = 1
    for i in range(2, n_max + 1):
        if lp[i] == 0:  # i is prime
            for j in range(i, n_max + 1, i):
                if lp[j] == 0:
                    lp[j] = i
                # Actually, we want the LARGEST prime factor. Let's redo.
    
    # Better: compute largest prime factor via sieve
    lp = [1] * (n_max + 1)
    for i in range(2, n_max + 1):
        if lp[i] == 1 and i > 1:
            # i might be prime or composite; we need to check
            pass
    
    # Let's do a proper sieve for largest prime factor
    # lp[n] = largest prime factor of n
    lp = [0] * (n_max + 1)
    lp[1] = 1
    for i in range(2, n_max + 1):
        if lp[i] == 0:
            # i is prime
            for j in range(i, n_max + 1, i):
                lp[j] = i  # since we go from small to large, the last prime that divides j is the largest
    
    # Now lp[n] gives the largest prime factor for n >= 2, and lp[1] = 1
    # Verify: lp[12] should be 3. 12 is divisible by 2, 3. When i=2, lp[12]=2. When i=3, lp[12]=3. Correct.
    
    best_val = float('inf')
    best_a = 1
    best_b = target - 1
    
    # Iterate a from 1 to target//2 (symmetric, so we only need half)
    half = target // 2
    for a in range(1, half + 1):
        b = target - a
        pa = lp[a]
        pb = lp[b]
        m = max(pa, pb)
        if m < best_val:
            best_val = m
            best_a = a
            best_b = b
            if best_val == 2:
                # Can't do better than 2 (both powers of 2, but sum is odd so impossible)
                break
        if time.time() > deadline:
            break
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
