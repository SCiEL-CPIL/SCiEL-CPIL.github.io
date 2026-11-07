"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def construct():
    deadline = time.time() + 55  # safety margin before 60s/90s kill
    MAXN = 10_000_000
    TARGET = Fraction(3, 2)
    
    # Best result found so far
    best_a, best_b = 11, 6
    best_err = abs(Fraction(11, 6) - TARGET)
    
    # Generate primes up to 10^7 using sieve
    def sieve(n):
        is_prime = bytearray(b'\x01') * (n + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        primes = [i for i in range(2, n + 1) if is_prime[i]]
        return primes
    
    if time.time() > deadline - 5:
        return [int(best_a), int(best_b)]
    
    primes = sieve(MAXN)
    
    # For each prime, compute powers and sigma(p^k)
    # sigma(p^k) = (p^(k+1) - 1) // (p - 1)
    # We want pairs (a, b) such that sigma(a) == sigma(b) and a/b close to 3/2
    
    # Strategy: enumerate candidate values of sigma(n) that can be achieved,
    # then find pairs (a, b) with the same sigma value.
    # 
    # Alternative: iterate over b from 1 to MAXN, compute sigma(b),
    # then look for a = round(3*b/2) and check if sigma(a) == sigma(b).
    # But computing sigma for all n up to 10^7 is O(N log log N) which is ~30s in Python.
    #
    # Better: Use the fact that we only need sigma at specific points.
    # For each b, a_candidate = round(3*b/2). Check if a_candidate <= MAXN.
    # Compute sigma(b) and sigma(a_candidate) via trial division using primes.
    # This is O(sqrt(n)) per query, too slow for 10^7 queries.
    #
    # Meet-in-the-middle: 
    # 1. Precompute sigma for all n up to some bound B (e.g., 10^6) using sieve.
    # 2. For larger n, factorize using primes up to sqrt(MAXN) and compute sigma.
    # 3. Iterate b, compute a = round(3b/2), check sigma(a) == sigma(b).
    
    # Let's use a hybrid approach:
    # Precompute sigma up to 2*MAXN//2 = MAXN using a sieve method (fast in Python with arrays)
    
    # Actually, let's just compute sigma for all n up to MAXN using a modified sieve.
    # sigma[n] = sum of divisors of n
    # Initialize sigma = [1] * (MAXN + 1)  -- wait, that's not right.
    # sigma[n] starts at 1, then for each prime p, for each multiple m of p, add p^k terms.
    
    # Standard approach:
    # sigma = [0] * (MAXN + 1)
    # for p in primes:
    #     for multiple in range(p, MAXN+1, p):
    #         # add p^k * (1 + p + ... ) -- this is complex
    # 
    # Simpler: 
    # sigma[n] = sum_{d|n} d
    # We can do: for d in 1..MAXN: for multiple in d, 2d, 3d, ...: sigma[multiple] += d
    # This is O(N log N) which is ~10^7 * 16 ≈ 1.6*10^8 operations -- too slow in pure Python.
    
    # Use NumPy for speed:
    import numpy as np
    
    sigma_arr = np.zeros(MAXN + 1, dtype=np.int64)
    for d in range(1, MAXN + 1):
        sigma_arr[d::d] += d
        if time.time() > deadline - 10:
            break
    
    # Now search for best pair
    # For b from 1 to MAXN, a = round(3*b/2)
    # Check if a <= MAXN and a > 0 and sigma_arr[a] == sigma_arr[b]
    # Also check a != b (since |a/b - 3/2| = 0 is invalid, and a==b gives ratio 1)
    
    for b in range(1, MAXN + 1):
        a = round(3 * b / 2)
        if a < 1 or a > MAXN:
            continue
        if a == b:
            continue
        sa = sigma_arr[a]
        sb = sigma_arr[b]
        if sa == sb:
            err = abs(Fraction(a, b) - TARGET)
            if err < best_err:
                best_err = err
                best_a, best_b = a, b
        if time.time() > deadline - 5:
            break
    
    return [int(best_a), int(best_b)]
# EVOLVE-BLOCK-END
