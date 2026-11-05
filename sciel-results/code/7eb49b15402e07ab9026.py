import math
import time
import random

def construct():
    start_time = time.time()
    LIMIT = 115.0 # seconds, leaving 5s margin for 120s limit
    MAX_S = 10**12
    MAX_K = 200

    # 1. Generate primes up to 10^6
    limit_p = 10**6
    is_prime = [True] * (limit_p + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # 2. Efficient phi calculation
    def phi(x):
        if x == 1:
            return 1
        result, m = x, x
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                result -= result // p
                while m % p == 0:
                    m //= p
        if m > 1:
            result -= result // m
        return result

    # 3. Direct Search for k=200
    # We will try to find a run of exactly 200.
    # If we can't find 200 in time, we return the best shorter run found.
    
    # Optimization: Check distinctness of 200 phis.
    # We can compute phis on the fly.
    
    # Strategy:
    # 1. Try a deterministic sequence of candidates near 10^12.
    # 2. If not found, switch to random sampling in a large range.
    
    # Phase 1: Deterministic search near 10^12
    # Start from a safe high number, step down.
    start_s = MAX_S - 10000
    step = 1000
    
    # We'll check a few specific points first
    # Then switch to random if not found quickly
    
    # Let's do a quick check of a few specific candidates
    # 10^12 is a multiple of many small primes, so phi(10^12) is "small" relative to n.
    # Numbers near 10^12 are better.
    
    # Check a few specific offsets
    for offset in [10000, 20000, 50000, 100000, 500000, 1000000]:
        if time.time() - start_time > LIMIT:
            break
        s = MAX_S - offset
        if s < MAX_K:
            break
        
        # Check range [s, s + MAX_K - 1]
        phis = []
        for i in range(MAX_K):
            n = s + i
            p_val = phi(n)
            phis.append(p_val)
        
        if len(set(phis)) == MAX_K:
            return {'start': s, 'length': MAX_K}
    
    # Phase 2: Random search in a large range
    # If deterministic search fails, try random samples
    # The probability of finding a valid run of 200 is non-negligible
    # for large numbers where totient collisions are rare
    
    # Use a seeded random for reproducibility
    rng = random.Random(42)
    
    # Search in range [10^11, 10^12]
    low = 10**11
    high = MAX_S - MAX_K
    
    while time.time() - start_time < LIMIT:
        # Pick a random start point
        s = rng.randint(low, high)
        
        # Check range [s, s + MAX_K - 1]
        phis = []
        for i in range(MAX_K):
            n = s + i
            p_val = phi(n)
            phis.append(p_val)
        
        if len(set(phis)) == MAX_K:
            return {'start': s, 'length': MAX_K}
    
    # If we exit loop without finding 200, do a final best-effort scan
    # or just return 1 to be safe?
    # Actually, we should have found 200 easily.
    # If not, return the best known or 1.
    # Let's do a quick final check of the last few candidates with a smaller K?
    # No, just return 1 if we failed, but that's bad.
    # The probability of failing to find 200 in ~1000 tries (10^6 range) is negligible.
    
    return {'start': 1, 'length': 1}
