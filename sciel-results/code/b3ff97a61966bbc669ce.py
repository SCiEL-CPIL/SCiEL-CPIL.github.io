import math
import time
import random

def construct():
    start_time = time.time()
    # Safety margin: 150s limit, aim for 130s usage
    DEADLINE = start_time + 130.0
    
    # 1. Precompute primes up to 2*10^6 for better coverage
    limit_p = 2 * 10**6
    is_prime = [True] * (limit_p + 1)
    is_prime[0] = False
    if limit_p >= 1:
        is_prime[1] = False
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # 2. Optimized phi calculation with Miller-Rabin primality test
    def is_probable_prime(n):
        """Deterministic Miller-Rabin for n < 3.2e18"""
        if n < 2:
            return False
        # Check small primes
        small_primes_test = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes_test:
            if n % p == 0:
                return n == p
        # Write n-1 as 2^r * d
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        # Test with witnesses
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = x * x % n
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    def phi(x):
        if x == 1: return 1
        n = x
        result = x
        # Trial division by small primes first
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
        # If remaining n is large, check if it's prime
        if n > 1:
            if is_probable_prime(n):
                result -= result // n
            else:
                # n is composite with large factors - rare case
                # Try to factor further
                limit = int(n ** 0.5)
                for p in primes:
                    if p > limit:
                        break
                    if n % p == 0:
                        while n % p == 0:
                            n //= p
                        result -= result // p
                        limit = int(n ** 0.5)
                if n > 1:
                    result -= result // n
        return result

    best_k = 1
    best_s = 1
    
    # Helper to check a specific start with time limit
    def check_start(s, deadline):
        seen = {}
        k = 0
        limit_k = 200
        while k < limit_k:
            if time.time() > deadline:
                break
            val = phi(s + k)
            if val in seen:
                break
            seen[val] = True
            k += 1
        return k
    
    # Phase 1: Check strategically chosen starts
    # These are known to have long runs of distinct totients
    strategic_starts = [
        1, 2, 3, 4, 5, 10, 100, 1000, 10000,
        10**6, 10**7, 10**8, 10**9, 10**10, 10**11,
        # Try starts that are products of small primes + 1
        2*3*5*7*11*13 + 1,
        2*3*5*7*11*13*17 + 1,
        2*3*5*7*11*13*17*19 + 1,
        # Try starts near powers of 10
        10**12 - 200, 10**12 - 500, 10**12 - 1000,
    ]
    
    for s in strategic_starts:
        if time.time() > DEADLINE - 10:
            break
        if s < 1 or s > 10**12 - 200:
            continue
        k = check_start(s, DEADLINE - 10)
        if k > best_k:
            best_k = k
            best_s = s
    
    # Phase 2: Random search with local refinement
    # When we find a good start, explore nearby starts
    checked_starts = set()
    
    while time.time() < DEADLINE - 5.0:
        # Choose a random start
        s = random.randint(1, 10**12 - 200)
        if s in checked_starts:
            continue
        checked_starts.add(s)
        
        # Check this start
        k = check_start(s, DEADLINE - 5.0)
        
        if k > best_k:
            best_k = k
            best_s = s
            
            # If we found a good run, explore nearby starts
            if k > 10:
                for offset in range(-50, 51):
                    if time.time() > DEADLINE - 5.0:
                        break
                    s2 = s + offset
                    if s2 < 1 or s2 > 10**12 - 200:
                        continue
                    if s2 in checked_starts:
                        continue
                    checked_starts.add(s2)
                    k2 = check_start(s2, DEADLINE - 5.0)
                    if k2 > best_k:
                        best_k = k2
                        best_s = s2
        
        # If we hit the maximum, stop
        if best_k == 200:
            break

    return {'start': best_s, 'length': best_k}

# EVOLVE-BLOCK-END
