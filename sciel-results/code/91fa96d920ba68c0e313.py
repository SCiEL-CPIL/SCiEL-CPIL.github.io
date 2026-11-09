"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    deadline = time.time() + 135  # safety margin before 150s kill
    
    # Deterministic Miller-Rabin for n < 2^64
    def is_prime_mr(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        # Small primes check first for speed
        small_primes_check = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes_check:
            if n == p:
                return True
            if n % p == 0:
                return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        # Witnesses for n < 2^64
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randrange(1, n)
            c = random.randrange(1, n)
            m = 128
            g = r = q = 1
            x = ys = y
            while g == 1:
                x = y
                for _ in range(r):
                    y = (y * y + c) % n
                k = 0
                while k < r and g == 1:
                    ys = y
                    for _ in range(min(m, r - k)):
                        y = (y * y + c) % n
                        q = q * abs(x - y) % n
                    g = gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                while True:
                    ys = (ys * ys + c) % n
                    g = gcd(abs(x - ys), n)
                    if g > 1:
                        break
            if g != n:
                return g
    
    def factorize(n, factors=None):
        if factors is None:
            factors = []
        if n <= 1:
            return factors
        if is_prime_mr(n):
            factors.append(n)
            return factors
        d = pollard_brent(n)
        factorize(d, factors)
        factorize(n // d, factors)
        return factors
    
    def largest_prime_factor(n):
        """Return the largest prime factor of n."""
        if n <= 1:
            return 1
        factors = factorize(n)
        return max(factors) if factors else 1
    
    def all_factors_leq(n, limit):
        """Check if all prime factors of n are <= limit."""
        if n <= 1:
            return True
        # Quick trial division by small primes first
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
            while n % p == 0:
                n //= p
            if p > limit:
                return False
        if n == 1:
            return True
        # Use Miller-Rabin: if n is prime, check if n <= limit
        if is_prime_mr(n):
            return n <= limit
        # Otherwise factorize
        factors = factorize(n)
        return all(f <= limit for f in factors)
    
    # Sieve primes up to a limit for searching
    def sieve_primes(limit):
        if limit < 2:
            return []
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
        return [i for i in range(2, limit + 1) if is_p[i]]
    
    best_k = 0
    best_p = 0
    
    # Search strategy: try various prime ranges
    # For each prime p, check p^2+1, p^2+2, ..., p^2+k
    # We want to maximize k such that all have prime factors <= p
    
    # Start with smaller primes where we can check more candidates
    # Then move to larger primes
    
    # Phase 1: Search primes up to 100,000 with thorough checking
    primes_small = sieve_primes(100000)
    
    for p in primes_small:
        if time.time() > deadline:
            break
        if p <= best_p and p > best_p:
            pass
        # Check the run starting at p^2 + 1
        k = 0
        p2 = p * p
        while k < 200:
            if time.time() > deadline:
                break
            val = p2 + k + 1
            if all_factors_leq(val, p):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 2: Search larger primes in chunks, being more selective
    # For larger p, the values p^2+i are ~p^2, so they have fewer small factors
    # on average, but p is larger so the bound is more lenient
    
    # Try primes in ranges [100k, 1M], [1M, 10M] etc.
    # But factorization gets expensive, so be selective
    
    # For larger primes, we can use a probabilistic approach:
    # Check a few candidates first to see if there's potential
    
    search_ranges = [
        (100000, 500000),
        (500000, 2000000),
        (2000000, 10000000),
    ]
    
    for lo, hi in search_ranges:
        if time.time() > deadline:
            break
        # Generate primes in this range using segmented sieve
        # For simplicity, use a basic approach for the range
        if hi - lo > 5000000:
            # Too large for simple sieve, sample instead
            step = max(1, (hi - lo) // 10000)
            for p in range(lo, hi, step):
                if time.time() > deadline:
                    break
                if p % 2 == 0 and p != 2:
                    continue
                if p % 3 == 0 and p != 3:
                    continue
                if not is_prime_mr(p):
                    continue
                # Quick check: just verify p^2+1 and p^2+2
                p2 = p * p
                if all_factors_leq(p2 + 1, p) and all_factors_leq(p2 + 2, p):
                    k = 0
                    while k < 200 and time.time() < deadline:
                        if all_factors_leq(p2 + k + 1, p):
                            k += 1
                        else:
                            break
                    if k > best_k:
                        best_k = k
                        best_p = p
        else:
            # Segmented sieve for this range
            size = hi - lo + 1
            is_p = bytearray(b'\x01') * size
            # Mark composites using small primes
            limit = int(math.isqrt(hi))
            small_primes = sieve_primes(limit)
            for sp in small_primes:
                start = ((lo + sp - 1) // sp) * sp
                if start < sp * sp:
                    start = sp * sp
                for j in range(start - lo, size, sp):
                    is_p[j] = 0
            if lo <= 1:
                for j in range(min(2, size)):
                    is_p[j] = 0
            for i in range(size):
                if time.time() > deadline:
                    break
                if is_p[i]:
                    p = lo + i
                    p2 = p * p
                    k = 0
                    while k < 200 and time.time() < deadline:
                        if all_factors_leq(p2 + k + 1, p):
                            k += 1
                        else:
                            break
                    if k > best_k:
                        best_k = k
                        best_p = p
    
    # Ensure we have a valid result
    if best_k == 0:
        # Fallback: find any prime with at least k=1
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            if all_factors_leq(p*p + 1, p):
                best_k = 1
                best_p = p
                break
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
