"""Stronger program: fast Miller-Rabin + Pollard-Brent, strategic prime search."""
import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    # Deterministic Miller-Rabin for n < 2^64
    _MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in _MR_BASES:
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

    # Pollard-Brent
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(1, n - 1)
            c = random.randint(1, n - 1)
            m = random.randint(1, n - 1)
            g, q, r = 1, 1, 1
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
                    g = math.gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                while True:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
                    if g > 1:
                        break
            if g != n:
                return g

    def factor(n, result=None):
        """Return list of prime factors (with multiplicity)."""
        if result is None:
            result = []
        if n <= 1:
            return result
        if is_prime(n):
            result.append(n)
            return result
        d = pollard_brent(n)
        factor(d, result)
        factor(n // d, result)
        return result

    def all_factors_leq(n, limit):
        """Check if all prime factors of n are <= limit."""
        if n <= 1:
            return True
        # Quick check: if n itself is prime and > limit, fail
        if is_prime(n):
            return n <= limit
        # Try small factors first
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for sp in small_primes:
            if sp > limit:
                break
            while n % sp == 0:
                n //= sp
        if n == 1:
            return True
        if is_prime(n):
            return n <= limit
        # Use Pollard-Brent
        factors = factor(n)
        for f in factors:
            if f > limit:
                return False
        return True

    def sieve_primes(limit):
        """Sieve primes up to limit."""
        if limit < 2:
            return []
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i, is_p in enumerate(sieve) if is_p]

    # Generate primes up to a reasonable limit for search
    # For p <= 10^9, we can't sieve all. Use a segmented approach or just test candidates.
    # Strategy: 
    #   Phase 1: sieve primes up to 10^6, test each
    #   Phase 2: if time remains, test larger primes in ranges

    best_k = 0
    best_p = None

    # Phase 1: sieve up to 200000
    primes_small = sieve_primes(200000)
    
    for p in primes_small:
        if time.time() - start_time > DEADLINE:
            break
        k = 0
        base = p * p
        while k < 200:
            if time.time() - start_time > DEADLINE:
                break
            n = base + k + 1
            if not all_factors_leq(n, p):
                break
            k += 1
        if k > best_k:
            best_k = k
            best_p = p
            # If we found k=200, that's the max, stop
            if k >= 200:
                break

    # Phase 2: if time remains, search larger primes
    if time.time() - start_time < DEADLINE - 10:
        # Test primes in ranges: 200000-1000000, 1000000-10000000, etc.
        # For each range, generate candidate primes and test
        for lo, hi in [(200000, 1000000), (1000000, 5000000), (5000000, 20000000), (20000000, 100000000)]:
            if time.time() - start_time > DEADLINE - 5:
                break
            # Sample primes in this range
            step = max(1, (hi - lo) // 5000)  # test at most ~5000 candidates per range
            for p in range(lo + 1, hi, step):
                if time.time() - start_time > DEADLINE - 2:
                    break
                if not is_prime(p):
                    continue
                if p > 10**9:
                    break
                k = 0
                base = p * p
                while k < 200:
                    if time.time() - start_time > DEADLINE - 1:
                        break
                    n = base + k + 1
                    if not all_factors_leq(n, p):
                        break
                    k += 1
                if k > best_k:
                    best_k = k
                    best_p = p
                    if k >= 200:
                        break
            if best_k >= 200:
                break

    if best_p is None:
        best_p = 2
        best_k = 1

    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
