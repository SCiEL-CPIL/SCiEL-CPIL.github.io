import time
import random
from math import isqrt, gcd

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    # Precompute small primes up to 10^6 for fast trial division
    SIEVE_LIMIT = 10**6
    sieve = bytearray(b'\x01') * (SIEVE_LIMIT + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(SIEVE_LIMIT) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    small_primes = [i for i in range(2, SIEVE_LIMIT + 1) if sieve[i]]

    # Deterministic Miller-Rabin for n < 3.3 * 10^24
    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    def is_prime(n):
        if n < 2:
            return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n == p:
                return True
            if n % p == 0:
                return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in MR_BASES:
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

    # Pollard-Brent factorization
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
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = gcd(abs(x - ys), n)
            if g != n:
                return g
            # retry

    def factor(n, factors):
        """Recursively factor n, appending prime factors to list."""
        if n <= 1:
            return
        if is_prime(n):
            factors.append(n)
            return
        # Try small primes first
        for p in small_primes[:200]:  # first 200 small primes
            if n % p == 0:
                while n % p == 0:
                    factors.append(p)
                    n //= p
                if n > 1:
                    factor(n, factors)
                return
        d = pollard_brent(n)
        factor(d, factors)
        factor(n // d, factors)

    def is_p_smooth(n, p):
        """Return True if all prime factors of n are <= p."""
        if n <= 1:
            return True
        # Quick check: divide out all small primes <= p
        temp = n
        for sp in small_primes:
            if sp > p:
                break
            if temp % sp == 0:
                while temp % sp == 0:
                    temp //= sp
        if temp == 1:
            return True
        # If temp is prime, check if it's <= p
        if is_prime(temp):
            return temp <= p
        # Need full factorization
        factors = []
        factor(temp, factors)
        return all(f <= p for f in factors)

    best_k = 0
    best_p = 2

    # Sieve primes for candidate search
    def sieve_primes(limit):
        if limit < 2:
            return []
        s = bytearray(b'\x01') * (limit + 1)
        s[0] = s[1] = 0
        for i in range(2, isqrt(limit) + 1):
            if s[i]:
                s[i*i::i] = bytearray(len(s[i*i::i]))
        return [i for i in range(2, limit + 1) if s[i]]

    # Phase 1: Search small primes quickly using trial division
    # For small p, p^2+i is small, so trial division is fast
    SMALL_P_LIMIT = 100000
    primes_small = sieve_primes(SMALL_P_LIMIT)
    for p in primes_small:
        if time.time() - start_time > DEADLINE:
            break
        k = 0
        base = p * p
        while k < 200:
            if time.time() - start_time > DEADLINE:
                break
            n = base + k + 1
            if not is_p_smooth(n, p):
                break
            k += 1
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Search larger primes using randomized sampling
    # Larger p means larger p^2+i, harder to be smooth, but we need to check
    LARGE_SEARCH_RANGES = [
        (100000, 500000),
        (500000, 1000000),
        (1000000, 5000000),
        (5000000, 10000000),
        (10000000, 50000000),
        (50000000, 100000000),
        (100000000, 500000000),
        (500000000, 1000000000),
    ]

    for lo, hi in LARGE_SEARCH_RANGES:
        if time.time() - start_time > DEADLINE:
            break
        # Sample some primes in this range
        num_samples = max(10, min(500, (hi - lo) // 1000))
        for _ in range(num_samples):
            if time.time() - start_time > DEADLINE:
                break
            p = random.randint(lo, hi)
            # Find next prime >= p
            while not is_prime(p):
                p += 1
                if p > hi:
                    break
            if p > hi or p > 10**9:
                continue
            # Check how many consecutive numbers starting from p^2+1 are p-smooth
            k = 0
            base = p * p
            while k < 200:
                if time.time() - start_time > DEADLINE:
                    break
                n = base + k + 1
                if not is_p_smooth(n, p):
                    break
                k += 1
            if k > best_k:
                best_k = k
                best_p = p

    # If we didn't find anything, return a valid small one
    if best_k == 0:
        best_p = 7
        best_k = 1

    return {'p': best_p, 'k': best_k}
