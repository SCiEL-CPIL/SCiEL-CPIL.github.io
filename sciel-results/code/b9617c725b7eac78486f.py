"""Improved program: faster factorization via pre-sieved primes, batched GCD, better search strategy."""
import math
import time
import random

def construct():
    deadline = time.time() + 145.0  # safety margin before 150s limit
    
    # Pre-sieve primes up to 100000 for fast trial division
    SIEVE_LIMIT = 100000
    is_p_arr = bytearray(b'\x01') * (SIEVE_LIMIT + 1)
    is_p_arr[0] = is_p_arr[1] = 0
    for i in range(2, int(SIEVE_LIMIT**0.5) + 1):
        if is_p_arr[i]:
            is_p_arr[i*i::i] = bytearray(len(range(i*i, SIEVE_LIMIT+1, i)))
    small_primes = [i for i in range(2, SIEVE_LIMIT+1) if is_p_arr[i]]
    
    # Deterministic Miller-Rabin for n < 2^64
    _witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        for p in _witnesses:
            if n == p:
                return True
            if n % p == 0:
                return False
        d = n - 1
        s = 0
        while d & 1 == 0:
            d >>= 1
            s += 1
        for a in _witnesses:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    # Pollard's Rho with Brent's cycle detection
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        for attempt in range(50):
            y = random.randint(2, n - 1)
            c = random.randint(1, n - 1)
            m = 128
            g, r, q = 1, 1, 1
            while g == 1:
                x = y
                for _ in range(r):
                    y = (y * y + c) % n
                k = 0
                while k < r and g == 1:
                    ys = y
                    for _ in range(min(m, r - k)):
                        y = (y * y + c) % n
                        q = (q * abs(x - y)) % n
                    g = math.gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                found = False
                while True:
                    ys = (ys * ys + c) % n
                    g2 = math.gcd(abs(x - ys), n)
                    if g2 > 1 and g2 < n:
                        return g2
                    if g2 == n:
                        break
                    if not found and abs(x - ys) == 0:
                        found = True
                # If g==n, retry
                continue
            if g != n and g > 1:
                return g
        # Fallback: return n (failure)
        return n
    
    # Check if all prime factors of v are <= limit
    def all_factors_leq(v, limit):
        if v <= 1:
            return True
        if limit >= v:
            return True  # v itself is at most limit
        # Quick: if v is prime, check directly
        if is_prime(v):
            return v <= limit
        n = v
        # Trial division by pre-sieved primes up to min(SIEVE_LIMIT, limit)
        # Only need primes up to limit
        for p in small_primes:
            if p > limit:
                break
            if p * p > n:
                break
            while n % p == 0:
                n //= p
            if n == 1:
                return True
        if n == 1:
            return True
        if n <= limit:
            # n might be prime or composite with factors <= limit
            # If n is prime, it's <= limit, so OK
            # If n is composite, its factors are <= sqrt(n) <= sqrt(limit) < limit, so OK
            # Actually we need to verify n has no factor > limit
            # Since n <= limit, and we've divided out all primes <= limit that divide n,
            # n is now either 1 or has no prime factor <= limit... wait no.
            # We divided out all primes p where p <= limit and p <= sqrt(n at that point).
            # Hmm, let me just check if n is prime
            if is_prime(n):
                return n <= limit
            # n is composite but <= limit, so all its factors are < n <= limit
            return True
        # n > limit at this point; n has no prime factor <= min(limit, sqrt(n))
        # from trial division. If n is prime, it must be > limit -> fail
        if is_prime(n):
            return False
        # n is composite with all prime factors > limit (since we divided out all <= limit)
        # Wait, that's not right either. We divided out primes up to min(limit, sqrt(n_initial)).
        # Let's just factor n with Pollard and check
        stack = [n]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if is_prime(m):
                if m > limit:
                    return False
            else:
                d = pollard_brent(m)
                if d == m:
                    # Pollard failed; try trial division a bit more
                    found_factor = False
                    for p in small_primes:
                        if p > limit:
                            break
                        if p * p > m:
                            break
                        if m % p == 0:
                            while m % p == 0:
                                m //= p
                            if m > 1:
                                stack.append(m)
                            found_factor = True
                            break
                    if not found_factor:
                        # Can't factor, assume it has a large factor
                        return False
                else:
                    stack.append(d)
                    stack.append(m // d)
        return True
    
    # Sieve primes up to a given limit
    def sieve(limit):
        if limit < 2:
            return []
        arr = bytearray(b'\x01') * (limit + 1)
        arr[0] = arr[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if arr[i]:
                arr[i*i::i] = bytearray(len(range(i*i, limit+1, i)))
        return [i for i in range(2, limit+1) if arr[i]]
    
    best_k = 0
    best_p = 2
    
    # Strategy: First quickly scan small primes (they're likely to give short runs),
    # then focus on mid-range primes where longer runs might exist.
    # The key insight: for larger p, p^2 is larger, so p^2+i has factors that are
    # more likely to be <= p. We want to maximize k.
    
    # Quick check of p=2
    k = 0
    while time.time() < deadline:
        v = 4 + k + 1
        if all_factors_leq(v, 2):
            k += 1
        else:
            break
    if k > best_k:
        best_k = k
        best_p = 2
    
    # Scan primes up to 200000
    primes_small = sieve(200000)
    for p in primes_small:
        if time.time() >= deadline:
            break
        if p == 2:
            continue
        # Quick early check: if p^2+1 has a prime factor > p, skip this p
        # p^2+1 is odd, so not divisible by 2
        v = p * p + 1
        if not all_factors_leq(v, p):
            continue
        # Now count the run
        k = 1
        while time.time() < deadline:
            v = p * p + k + 1
            if all_factors_leq(v, p):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
    
    # Scan primes from 200000 to 2000000
    if time.time() < deadline - 5:
        primes_mid = sieve(2000000)
        start_idx = len(primes_small)
        for p in primes_mid[start_idx:]:
            if time.time() >= deadline:
                break
            # Quick early check
            v = p * p + 1
            if not all_factors_leq(v, p):
                continue
            k = 1
            while time.time() < deadline:
                v = p * p + k + 1
                if all_factors_leq(v, p):
                    k += 1
                else:
                    break
            if k > best_k:
                best_k = k
                best_p = p
    
    # If we have lots of time left, scan larger primes
    if time.time() < deadline - 30:
        primes_large = sieve(20000000)
        start_idx = len(primes_mid)
        for p in primes_large[start_idx:]:
            if time.time() >= deadline:
                break
            v = p * p + 1
            if not all_factors_leq(v, p):
                continue
            k = 1
            while time.time() < deadline:
                v = p * p + k + 1
                if all_factors_leq(v, p):
                    k += 1
                else:
                    break
            if k > best_k:
                best_k = k
                best_p = p
    
    return {'p': best_p, 'k': best_k}
