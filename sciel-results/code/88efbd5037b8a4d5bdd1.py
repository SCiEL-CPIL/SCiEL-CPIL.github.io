import time
import random
from math import isqrt, gcd

def construct():
    start_time = time.time()
    DEADLINE = 145.0

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

    # Pollard-Brent factorization (fallback for large residuals)
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

    def factor(n, factors):
        if n <= 1:
            return
        if is_prime(n):
            factors.append(n)
            return
        for p in small_primes[:200]:
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

    def max_prime_factor_leq(n, p_limit):
        """Return True if all prime factors of n are <= p_limit."""
        if n <= 1:
            return True
        temp = n
        for sp in small_primes:
            if sp > p_limit:
                break
            if temp % sp == 0:
                while temp % sp == 0:
                    temp //= sp
        if temp == 1:
            return True
        if is_prime(temp):
            return temp <= p_limit
        factors = []
        factor(temp, factors)
        return all(f <= p_limit for f in factors)

    def sieve_primes(limit):
        if limit < 2:
            return []
        s = bytearray(b'\x01') * (limit + 1)
        s[0] = s[1] = 0
        for i in range(2, isqrt(limit) + 1):
            if s[i]:
                s[i*i::i] = bytearray(len(s[i*i::i]))
        return [i for i in range(2, limit + 1) if s[i]]

    def find_max_k(p):
        """Use segmented sieve to find max k such that p^2+1..p^2+k are all p-smooth."""
        K = 200
        base = p * p
        
        # For small p, use direct trial division
        if p <= 100000:
            k = 0
            for i in range(1, K + 1):
                n = base + i
                if not max_prime_factor_leq(n, p):
                    break
                k += 1
            return k
        
        # Segmented sieve for [base+1, base+K]
        # Initialize residuals
        residuals = [base + i for i in range(1, K + 1)]
        
        # Sieve out all primes <= p
        for sp in small_primes:
            if sp > p:
                break
            # Find first multiple of sp in [base+1, base+K]
            start = ((base + 1 + sp - 1) // sp) * sp
            if start <= base:
                start += sp
            for j in range(start, base + K + 1, sp):
                idx = j - base - 1
                if idx >= 0 and idx < K:
                    while residuals[idx] % sp == 0:
                        residuals[idx] //= sp
        
        # Now check each residual
        k = 0
        for i in range(K):
            r = residuals[i]
            if r == 1:
                k += 1
                continue
            if is_prime(r):
                if r <= p:
                    k += 1
                    continue
            else:
                # Need to factor r
                if max_prime_factor_leq(r, p):
                    k += 1
                    continue
            break
        return k

    best_k = 0
    best_p = 2

    # Phase 1: Exhaustive search of small primes
    SMALL_P_LIMIT = 200000
    primes_small = sieve_primes(SMALL_P_LIMIT)
    for p in primes_small:
        if time.time() - start_time > DEADLINE - 5:
            break
        k = find_max_k(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Search larger primes with targeted sampling
    # For larger p, p-smoothness is more likely for small offsets
    LARGE_RANGES = [
        (200000, 1000000),
        (1000000, 10000000),
        (10000000, 100000000),
        (100000000, 1000000000),
    ]

    for lo, hi in LARGE_RANGES:
        if time.time() - start_time > DEADLINE - 5:
            break
        # Number of samples proportional to range size but bounded
        num_samples = max(20, min(2000, (hi - lo) // 100))
        for _ in range(num_samples):
            if time.time() - start_time > DEADLINE - 5:
                break
            p = random.randint(lo, hi)
            # Find next prime >= p
            if not is_prime(p):
                # Try a few nearby values
                found = False
                for offset in range(0, 200):
                    if p + offset > hi or p + offset > 10**9:
                        break
                    if is_prime(p + offset):
                        p = p + offset
                        found = True
                        break
                if not found:
                    continue
            if p > 10**9:
                continue
            k = find_max_k(p)
            if k > best_k:
                best_k = k
                best_p = p

    # Fallback
    if best_k == 0:
        best_p = 7
        best_k = 1

    return {'p': best_p, 'k': best_k}
