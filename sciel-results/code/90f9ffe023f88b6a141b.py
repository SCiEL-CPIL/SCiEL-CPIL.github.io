import math
import time

def construct():
    deadline = time.time() + 115.0
    
    # Deterministic Miller-Rabin for n < 2^64
    def is_prime(n):
        if n < 2:
            return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for sp in small_primes:
            if n == sp:
                return True
            if n % sp == 0:
                return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        # Witnesses sufficient for n < 2^64
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    # Pollard's rho (Brent variant) for finding a factor
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        y, c, m = 2, 1, 128
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
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g > 1:
                    break
        if g == n:
            # Retry with different c
            return pollard_brent(n)
        return g
    
    def factorize(n, factors=None):
        """Full factorization of n into prime factors."""
        if factors is None:
            factors = []
        if n == 1:
            return factors
        if is_prime(n):
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
        return max(factors)
    
    def all_factors_leq(n, p):
        """Check if all prime factors of n are <= p."""
        if n <= 1:
            return True
        # Quick check: if n itself is prime and > p, fail
        if is_prime(n):
            return n <= p
        # Try trial division by small primes first for speed
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        m = n
        for sp in small_primes:
            if sp > p:
                break
            if sp * sp > m:
                break
            while m % sp == 0:
                m //= sp
        if m == 1:
            return True
        if is_prime(m):
            return m <= p
        # Need full factorization
        factors = factorize(m)
        return all(f <= p for f in factors)
    
    # Sieve for primes up to a limit
    def sieve(limit):
        if limit < 2:
            return []
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0:2] = b'\x00\x00'
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
        return [i for i in range(2, limit + 1) if is_p[i]]
    
    best_k = 0
    best_p = 2
    
    # Check small primes first
    small_primes = sieve(10000)
    
    for p in small_primes:
        if time.time() > deadline:
            break
        k = 0
        p_sq = p * p
        for k_val in range(1, 201):
            if time.time() > deadline:
                break
            v = p_sq + k_val
            if not all_factors_leq(v, p):
                break
            k = k_val
        if k > best_k:
            best_k = k
            best_p = p
    
    # Now try larger primes using segmented sieve to find primes
    # For each prime p, we need to check p^2+1 ... p^2+k
    # Key insight: for a given p, p^2 + j for j=1..k
    # We want all prime factors of p^2+j to be <= p
    
    # Strategy: search in ranges of primes, using segmented sieve
    # For efficiency, we can skip many primes. Let's try primes in various ranges.
    
    # Let's try a segmented sieve approach to find primes in ranges
    def segmented_sieve(low, high):
        """Find primes in [low, high] using segmented sieve."""
        if high < 2:
            return []
        limit = int(math.isqrt(high)) + 1
        small_primes = sieve(limit)
        size = high - low + 1
        is_prime_seg = bytearray(b'\x01') * size
        if low == 1:
            is_prime_seg[0] = 0
        for p in small_primes:
            start = max(p * p, ((low + p - 1) // p) * p)
            for j in range(start, high + 1, p):
                is_prime_seg[j - low] = 0
        return [low + i for i in range(size) if is_prime_seg[i]]
    
    # Try larger primes in strategic ranges
    # The key is that for large p, p^2 + j is roughly p^2, so its factors
    # are roughly up to p^2. We need all factors <= p, meaning p^2+j must be p-smooth.
    
    # Let's try primes in ranges, starting from larger values
    ranges_to_try = []
    # Try primes around 10^4 to 10^6
    for start in [10000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000]:
        if time.time() > deadline:
            break
        end = min(start + 50000, 10**9)
        if start >= 10**9:
            break
        ranges_to_try.append((start, end))
    
    for (lo, hi) in ranges_to_try:
        if time.time() > deadline:
            break
        seg_primes = segmented_sieve(lo, hi)
        for p in seg_primes:
            if time.time() > deadline:
                break
            k = 0
            p_sq = p * p
            for k_val in range(1, 201):
                if time.time() > deadline:
                    break
                v = p_sq + k_val
                if not all_factors_leq(v, p):
                    break
                k = k_val
            if k > best_k:
                best_k = k
                best_p = p
    
    return {'p': best_p, 'k': best_k}
