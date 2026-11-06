"""Rewritten program: segmented sieve for batch p-smoothness checking."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s limit
    t0 = time.time()
    
    # Deterministic Miller-Rabin for 64-bit integers
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
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
    
    # Pollard's Rho with Brent's cycle detection
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        while True:
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
    
    # Sieve primes up to limit
    def sieve(limit):
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                step = i
                start = i * i
                is_p[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))
        return [i for i in range(2, limit+1) if is_p[i]]
    
    # Batch check: for a given prime p, check how many consecutive integers
    # p^2+1, p^2+2, ..., p^2+k are all p-smooth (all prime factors <= p).
    # Uses segmented sieve approach: for each prime q <= p, divide out all
    # occurrences from the range. If after processing all such primes, the
    # remaining values are all 1, they are p-smooth.
    #
    # Optimization: we only need to process primes q where q <= p AND q appears
    # as a factor in some value in [p^2+1, p^2+200]. For large p, most values
    # will have their factors quickly removed by small primes.
    #
    # Strategy: For each value v in [p^2+1, p^2+200], we need to verify all
    # prime factors are <= p. We do this by:
    # 1. Trial division by primes up to min(p, some_limit)
    # 2. If remainder is 1 -> p-smooth
    # 3. If remainder is prime and <= p -> p-smooth  
    # 4. Otherwise -> not p-smooth (has a prime factor > p)
    #
    # Key insight: if p is large (say > 200), then for v = p^2 + j where j <= 200,
    # any prime factor q of v must satisfy q <= v. But we need q <= p.
    # Since v = p^2 + j, the largest possible prime factor is v itself (if v is prime).
    # v is prime and v > p (since v = p^2 + j > p for p >= 2, j >= 1).
    # So if v is prime, it's NOT p-smooth (for p >= 2).
    # 
    # Wait, that's not right. v = p^2 + j. If v is prime, then v > p (for p >= 2, j >= 1),
    # so v has a prime factor (itself) that is > p. So v is not p-smooth.
    #
    # This means: for any p, if p^2 + j is prime for some j in [1, k], then the run breaks.
    # So we need p^2+1, ..., p^2+k to all be composite with all prime factors <= p.
    #
    # For a given p, let's check each v = p^2 + j:
    # - If v is prime: not p-smooth (v > p)
    # - If v is composite: need all prime factors <= p
    #
    # For v = p^2 + j with j small, v is roughly p^2. Its prime factors could be:
    # - Small factors (trial division handles these)
    # - Large factors: if v = a * b with a, b both > p, then v > p^2, contradiction since v = p^2 + j.
    #   Actually v = p^2 + j, so if both factors > p, then a*b > p*p = p^2, which is possible if j > 0.
    #   E.g., v = (p+1)(p-1) + 2j... no. Let's think: v = p^2 + j. If v = a*b with a <= b,
    #   then a <= sqrt(v) < p+1 (for j < 2p+1). So a <= p. 
    #   Therefore, the smaller prime factor of any composite v = p^2 + j (with j < 2p) is <= p.
    #   The larger factor is v/a which could be > p.
    #
    # So for v = p^2 + j where j <= 200 and p > 100 (so 200 < 2p):
    # - If v is prime: NOT p-smooth
    # - If v is composite: let q be the smallest prime factor. q <= sqrt(v) < p+1, so q <= p.
    #   Then v/q is the cofactor. We need v/q to also be p-smooth.
    #   v/q <= (p^2 + 200)/2 for q >= 2. This could still have prime factors > p.
    #
    # Better approach: for each v, factor it completely and check all factors <= p.
    # But we can be smart: since sqrt(v) < p + 1 for j < 2p+1, any prime factor of v
    # that is <= sqrt(v) is automatically <= p. We only need to worry about the
    # cofactor after dividing out all small prime factors.
    #
    # Algorithm for checking if v is p-smooth (given v = p^2 + j, j <= 200, p > 100):
    # 1. Trial divide by primes up to min(p, sqrt(v)) ~ p
    #    But we can stop early: if remaining n == 1, done.
    #    If remaining n is prime: check if n <= p.
    #    If remaining n is composite: it must have a factor > p (since all factors <= sqrt(n) 
    #    have been tried... wait, we tried up to sqrt(v) not sqrt(n)).
    #
    # Let me reconsider. After trial division by all primes up to p:
    # - If n == 1: p-smooth
    # - If n is prime: p-smooth iff n <= p
    # - If n is composite: n has no prime factor <= p, so it has a prime factor > p. NOT p-smooth.
    #
    # But trial division up to p for each of 200 values is expensive for large p.
    #
    # Key optimization: for v = p^2 + j, we can note that:
    # - If j is even and p is odd, v is odd, so 2 doesn't divide v.
    # - More generally, we can use the structure of v = p^2 + j.
    #
    # Alternative: use the fact that for the range [p^2+1, p^2+200], we can do a
    # segmented sieve with primes up to p. But p can be up to 10^9, so we can't
    # sieve with all primes up to p.
    #
    # Practical approach: for each candidate p, check p^2+1 first (cheapest filter).
    # If p^2+1 is not p-smooth, skip. Then check p^2+2, etc.
    #
    # For checking if a single value v is p-smooth:
    # 1. Trial divide by small primes (up to some limit like 10000)
    # 2. If remaining n == 1: yes
    # 3. If remaining n is prime: check n <= p
    # 4. If remaining n is composite: use Pollard's rho to factor, check all factors <= p
    #
    # This is what the original does. The key improvement is to search more primes
    # and use better filtering.
    
    # Quick check: is v p-smooth? Uses trial division + Miller-Rabin + Pollard
    def is_p_smooth(v, p):
        """Check if all prime factors of v are <= p."""
        if v <= 1:
            return True
        n = v
        # Trial division by small primes up to min(p, 10000)
        limit = min(p, 10000)
        # Generate small primes on the fly or use precomputed
        # For speed, trial divide by 2, 3, then odd numbers
        if n % 2 == 0:
            if p < 2:
                return False
            while n % 2 == 0:
                n //= 2
        if n % 3 == 0:
            if p < 3:
                return False
            while n % 3 == 0:
                n //= 3
        d = 5
        while d * d <= n and d <= limit:
            if n % d == 0:
                if p < d:
                    return False
                while n % d == 0:
                    n //= d
            d += 2
        if n == 1:
            return True
        if is_prime(n):
            return n <= p
        # n is composite, use Pollard's rho
        stack = [n]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if is_prime(m):
                if m > p:
                    return False
            else:
                d = pollard_brent(m)
                if d == m:
                    stack.append(m)
                else:
                    stack.append(d)
                    stack.append(m // d)
        return True
    
    # Precompute small primes for trial division
    small_primes_list = sieve(10000)
    
    def is_p_smooth_fast(v, p, small_primes):
        """Fast p-smooth check using precomputed small primes."""
        if v <= 1:
            return True
        n = v
        for q in small_primes:
            if q > p:
                break
            if q * q > n:
                break
            if n % q == 0:
                while n % q == 0:
                    n //= q
        if n == 1:
            return True
        if is_prime(n):
            return n <= p
        # Use Pollard's rho
        stack = [n]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if is_prime(m):
                if m > p:
                    return False
            else:
                d = pollard_brent(m)
                if d == m:
                    stack.append(m)
                else:
                    stack.append(d)
                    stack.append(m // d)
        return True
    
    best_k = 0
    best_p = 2
    
    # Check small primes first
    small_search_primes = sieve(100000)
    
    for p in small_search_primes:
        if time.time() >= deadline:
            break
        # Quick filter: p^2+1 must be p-smooth
        v1 = p * p + 1
        if not is_p_smooth_fast(v1, p, small_primes_list):
            continue
        k = 1
        while k < 200 and time.time() < deadline:
            v = p * p + k + 1
            if is_p_smooth_fast(v, p, small_primes_list):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
    
    # Search larger primes using segmented sieve to generate primes
    # For larger p, the check is more expensive but we can filter quickly
    # since p^2+1 being prime is very likely for large p
    
    # For p > 100000, use a different strategy:
    # p^2+1 is rarely p-smooth for large p. Let's estimate:
    # p^2+1 has a prime factor > p with high probability.
    # Actually, p^2+1 = a*b. If a <= b, then a <= sqrt(p^2+1) < p+1.
    # So a <= p. Then b = (p^2+1)/a. For b to be p-smooth, all factors of b must be <= p.
    # b = (p^2+1)/a. Since a >= 1, b <= p^2+1.
    # If b is prime, b > p (since b = (p^2+1)/a and if a=1, b=p^2+1 > p; if a>1, b < p^2 but could still be > p).
    # 
    # The probability that p^2+1 is p-smooth decreases as p grows.
    # Let's just search and see.
    
    # Generate primes in larger ranges using segmented sieve
    def segmented_sieve(low, high):
        """Generate primes in [low, high] using segmented sieve."""
        if high < 2:
            return []
        limit = int(math.isqrt(high)) + 1
        small_primes = sieve(limit)
        size = high - low + 1
        is_prime_seg = bytearray(b'\x01') * size
        if low <= 0:
            is_prime_seg[0] = 0
        if low <= 1:
            is_prime_seg[1 - low] = 0
        for p in small_primes:
            start = max(p * p, ((low + p - 1) // p) * p)
            for j in range(start, high + 1, p):
                is_prime_seg[j - low] = 0
        return [i for i in range(low, high + 1) if is_prime_seg[i - low]]
    
    # Search in chunks for larger primes
    search_low = 100000
    chunk_size = 500000
    
    while search_low < 10000000 and time.time() < deadline:
        search_high = min(search_low + chunk_size, 10000000)
        primes_in_range = segmented_sieve(search_low, search_high)
        
        for p in primes_in_range:
            if time.time() >= deadline:
                break
            # Quick filter: p^2+1 must be p-smooth
            v1 = p * p + 1
            # For large p, p^2+1 is very likely to have a prime factor > p
            # Quick check: if p^2+1 is prime, it's not p-smooth
            if is_prime(v1):
                continue
            if not is_p_smooth_fast(v1, p, small_primes_list):
                continue
            k = 1
            while k < 200 and time.time() < deadline:
                v = p * p + k + 1
                if is_p_smooth_fast(v, p, small_primes_list):
                    k += 1
                else:
                    break
            if k > best_k:
                best_k = k
                best_p = p
        
        search_low = search_high + 1
        # Increase chunk size as we go
        chunk_size = min(chunk_size * 2, 5000000)
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
