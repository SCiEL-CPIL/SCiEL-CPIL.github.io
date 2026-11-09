"""Rewritten program: fast Miller-Rabin + Pollard-Brent factoring, broader prime search."""
# EVOLVE-BLOCK-START

import time
import math
import random

def construct():
    # ---------- Deterministic Miller-Rabin for 64-bit ----------
    def is_prime(n):
        if n < 2:
            return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n % p == 0:
                return n == p
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        # Deterministic witnesses for n < 2^64
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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

    # ---------- GCD ----------
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    # ---------- Pollard-Brent ----------
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randrange(2, n)
            c = random.randrange(1, n)
            m = random.randrange(1, n)
            g = r = q = 1
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

    # ---------- Factor a number into prime factors ----------
    def factor(n, factors=None):
        if factors is None:
            factors = []
        if n == 1:
            return factors
        if is_prime(n):
            factors.append(n)
            return factors
        d = pollard_brent(n)
        factor(d, factors)
        factor(n // d, factors)
        return factors

    # ---------- Check if all prime factors of n are <= p ----------
    def is_p_smooth(n, p, small_primes_list):
        if n <= 1:
            return True
        # Trial divide by small primes up to min(p, sqrt(n))
        for sp in small_primes_list:
            if sp > p:
                break
            if sp * sp > n:
                break
            while n % sp == 0:
                n //= sp
        if n == 1:
            return True
        if n <= p:
            # n is either 1, a prime <= p, or composite with all factors <= p
            # Actually if n <= p and n > 1, n could be prime or composite
            # If n is prime, it's <= p, so ok
            # If n is composite, all its factors are < n <= p, so ok
            return True
        # n > p, need to check if n has any prime factor > p
        # Factor n and check
        try:
            facs = factor(n)
            for f in facs:
                if f > p:
                    return False
            return True
        except Exception:
            return False

    # ---------- Sieve small primes ----------
    def sieve_primes(limit):
        if limit < 2:
            return []
        is_p = [True] * (limit + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if is_p[i]:
                for j in range(i * i, limit + 1, i):
                    is_p[j] = False
        return [i for i, v in enumerate(is_p) if v]

    # ---------- Main search ----------
    deadline = time.time() + 140  # safety margin before 150s kill

    # Precompute small primes for trial division
    small_primes_list = sieve_primes(10000)

    best_k = 0
    best_p = 2

    # Strategy: search primes in increasing order, but skip ahead intelligently
    # For small p, k is small. For larger p, k can be larger.
    # We'll search primes up to some limit, checking k for each.
    
    # First, do a quick scan of small primes to get a baseline
    # Then focus on larger primes where longer runs are more likely
    
    # Generate candidate primes
    # We'll use a segmented sieve approach for efficiency
    
    def generate_primes_in_range(low, high):
        """Generate primes in [low, high] using segmented sieve."""
        if high < 2:
            return []
        low = max(low, 2)
        # Get primes up to sqrt(high)
        limit = int(math.isqrt(high)) + 1
        base_primes = sieve_primes(limit)
        
        size = high - low + 1
        is_p = [True] * size
        is_p[0] = (low == 1)  # 1 is not prime
        
        for p in base_primes:
            if p * p > high:
                break
            # Find first multiple of p >= low
            start = max(p * p, ((low + p - 1) // p) * p)
            for j in range(start - low, size, p):
                is_p[j] = False
        
        return [low + i for i in range(size) if is_p[i] and low + i >= 2]

    # Search strategy:
    # 1. Quick scan of primes up to 10000 to find initial best
    # 2. Then scan larger ranges, focusing on where longer runs might occur
    
    # Phase 1: Small primes
    small_primes = generate_primes_in_range(2, 10000)
    for p in small_primes:
        if time.time() > deadline:
            break
        p2 = p * p
        k = 0
        for i in range(1, 201):
            n = p2 + i
            if is_p_smooth(n, p, small_primes_list):
                k = i
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Larger primes in segments
    # Longer runs are more likely for larger p
    # Search in increasing segments
    seg_low = 10001
    seg_high = 100000
    while seg_low <= 10**9:
        if time.time() > deadline:
            break
        primes_in_seg = generate_primes_in_range(seg_low, seg_high)
        for p in primes_in_seg:
            if time.time() > deadline:
                break
            p2 = p * p
            k = 0
            for i in range(1, 201):
                n = p2 + i
                if is_p_smooth(n, p, small_primes_list):
                    k = i
                else:
                    break
            if k > best_k:
                best_k = k
                best_p = p
        seg_low = seg_high + 1
        seg_high = seg_high * 2 + 1  # grow the segment
    
    return {'p': best_p, 'k': best_k}

# EVOLVE-BLOCK-END
