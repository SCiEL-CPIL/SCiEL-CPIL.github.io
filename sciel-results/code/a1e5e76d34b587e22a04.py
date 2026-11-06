# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    deadline = time.time() + 115.0  # safety margin before 150s limit
    
    # Sieve of Eratosthenes for primes up to limit
    def sieve(limit):
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                step = i
                start = i * i
                is_p[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))
        return [i for i in range(2, limit+1) if is_p[i]]
    
    # Sieve primes up to 10^6 for trial division
    SMALL_PRIMES = sieve(1000000)
    
    # Deterministic Miller-Rabin for 64-bit integers
    def is_prime(n):
        if n < 2:
            return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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
    
    # Factorize n, returning list of prime factors (with multiplicity)
    def factorize(n):
        factors = []
        for p in SMALL_PRIMES:
            if p * p > n:
                break
            while n % p == 0:
                factors.append(p)
                n //= p
        if n == 1:
            return factors
        if is_prime(n):
            factors.append(n)
            return factors
        stack = [n]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if is_prime(m):
                factors.append(m)
            else:
                d = pollard_brent(m)
                if d == m:
                    stack.append(m)
                else:
                    stack.append(d)
                    stack.append(m // d)
        return factors
    
    # Check if all prime factors of v are <= limit
    def all_factors_leq(v, limit):
        if v <= 1:
            return True
        # Quick check with small primes
        for p in SMALL_PRIMES:
            if p > limit:
                break
            if p * p > v:
                break
            while v % p == 0:
                v //= p
        if v == 1:
            return True
        if is_prime(v):
            return v <= limit
        # Factorize remaining
        factors = factorize(v)
        return all(f <= limit for f in factors)
    
    # Segmented sieve to check p-smoothness for range [p^2+1, p^2+k]
    # Returns the maximum k such that all values p^2+1 ... p^2+k are p-smooth
    def max_smooth_run(p, max_k=200):
        """
        Find the largest k <= max_k such that p^2+1, ..., p^2+k all have
        prime factors <= p.
        Uses segmented sieve: for each prime q <= p, mark multiples in the range.
        A number is p-smooth iff after removing all factors q <= p, remainder is 1.
        """
        start = p * p + 1
        end = p * p + max_k
        n_vals = max_k
        
        # remaining[i] holds the part of (start+i) not yet divided out
        remaining = [start + i for i in range(n_vals)]
        
        # For each prime q <= p, divide out q from all positions where it divides
        # We only need primes q <= p AND q <= sqrt(end) for the "remaining becomes prime" check
        # Actually we need all primes q <= p to fully factorize
        # But if q > sqrt(end), then q can appear at most once in any number in range
        # and if q divides start+i, then start+i = q * m where m < q, so m must be p-smooth too
        # Strategy: trial divide by all primes q <= p. After that, remaining[i] should be 1
        # or a prime <= p.
        
        for q in SMALL_PRIMES:
            if q > p:
                break
            # Find first multiple of q in [start, end]
            first = ((start + q - 1) // q) * q
            if first > end:
                continue
            # Mark all positions where q divides
            for pos in range(first - start, n_vals, q):
                while remaining[pos] % q == 0:
                    remaining[pos] //= q
        
        # Now check each remaining value
        # It should be 1 or a prime <= p
        k = 0
        for i in range(n_vals):
            rem = remaining[i]
            if rem == 1:
                k += 1
            elif is_prime(rem) and rem <= p:
                k += 1
            else:
                # rem has a prime factor > p
                break
        
        return k
    
    best_k = 0
    best_p = 2
    
    # Check p = 2 first
    k = max_smooth_run(2, 200)
    if k > best_k:
        best_k = k
        best_p = 2
    
    # Search odd primes p using segmented sieve approach
    p = 3
    while time.time() < deadline and p < 10**9:
        if is_prime(p):
            k = max_smooth_run(p, 200)
            if k > best_k:
                best_k = k
                best_p = p
        p += 2
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
