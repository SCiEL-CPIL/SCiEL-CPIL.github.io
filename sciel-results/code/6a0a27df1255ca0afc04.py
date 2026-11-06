import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115 seconds deadline for safety
    
    # Sieve of Eratosthenes for small primes
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    small_primes = sieve(100000) # Primes up to 10^5 for trial division
    small_primes_set = set(small_primes)
    
    # Miller-Rabin primality test
    def is_prime_mr(n):
        if n < 2:
            return False
        # Small primes
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n == p:
                return True
            if n % p == 0:
                return False
        # Write n-1 as d*2^r
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        # Bases sufficient for n < 2^64
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    # Pollard's Rho (Brent variant)
    _brent_counter = [0]
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            _brent_counter[0] += 1
            seed = _brent_counter[0] * 123456789 + n
            rng = random.Random(seed)
            y = rng.randint(1, n - 1)
            c = rng.randint(1, n - 1)
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
                if r > 1 << 20:  # Safety: reset if too many iterations
                    break
            if g == n:
                while True:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
                    if g > 1:
                        break
            if g != n:
                return g

    def factorize(n):
        if n == 1:
            return []
        if n in small_primes_set:
            return [n]
        factors = []
        # Trial division by small primes
        for p in small_primes:
            if p * p > n:
                break
            if n % p == 0:
                while n % p == 0:
                    factors.append(p)
                    n //= p
        if n == 1:
            return factors
        if is_prime_mr(n):
            factors.append(n)
            return factors
        # Use Pollard-Brent
        stack = [n]
        while stack:
            x = stack.pop()
            if is_prime_mr(x):
                factors.append(x)
            else:
                d = pollard_brent(x)
                stack.append(d)
                stack.append(x // d)
        return factors

    def is_p_smooth(n, p_limit):
        """Check if all prime factors of n are <= p_limit."""
        if n == 1:
            return True
        # Trial divide by small primes up to min(p_limit, sqrt(n))
        temp_n = n
        for sp in small_primes:
            if sp > p_limit:
                break
            if sp * sp > temp_n:
                break
            if temp_n % sp == 0:
                while temp_n % sp == 0:
                    temp_n //= sp
        if temp_n == 1:
            return True
        # temp_n > 1, all its prime factors are > min(p_limit, 100000) or > sqrt(original after division)
        # Check if temp_n is prime
        if is_prime_mr(temp_n):
            return temp_n <= p_limit
        # temp_n is composite. Need to check if it has any prime factor > p_limit.
        # Use Pollard-Brent to find a factor
        d = pollard_brent(temp_n)
        # Recursively check both parts
        return is_p_smooth(d, p_limit) and is_p_smooth(temp_n // d, p_limit)

    def segmented_sieve_check(p, max_k=200):
        """
        Use segmented sieve over [p^2+1, p^2+max_k] to quickly determine
        the run length of consecutive numbers with all prime factors <= p.
        
        Returns the run length k (0 to max_k).
        """
        p_sq = p * p
        lo = p_sq + 1
        hi = p_sq + max_k
        
        # Create array to track largest prime factor > p for each position
        # We'll use a different approach: mark which numbers have a prime factor > p
        
        # Initialize: assume all numbers are "good" (no prime factor > p)
        # We need to check if any number in [lo, hi] has a prime factor > p
        
        # Strategy: For each prime q > p, if q divides any number in [lo, hi],
        # that number is "bad". But we can't iterate over all primes > p.
        
        # Better strategy: For each number n in [lo, hi], check if n has a prime factor > p.
        # We can do this by trial division up to sqrt(n), but that's expensive.
        
        # Alternative: Use the fact that if n has a prime factor > p, then either:
        # 1. n itself is prime and > p, or
        # 2. n = a * b where a > p and b >= 1, so n >= a > p
        #    or n = a * b where b > p and a >= 1
        
        # Actually, the most efficient approach for a small window:
        # For each number n in [lo, hi], divide out all prime factors <= p.
        # If the remaining factor is > 1, then n has a prime factor > p.
        
        # But we can be smarter: only check primes up to sqrt(max(n)) for each n,
        # and stop if we find a factor > p.
        
        # Even smarter: precompute which primes <= p we need to divide by.
        # Since p can be up to 10^9, we can't have all primes <= p.
        # But we only need primes up to sqrt(hi) for trial division.
        
        # For the window [lo, hi] where hi - lo = max_k <= 200:
        # sqrt(hi) <= sqrt(p^2 + 200) ≈ p + 100/p for large p
        # So we need primes up to about p + small amount
        
        # This is still too many primes if p is large.
        
        # Better approach: For each n in [lo, hi], do trial division by small primes
        # (up to 10^5) and if the remaining factor is > 1, check if it's prime and > p,
        # or if it's composite, use Pollard-Brent to find a factor and check.
        
        # But the key insight: most numbers will have a small prime factor > p?
        # No, that's not right.
        
        # Let me reconsider: The condition is that ALL prime factors of n are <= p.
        # So n must be p-smooth.
        
        # For a quick check: if n has any prime factor > p, it's bad.
        # We can check this by:
        # 1. Divide out all prime factors <= p (using small primes up to 10^5)
        # 2. If remaining > 1, check if it's prime and > p, or composite with a factor > p
        
        # This is essentially what is_p_smooth does, but we want to do it for all 200 numbers
        # at once efficiently.
        
        # For the segmented sieve approach:
        # We want to find, for each n in [lo, hi], whether n has a prime factor > p.
        
        # One approach: For each prime q where p < q <= sqrt(hi), mark all multiples of q in [lo, hi].
        # If any number is marked, it has a prime factor > p.
        # Also, if any number n in [lo, hi] is itself prime and > p, it's bad.
        
        # But sqrt(hi) ≈ p, so we'd need primes in (p, sqrt(hi)] which is empty for large p.
        # Wait, sqrt(hi) ≈ p, so primes > p up to sqrt(hi) don't exist if p > sqrt(hi).
        # But p < sqrt(hi) is possible for small p.
        
        # Actually, for p > sqrt(hi), any prime factor > p of n must be n itself (if n is prime)
        # or n = q * r where q > p and r < n/q < n/p < p (since q > p implies r < n/q < n/p).
        # So if p > sqrt(hi), then n has a prime factor > p iff n is prime and n > p.
        
        # For p <= sqrt(hi), we need to check primes in (p, sqrt(hi)].
        
        # This is getting complicated. Let me just use the is_p_smooth check for each number,
        # but with an early exit: stop as soon as we find a bad number.
        
        for j in range(1, max_k + 1):
            n = p_sq + j
            if not is_p_smooth(n, p):
                return j - 1
        return max_k

    best_k = 0
    best_p = 2
    
    # Generate primes for p
    # We can scan primes up to 10^7
    limit_p = 10**7
    primes_p = sieve(limit_p)
    
    for p in primes_p:
        if time.time() > deadline:
            break
        
        # Quick check using segmented sieve approach
        k = segmented_sieve_check(p)
        
        if k > best_k:
            best_k = k
            best_p = p
        
        # Early termination: if we've found k=200, we can't do better
        if best_k >= 200:
            break

    return {'p': best_p, 'k': best_k}
