"""Rewritten program: search for primes p with long runs of p-smooth numbers after p^2."""
# EVOLVE-BLOCK-START
import time
import math
import random
from functools import reduce
from math import gcd

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    start_time = time.time()
    
    # --- Number theoretic helpers ---
    
    # Deterministic Miller-Rabin for 64-bit integers
    def is_prime_mr(n):
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False
        # Write n-1 as 2^r * d
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        # Test bases sufficient for n < 2^64
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
    
    # Pollard's rho with Brent's improvement
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(2, n - 1)
            c = random.randint(1, n - 1)
            m = random.randint(1, 1000)
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
    
    # Full factorization
    def factorize(n):
        if n <= 1:
            return []
        if n < 4:
            return [n]
        factors = []
        # Trial division by small primes first
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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
            m = stack.pop()
            if m == 1:
                continue
            if is_prime_mr(m):
                factors.append(m)
                continue
            d = pollard_brent(m)
            stack.append(d)
            stack.append(m // d)
        return factors
    
    def largest_prime_factor(n):
        if n <= 1:
            return 1
        fs = factorize(n)
        return max(fs) if fs else 1
    
    # --- Sieve for primes ---
    
    def sieve_primes(limit):
        if limit < 2:
            return []
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i in range(limit + 1) if sieve[i]]
    
    # Precompute primes up to a reasonable limit for trial division in factorization
    SMALL_PRIME_LIMIT = 100000
    small_primes = sieve_primes(SMALL_PRIME_LIMIT)
    
    # Enhanced factorization using precomputed small primes
    def factorize_enhanced(n):
        if n <= 1:
            return []
        if n < 4:
            return [n]
        factors = []
        for p in small_primes:
            if p * p > n:
                break
            while n % p == 0:
                factors.append(p)
                n //= p
        if n == 1:
            return factors
        if is_prime_mr(n):
            factors.append(n)
            return factors
        # Use Pollard-Brent for remaining composite
        stack = [n]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if is_prime_mr(m):
                factors.append(m)
                continue
            d = pollard_brent(m)
            if d == m:
                factors.append(m)
            else:
                stack.append(d)
                stack.append(m // d)
        return factors
    
    def largest_prime_factor_enhanced(n):
        if n <= 1:
            return 1
        fs = factorize_enhanced(n)
        return max(fs) if fs else 1
    
    # --- Quick filter: check if all prime factors of n are <= p ---
    
    def all_factors_leq(n, p):
        """Return True if all prime factors of n are <= p."""
        if n <= 1:
            return True
        # Quick check: if n > p^2 and n is prime, then it has a factor > p
        # Actually, we need to be more careful
        # First try trial division by small primes
        temp = n
        for q in small_primes:
            if q > p:
                break
            if q * q > temp:
                break
            while temp % q == 0:
                temp //= q
        if temp == 1:
            return True
        # Check if remaining temp is prime and <= p
        if temp <= p:
            # Need to verify temp is prime (or 1)
            if temp == 1:
                return True
            if is_prime_mr(temp):
                return True
            # temp is composite but <= p, need to factor it
            # But if temp <= p and all its factors are <= p, we're good
            # Let's just factorize it
            fs = factorize_enhanced(temp)
            return all(f <= p for f in fs)
        # temp > p
        # If temp is prime, it's a factor > p
        if is_prime_mr(temp):
            return False
        # temp is composite and > p
        # It could still have all factors <= p (e.g., p=5, temp=25)
        # But temp > p, so if all factors are <= p, temp must be composite
        # Factorize and check
        fs = factorize_enhanced(temp)
        return all(f <= p for f in fs)
    
    # --- Main search ---
    
    best_k = 0
    best_p = 2
    
    # Strategy: search primes in a smart order
    # Small primes tend to have shorter runs, but let's check a range
    # We'll search primes up to some limit, and for each, compute the run length
    
    # First, let's try a batch of primes
    # Generate primes up to a limit
    PRIME_SEARCH_LIMIT = 200000  # Search primes up to this
    primes_list = sieve_primes(PRIME_SEARCH_LIMIT)
    
    # Also add some larger primes for variety
    # But first, focus on the sieve primes
    
    for p in primes_list:
        if time.time() > deadline - 10:
            break
        if p > 10**9:
            break
        
        # Compute run length
        k = 0
        base = p * p
        while k < 200:
            n = base + k + 1
            if all_factors_leq(n, p):
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # If we have time, try some larger primes
    if time.time() < deadline - 30:
        # Try primes in the range [100000, 1000000]
        # Use a simple primality test for larger numbers
        for p in range(100001, 1000000, 2):
            if time.time() > deadline - 15:
                break
            if not is_prime_mr(p):
                continue
            k = 0
            base = p * p
            while k < 200:
                n = base + k + 1
                if all_factors_leq(n, p):
                    k += 1
                else:
                    break
            if k > best_k:
                best_k = k
                best_p = p
    
    # Ensure k is at least 1
    if best_k < 1:
        best_k = 1
        best_p = 2
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
