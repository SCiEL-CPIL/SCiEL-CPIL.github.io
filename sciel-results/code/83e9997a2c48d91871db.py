import math
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 110  # 10s margin
    
    # Deterministic Miller-Rabin bases for n < 2^64
    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0:
            return False
        if n % 3 == 0:
            return n == 3
        
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
            
        for a in MR_BASES:
            if a % n == 0:
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

    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
            
        while True:
            y = random.randint(1, n - 1)
            c = random.randint(1, n - 1)
            m = 128
            g = 1
            r = 1
            q = 1
            x = 0
            ys = []
            
            while g == 1:
                x = y
                for _ in range(r):
                    y = (y * y + c) % n
                k = 0
                while k < r and g == 1:
                    ys.append(y)
                    for _ in range(min(m, r - k)):
                        y = (y * y + c) % n
                        q = (q * abs(x - y)) % n
                    g = math.gcd(q, n)
                    k += m
                r *= 2
            
            if g == n:
                while True:
                    ys.pop()
                    g = math.gcd(abs(x - ys[-1]), n)
                    if g > 1:
                        break
            
            if g != n:
                return g

    def has_large_prime_factor(n, limit):
        """
        Returns True if n has a prime factor > limit.
        Returns False if all prime factors <= limit.
        """
        if n == 1:
            return False
        if n <= limit:
            # If n is prime, it's <= limit. If composite, factors are <= n <= limit.
            # Wait, if n is composite, its factors are smaller.
            # So if n <= limit, all factors are <= limit.
            return False
        
        # Check if n is prime
        if is_prime(n):
            return n > limit
        
        # Find a factor
        d = pollard_brent(n)
        if d == n:
            # Retry or fallback (should not happen often with Brent)
            return has_large_prime_factor(n, limit)
        
        # Check factors d and n//d
        if has_large_prime_factor(d, limit) or has_large_prime_factor(n // d, limit):
            return True
        
        return False

    def check_run(p):
        """
        Returns the length of the run of p-smooth numbers starting from p^2+1.
        Max 200.
        """
        k = 0
        p_sq = p * p
        for offset in range(1, 201):
            n = p_sq + offset
            # Quick check: if n is prime, it has a factor > p (itself)
            # Note: n = p^2 + i. Since i < p (usually, for small p), p does not divide n.
            # If n is prime, n > p (since n > p^2 >= p for p>=2).
            if is_prime(n):
                break
            
            # Check if all factors <= p
            if has_large_prime_factor(n, p):
                break
            
            k += 1
        return k

    # Generate primes up to 10^6 for sequential search
    limit = 1000000
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    
    primes = [i for i, is_p in enumerate(sieve) if is_p]
    
    best_k = 0
    best_p = 2
    
    # Phase 1: Sequential scan of small primes
    for p in primes:
        if time.time() - start_time > DEADLINE - 30:
            break
            
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Random sampling of larger primes
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 10:
        # Simple random prime generation
        # We can pick random odd numbers and check primality
        attempts = 0
        while time.time() - start_time < DEADLINE:
            # Pick random number in range
            # Use a range that is not too large to avoid bias, but large enough
            # Let's try 10^6 to 10^8
            candidate = random.randint(10**6, 10**8)
            if candidate % 2 == 0:
                candidate += 1
            # Check primality
            if is_prime(candidate):
                k = check_run(candidate)
                if k > best_k:
                    best_k = k
                    best_p = candidate
            attempts += 1
            if attempts > 10000: # Safety break
                break

    # Ensure valid result
    if best_k < 1:
        best_p = 7
        best_k = 1
        
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
