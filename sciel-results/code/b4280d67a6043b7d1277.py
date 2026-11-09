import time
import math
import random

def construct():
    deadline = time.time() + 140
    
    # Small primes sieve
    LIMIT = 10**6
    sieve = bytearray(b'\x01') * (LIMIT + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(LIMIT**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    small_primes = [i for i in range(2, LIMIT + 1) if sieve[i]]
    
    # Miller-Rabin
    def is_prime(n):
        if n < 2:
            return False
        # Small primes check
        small_primes_mr = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes_mr:
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
        # Bases for deterministic test for n < 2^64
        bases = [2, 325, 9375, 28178, 450775, 9780504, 1795265022]
        for a in bases:
            if a % n == 0:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    # Pollard-Brent
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(1, n - 1)
            c = random.randint(1, n - 1)
            m = random.randint(1, n - 1)
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
                        q = (q * abs(x - y)) % n
                    g = math.gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
            if g != n:
                return g

    # Check if n is p-smooth
    def is_p_smooth(n, p):
        if n == 1:
            return True
        # Trial division by small primes
        for q in small_primes:
            if q > p:
                break
            while n % q == 0:
                n //= q
                if n == 1:
                    return True
        if n == 1:
            return True
        
        # If we broke out because q > p, and n > 1, 
        # then n has no prime factors <= p (from the small primes set).
        # But wait, if p is large (e.g. 10^9), we only checked up to 10^6.
        # So n might have a prime factor between 10^6 and p.
        
        if is_prime(n):
            return n <= p
        else:
            # n is composite
            d = pollard_brent(n)
            return is_p_smooth(d, p) and is_p_smooth(n // d, p)

    best_k = 0
    best_p = 2
    
    # Iterate through potential primes p
    # We can iterate odd numbers and check primality
    # Or generate primes. MR is fast enough for scanning.
    # Let's scan from 3 upwards.
    p = 3
    while time.time() < deadline:
        if is_prime(p):
            k = 0
            # Check p^2 + 1, ..., p^2 + 200
            # Early exit if k reaches 200
            for i in range(1, 201):
                if time.time() > deadline - 5:
                    break
                n_val = p * p + i
                if is_p_smooth(n_val, p):
                    k = i
                else:
                    break
            if k > best_k:
                best_k = k
                best_p = p
                # Optional: if best_k is very high, we can stop or continue
                # The task says maximize k.
        p += 2
        
    return {'p': best_p, 'k': best_k}
