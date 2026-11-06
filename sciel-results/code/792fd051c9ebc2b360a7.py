import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Deterministic Miller-Rabin for n < 2^64
    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def is_prime(n):
        if n < 2:
            return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in MR_BASES:
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
    
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        while True:
            y = random.randrange(1, n)
            c = random.randrange(1, n)
            m = 128
            g, q, r = 1, 1, 1
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
    
    # Precompute small primes for trial division
    def sieve_small_primes(limit):
        if limit < 2:
            return []
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    small_primes = sieve_small_primes(1000)
    
    def factorize(n, limit):
        """
        Returns True if all prime factors of n are <= limit.
        """
        if n <= 1:
            return True
        
        # Trial division by small primes
        for p in small_primes:
            if p > limit:
                break
            if n % p == 0:
                while n % p == 0:
                    n //= p
                if n == 1:
                    return True
                if is_prime(n):
                    return n <= limit
                break
        
        if n == 1:
            return True
        if is_prime(n):
            return n <= limit
        
        # Pollard Rho
        d = pollard_brent(n)
        return factorize(d, limit) and factorize(n // d, limit)
    
    def check_run(p, max_k=200):
        """
        Returns k such that p^2+1 ... p^2+k are p-smooth.
        Stops at max_k or when a non-p-smooth number is found.
        """
        k = 0
        base = p * p
        while k < max_k:
            n = base + k + 1
            if not factorize(n, p):
                break
            k += 1
        return k
    
    # Sieve of Eratosthenes to get primes up to a limit
    def sieve_primes(limit):
        if limit < 2:
            return []
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    best_k = 0
    best_p = 2
    
    # Phase 1: Quick scan of small primes to establish baseline
    limit1 = 100000
    primes1 = sieve_primes(limit1)
    
    for p in primes1:
        if time.time() > deadline - 60:
            break
        
        k = check_run(p, 50)  # Limit to 50 for speed in initial scan
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 2: Deeper search on promising primes (those with k >= 5 in Phase 1)
    if time.time() < deadline - 30:
        # Re-check the best candidates with full max_k
        for p in primes1:
            if time.time() > deadline - 30:
                break
            # Only re-check if it showed promise
            quick_k = check_run(p, 20)
            if quick_k >= 15:
                k = check_run(p, 200)
                if k > best_k:
                    best_k = k
                    best_p = p
    
    # Phase 3: Search larger primes using random sampling
    if time.time() < deadline - 10:
        # Try primes in different ranges
        ranges = [(10**5, 10**6), (10**6, 10**7), (10**7, 10**8), (10**8, 10**9)]
        
        for lo, hi in ranges:
            if time.time() > deadline - 10:
                break
            
            # Sample a few random primes in this range
            attempts = 50
            for _ in range(attempts):
                if time.time() > deadline - 10:
                    break
                x = random.randrange(lo, hi)
                if x % 2 == 0:
                    x += 1
                # Find next prime
                while not is_prime(x):
                    x += 2
                    if x > hi:
                        break
                if x > hi:
                    continue
                
                # Quick check first
                quick_k = check_run(x, 30)
                if quick_k >= 10:
                    k = check_run(x, 200)
                    if k > best_k:
                        best_k = k
                        best_p = x
    
    return {'p': best_p, 'k': best_k}
