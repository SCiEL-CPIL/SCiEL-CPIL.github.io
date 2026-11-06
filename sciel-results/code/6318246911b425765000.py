import time
import random
import math
from functools import lru_cache

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin
    
    # Miller-Rabin primality test
    def is_prime(n):
        if n < 2:
            return False
        # Small primes check
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        # Write n-1 as d*2^s
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        # Witnesses sufficient for n < 3,317,044,064,679,887,385,961,981
        # We only need up to 10^9, so [2, 3, 5, 7, 11, 13] is enough
        for a in [2, 3, 5, 7, 11, 13]:
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

    # Pollard's Rho for finding a factor
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            x = random.randint(2, n - 1)
            y = x
            c = random.randint(1, n - 1)
            d = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d

    # Check if n is p-smooth (all prime factors <= p)
    def is_p_smooth(n, p):
        if n == 1:
            return True
        # If n is prime, check if it's <= p
        if is_prime(n):
            return n <= p
        # If n has a factor > p, it's not smooth.
        # We can try to divide by small primes first to reduce n
        # But Pollard Rho is generally fast.
        # Optimization: if n < p*p and not prime, its factors are < p? No.
        # Just use Pollard Rho to split.
        try:
            factor = pollard_rho(n)
        except Exception:
            # Fallback to trial division if Pollard fails (rare)
            factor = n
            for i in range(2, int(math.isqrt(n)) + 1):
                if n % i == 0:
                    factor = i
                    break
            if factor == n:
                return n <= p
        return is_p_smooth(factor, p) and is_p_smooth(n // factor, p)

    # Segmented Sieve to generate primes
    def get_primes_up_to(limit):
        if limit < 2:
            return []
        size = int(math.isqrt(limit)) + 1
        sieve = bytearray(b'\x01') * (size + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, size + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        
        primes = []
        # We can just generate primes up to limit using a simple sieve if limit is small enough
        # But for 10^6, simple sieve is fine.
        # Let's use a standard sieve for simplicity and speed for N < 10^7
        if limit > 10**7:
             # Segmented sieve logic here if needed, but 10^6 is fast
             pass
             
        # Simple sieve for limit <= 10^7
        sieve_simple = bytearray(b'\x01') * (limit + 1)
        sieve_simple[0] = sieve_simple[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if sieve_simple[i]:
                sieve_simple[i*i::i] = bytearray(len(sieve_simple[i*i::i]))
        
        return [i for i, is_p in enumerate(sieve_simple) if is_p]

    # We will search primes in increasing order.
    # Start with a reasonable limit. We can increase it if time permits.
    # Let's start with primes up to 100,000.
    # Then 1,000,000.
    
    best_k = 0
    best_p = 0
    
    # Helper to check a specific prime p
    def check_prime(p):
        nonlocal best_k, best_p
        k = 0
        base = p * p
        while k < 200:
            n = base + k + 1
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
        return k

    # Generate primes in chunks to allow early termination
    # Chunk size 10,000 or so
    chunk_size = 10000
    current_limit = 1000
    
    while time.time() < DEADLINE:
        # Generate primes in [current_limit, current_limit + chunk_size]
        # Using simple sieve for the whole range up to current_limit + chunk_size is inefficient if we do it repeatedly.
        # Better: generate all primes up to a large limit once, then iterate.
        # But we don't know how large.
        # Let's generate primes up to 1,000,000 first.
        if current_limit < 1000000:
            primes = get_primes_up_to(1000000)
            # Filter primes > current_limit
            for p in primes:
                if p <= current_limit:
                    continue
                if time.time() > DEADLINE:
                    break
                check_prime(p)
            current_limit = 1000000
        else:
            # If we've checked up to 1M, maybe go higher?
            # The probability of finding k=200 is very low for large p.
            # Usually small p gives better k? Or is it random?
            # Actually, the problem asks to maximize k.
            # Known results: for p=2, k is small.
            # Let's just stick to 1M for now, as it's a good balance.
            break

    if best_p == 0:
        # Fallback
        return {'p': 2, 'k': 1}
        
    return {'p': best_p, 'k': best_k}
