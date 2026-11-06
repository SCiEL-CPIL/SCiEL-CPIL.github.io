import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for safety
    
    # Precompute small primes for trial division
    # Limit to 10^6 is a good balance
    def get_primes(limit):
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0:2] = b'\x00\x00'
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    small_primes = get_primes(10**6)
    
    # Deterministic Miller-Rabin for 64-bit integers
    def is_prime(n):
        if n < 2: return False
        # Small primes check
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        # Miller-Rabin
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        # Bases for n < 3,317,044,064,679,887,385,961,981
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n: continue
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

    # Pollard's Rho
    def pollard_rho(n):
        if n % 2 == 0: return 2
        if n % 3 == 0: return 3
        while True:
            c = random.randint(1, n - 1)
            x = random.randint(0, n - 1)
            y = x
            d = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d
            # Retry with different c if d == n

    def factorize(n):
        if n == 1: return []
        if is_prime(n): return [n]
        d = pollard_rho(n)
        return factorize(d) + factorize(n // d)

    def is_p_smooth(n, p):
        if n == 1: return True
        # Trial division by small primes
        for q in small_primes:
            if q > p: break
            if q * q > n: break
            while n % q == 0:
                n //= q
        if n == 1: return True
        # If n > 1, check if it is prime
        if is_prime(n):
            return n <= p
        # If composite, factorize and check
        # Optimization: if n is composite, it has a factor <= sqrt(n)
        # If sqrt(n) <= p, then the factor is <= p.
        # We just need to ensure ALL factors are <= p.
        factors = factorize(n)
        for f in factors:
            if f > p:
                return False
        return True

    best_k = 0
    best_p = 2
    
    # Helper to check if p is prime (using small primes first)
    def is_prime_fast(n):
        if n < 2: return False
        for q in small_primes:
            if q * q > n: return True
            if n % q == 0: return n == q
        return is_prime(n)

    # Search primes
    # We'll iterate through potential primes p
    # Start with small p, then larger
    # Note: p must be prime.
    
    p = 2
    # We will search up to 10^7 or until deadline
    # To speed up, we can jump.
    while p <= 10**7:
        if time.time() > deadline:
            break
            
        if is_prime_fast(p):
            # Check k
            k = 0
            # We want to maximize k.
            # Check p^2+1 ... p^2+200
            # If any fails, stop.
            for j in range(1, 201):
                n = p * p + j
                if not is_p_smooth(n, p):
                    break
                k += 1
            if k > best_k:
                best_k = k
                best_p = p
        
        p += 1 if p == 2 else 2 # Skip even numbers after 2

    # If we didn't find anything good, or if we have time, try some larger random primes?
    # For now, return best found.
    return {'p': best_p, 'k': best_k}
