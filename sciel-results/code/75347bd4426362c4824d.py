"""Improved program using Pollard-Brent for efficient p-smooth checking."""
import math
import time
import random

# EVOLVE-BLOCK-START

def is_prime_mr(n):
    """Deterministic Miller-Rabin for n < 3.2e18."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
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

def pollard_rho(n):
    """Find a non-trivial factor of n using Pollard's Rho."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    
    # Use random constants
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

def sieve_primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def is_p_smooth(n, p, small_primes):
    """Check if n is p-smooth."""
    if n <= 1:
        return True
    if n <= p:
        return True
    
    # Trial division by small primes
    for q in small_primes:
        if q > p:
            break
        if q * q > n:
            break
        if n % q == 0:
            while n % q == 0:
                n //= q
            if n <= 1:
                return True
            if n <= p:
                return True
            # Recurse on remainder
            return is_p_smooth(n, p, small_primes)
            
    # n is not divisible by any small prime <= p
    # Check if n is prime
    if is_prime_mr(n):
        # n is prime. Is n <= p?
        # We already checked n <= p at start.
        # So n > p.
        return False
        
    # n is composite. Find a factor.
    d = pollard_rho(n)
    return is_p_smooth(d, p, small_primes) and is_p_smooth(n // d, p, small_primes)

def construct():
    start_time = time.time()
    time_limit = 110
    
    # Precompute small primes for trial division
    # 10^5 is a good cutoff for trial division before Pollard
    small_primes = sieve_primes(100000)
    
    # Precompute primes to iterate over
    # We can generate primes up to 10^7
    max_p_search = 10000000
    all_primes = sieve_primes(max_p_search)
    
    best_k = 0
    best_p = 2
    
    def check_p(p):
        """Check k for given prime p."""
        k = 0
        for i in range(1, 201):
            n = p * p + i
            if not is_p_smooth(n, p, small_primes):
                break
            k += 1
        return k

    # Iterate through primes
    # We can skip very small primes if we want, but they are fast to check
    for p in all_primes:
        if time.time() - start_time > time_limit:
            break
        
        # Optimization: if p is very large, p^2+i is large.
        # But Pollard Rho is fast.
        
        k = check_p(p)
        if k > best_k:
            best_k = k
            best_p = p
    
    return {'p': best_p, 'k': best_k}

# EVOLVE-BLOCK-END
