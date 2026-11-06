import time
import math
import random
import sys

# Increase recursion limit for factorization
sys.setrecursionlimit(10000)

def is_prime(n):
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    # Deterministic Miller-Rabin for n < 3,317,044,064,679,887,385,961,981
    # Bases: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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

def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randrange(2, n)
        c = random.randrange(1, n)
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
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g > 1:
                    break
        if g != n:
            return g

# Pre-compute small primes for trial division
SMALL_PRIME_LIMIT = 10**5
def sieve(limit):
    is_prime_flag = [True] * (limit + 1)
    is_prime_flag[0] = is_prime_flag[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime_flag[i]:
            for j in range(i*i, limit + 1, i):
                is_prime_flag[j] = False
    return [i for i, prime in enumerate(is_prime_flag) if prime]

SMALL_PRIMES = sieve(SMALL_PRIME_LIMIT)

def is_p_smooth(n, p):
    """Check if n has all prime factors <= p."""
    if n <= p:
        return True
    
    # Trial division by small primes
    for q in SMALL_PRIMES:
        if q > p:
            break
        if q * q > n:
            break
        while n % q == 0:
            n //= q
        if n == 1:
            return True
        if n <= p:
            return True
            
    # If n is prime, it must be <= p
    if is_prime(n):
        return n <= p
        
    # Otherwise, find a factor using Pollard's Rho
    # n is composite and > p
    d = pollard_brent(n)
    if d == n:
        # Should not happen if n is composite, but just in case
        return is_prime(n) and n <= p
        
    return is_p_smooth(d, p) and is_p_smooth(n // d, p)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Sieve primes up to 10^7
    # This takes ~0.5s and uses ~10MB
    limit = 10**7
    primes = sieve(limit)
    
    best_k = 0
    best_p = 2
    
    for p in primes:
        if time.time() > deadline:
            break
            
        # Quick check: if p^2 + 1 is prime, k=0, skip
        n1 = p * p + 1
        if is_prime(n1):
            continue
            
        # Check run length
        k = 0
        while k < 200:
            n = p * p + k + 1
            if not is_p_smooth(n, p):
                break
            k += 1
            
        if k > best_k:
            best_k = k
            best_p = p
            
    return {'p': best_p, 'k': best_k}
