import math
import random

def miller_rabin(n):
    if n < 2:
        return False
    # Small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    
    # Write n-1 as d * 2^r
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
        
    # Deterministic for n < 3,317,044,064,679,887,385,961,981
    # We only need up to 10^12, so bases [2, 3, 5, 7, 11, 13] are sufficient.
    for a in [2, 3, 5, 7, 11, 13]:
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
    if n % 2 == 0:
        return 2
    while True:
        c = random.randint(1, n - 1)
        x = random.randint(2, n - 1)
        y = x
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d

def factorize(n):
    factors = []
    def _factorize(n):
        if n == 1:
            return
        if miller_rabin(n):
            factors.append(n)
            return
        d = pollard_rho(n)
        _factorize(d)
        _factorize(n // d)
    _factorize(n)
    return factors

def phi(x):
    if x == 1:
        return 1
    factors = factorize(x)
    unique_factors = set(factors)
    result = x
    for p in unique_factors:
        result -= result // p
    return result

def construct():
    # We want to find s, k such that phi(s)...phi(s+k-1) are distinct.
    # Max k is 200.
    # We search in a high range to minimize collision probability.
    
    base_s = 10**11
    max_k = 200
    
    # Try a few starting points
    for offset in range(0, 50):
        s = base_s + offset * 10000
        if s + max_k - 1 > 10**12:
            break
            
        distinct = True
        seen = set()
        
        for i in range(max_k):
            n = s + i
            v = phi(n)
            if v in seen:
                distinct = False
                break
            seen.add(v)
            
        if distinct:
            return {'start': s, 'length': max_k}
            
    # Fallback
    return {'start': 1, 'length': 1}
