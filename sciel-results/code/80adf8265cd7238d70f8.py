"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random

def is_prime_miller_rabin(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    # Deterministic for n < 3,317,044,064,679,887,385,961,981
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # Bases sufficient for n < 2^64
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
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
        y = random.randrange(1, n)
        c = random.randrange(1, n)
        m = 128
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

def factorize(n, primes=None):
    """Return list of prime factors of n (with multiplicity)."""
    if n <= 1:
        return []
    if primes is None:
        primes = []
    factors = []
    for p in primes:
        if p * p > n:
            break
        while n % p == 0:
            factors.append(p)
            n //= p
    if n == 1:
        return factors
    if is_prime_miller_rabin(n):
        factors.append(n)
    else:
        d = pollard_brent(n)
        factors.extend(factorize(d, primes))
        factors.extend(factorize(n // d, primes))
    return factors

def largest_prime_factor(n, primes):
    """Return the largest prime factor of n."""
    if n <= 1:
        return 1
    max_pf = 1
    for p in primes:
        if p * p > n:
            break
        while n % p == 0:
            max_pf = p
            n //= p
    if n > 1:
        if is_prime_miller_rabin(n):
            max_pf = max(max_pf, n)
        else:
            factors = factorize(n, primes)
            max_pf = max(max_pf, max(factors))
    return max_pf

def construct():
    # Generate primes up to 10000 for trial division
    limit = 10000
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    small_primes = [i for i in range(2, limit + 1) if sieve[i]]
    
    best_k = 0
    best_p = 2
    
    # Search over primes p
    # First, search small primes thoroughly
    for p in small_primes:
        if p > 50000:
            break
        k = 0
        for j in range(1, 201):
            n = p * p + j
            lp = largest_prime_factor(n, small_primes)
            if lp <= p:
                k = j
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
        if k >= 200:
            break
    
    # Try some larger primes heuristically
    # Check primes around specific values
    test_primes = []
    for p in range(50001, 200000, 2):
        if is_prime_miller_rabin(p):
            test_primes.append(p)
    
    for p in test_primes:
        # Quick check: is p^2+1 p-smooth?
        n = p * p + 1
        lp = largest_prime_factor(n, small_primes)
        if lp > p:
            continue
        k = 1
        for j in range(2, 201):
            n = p * p + j
            lp = largest_prime_factor(n, small_primes)
            if lp <= p:
                k = j
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
        if k >= 200:
            break
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
