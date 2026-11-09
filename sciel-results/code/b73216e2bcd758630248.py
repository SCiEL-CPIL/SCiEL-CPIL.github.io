"""Program for: Long runs after p^2 with no prime factor exceeding p (p <= 10^9).
Optimized search strategy with efficient trial division and short-circuiting smoothness checks."""

import time
import math
import random

# ---------- Deterministic Miller-Rabin for n < 3.3e24 ----------
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
        if a % n == 0:
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

# ---------- Pollard's Brent ----------
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
                if g != 1:
                    break
        if g != n:
            return g

# ---------- Precompute small primes for trial division ----------
_SMALL_PRIMES = []
def _init_small_primes(limit=100000):
    """Generate small primes for fast trial division."""
    if limit < 2:
        return
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    _SMALL_PRIMES.extend(i for i in range(2, limit + 1) if is_p[i])

_init_small_primes(100000)

def is_p_smooth(n, p):
    """Check if all prime factors of n are <= p. Returns True if n is 1 or p-smooth."""
    if n <= 1:
        return True
    
    # Try small primes first with trial division
    # Limit trial division to min(p, max_small_prime)
    max_trial = min(p, _SMALL_PRIMES[-1])
    
    for q in _SMALL_PRIMES:
        if q > max_trial:
            break
        if n % q == 0:
            while n % q == 0:
                n //= q
            if n == 1:
                return True
        
    if n == 1:
        return True
    
    # If n is prime, check if it's <= p
    if is_prime(n):
        return n <= p
    
    # Use Pollard-Brent to find a factor
    # We need to verify all factors are <= p
    return _all_factors_leq(n, p)

def _all_factors_leq(n, p):
    """Recursively check if all prime factors of n are <= p."""
    if n == 1:
        return True
    if is_prime(n):
        return n <= p
    
    d = pollard_brent(n)
    # Check d
    if is_prime(d):
        if d > p:
            return False
        # d is valid, check the rest
        return _all_factors_leq(n // d, p)
    else:
        # d is composite, check both parts
        if not _all_factors_leq(d, p):
            return False
        return _all_factors_leq(n // d, p)

def sieve_primes(limit):
    """Generate all primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return [i for i in range(2, limit + 1) if is_p[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill
    
    best_k = 0
    best_p = 2
    
    # Generate primes up to a reasonable limit
    # 5 million is a good balance.
    sieve_limit = 5_000_000
    primes = sieve_primes(sieve_limit)
    
    # Iterate through primes
    for p in primes:
        if time.time() - start_time > deadline:
            break
            
        k = 0
        p_sq = p * p
        while k < 200:
            if time.time() - start_time > deadline:
                break
            n = p_sq + k + 1
            if not is_p_smooth(n, p):
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
            
    # If time remains, try larger primes with random sampling
    if time.time() - start_time < deadline - 10:
        lo = sieve_limit + 1
        hi = 10**8
        for _ in range(1000):
            if time.time() - start_time > deadline - 2:
                break
            candidate = random.randrange(lo, hi + 1)
            if candidate % 2 == 0:
                candidate += 1
            while not is_prime(candidate):
                candidate += 2
                if candidate > hi:
                    break
            if candidate > hi:
                continue
                
            k = 0
            p_sq = candidate * candidate
            while k < 200:
                if time.time() - start_time > deadline - 1:
                    break
                n = p_sq + k + 1
                if not is_p_smooth(n, candidate):
                    break
                k += 1
            if k > best_k:
                best_k = k
                best_p = candidate

    return {'p': best_p, 'k': best_k}
