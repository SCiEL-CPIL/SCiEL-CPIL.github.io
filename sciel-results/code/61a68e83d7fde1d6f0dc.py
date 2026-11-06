import math
import random
import time

# Deterministic Miller-Rabin for n < 3.3e24
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime_mr(n):
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_BASES:
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
    if n % 5 == 0:
        return 5
    while True:
        y = random.randrange(2, n - 1)
        c = random.randrange(1, n - 1)
        m = 128
        g = r = q = 1
        x = ys = 0
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

def sieve(limit):
    if limit < 2:
        return []
    bs = bytearray([1]) * (limit + 1)
    bs[0] = bs[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if bs[i]:
            step = i
            start = i * i
            bs[start::step] = bytearray(len(bs[start::step]))
    return [i for i in range(2, limit + 1) if bs[i]]

# Precompute primes up to 10^6 for trial division
_SMALL_PRIMES = sieve(1000000)
_SMALL_PRIMES_SET = set(_SMALL_PRIMES)

def has_large_prime_factor(n, p):
    """
    Check if n has a prime factor > p.
    Returns True if n has a prime factor > p, False otherwise.
    """
    if n <= 1:
        return False
    if n <= p:
        return False
    
    # Try dividing by small primes
    temp = n
    for q in _SMALL_PRIMES:
        if q > p:
            break
        if q * q > temp:
            break
        if temp % q == 0:
            while temp % q == 0:
                temp //= q
            if temp == 1:
                return False
    
    if temp == 1:
        return False
    if temp <= p:
        return False
    
    # temp > p and temp is not divisible by any prime <= min(p, sqrt(temp))
    # Check if temp is prime
    if is_prime_mr(temp):
        return temp > p
    
    # temp is composite and > p, so it has a prime factor > p
    # (because if all prime factors were <= p, then temp would be p-smooth,
    #  but we've divided out all primes <= min(p, sqrt(temp)), and temp > p)
    # Actually, we need to be careful. If temp is composite and all its prime
    # factors are <= p, then temp is p-smooth. But we've divided out all primes
    # <= min(p, sqrt(temp)). If temp > p, then sqrt(temp) could be > p or < p.
    # If sqrt(temp) <= p, then we've divided out all primes <= sqrt(temp), so
    # temp must be prime (contradiction) or 1. Since temp > 1 and not prime,
    # this case shouldn't happen.
    # If sqrt(temp) > p, then we've only divided out primes <= p. So temp could
    # be a product of primes > p, or a prime > p, or a product of some primes
    # <= p (already divided out) and some > p.
    # Since we've divided out all primes <= p, temp has no prime factor <= p.
    # So if temp is composite, it has a prime factor > p.
    return True

def find_run_length(p, k_max=200):
    """
    Find the maximum k such that p^2+1, ..., p^2+k are all p-smooth.
    """
    base = p * p
    k = 0
    for j in range(1, k_max + 1):
        n = base + j
        if has_large_prime_factor(n, p):
            break
        k += 1
    return k

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin
    random.seed(42)
    
    best_k = 0
    best_p = 2
    
    # Phase 1: Scan small primes (fast, covers many candidates)
    small_primes = sieve(100000)
    for p in small_primes:
        if time.time() >= deadline - 2.0:
            break
        k = find_run_length(p)
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == 200:
                return {'p': best_p, 'k': best_k}
    
    # Phase 2: Search for larger primes
    # For large p, the probability of p-smoothness is high.
    # We sample random numbers and check if they are prime.
    
    # Ranges to sample, from smaller to larger
    ranges = [
        (100000, 1000000),
        (1000000, 10000000),
        (10000000, 100000000),
        (100000000, 1000000000),
    ]
    
    for low, high in ranges:
        if time.time() >= deadline - 5.0:
            break
        
        while time.time() < deadline - 3.0:
            # Pick a random odd number
            n = random.randrange(low, high)
            if n % 2 == 0:
                n += 1
            if n > 10**9:
                n = 10**9 - 1 - (n % 10)
            
            # Check if prime
            if not is_prime_mr(n):
                continue
            
            p = n
            k = find_run_length(p)
            if k > best_k:
                best_k = k
                best_p = p
                if best_k == 200:
                    return {'p': best_p, 'k': best_k}
    
    return {'p': best_p, 'k': best_k}
