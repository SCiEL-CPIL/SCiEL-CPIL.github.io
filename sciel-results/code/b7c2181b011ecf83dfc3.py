import time
import random
import math

# ── Deterministic Miller-Rabin for n < 2^64 ──────────────────────────────────
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
                 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
                 101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
                 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]

def _gen_primes_upto(limit):
    comp = bytearray(limit + 1)
    for i in range(2, int(math.isqrt(limit)) + 1):
        if not comp[i]:
            comp[i*i::i] = b'\x01' * ((limit - i*i)//i + 1)
    return [i for i in range(2, limit+1) if not comp[i]]

# Precompute primes up to 2,000,000
# This allows us to test p up to 2,000,000 efficiently
_PRIME_LIMIT = 2_000_000
_PRIMES = _gen_primes_upto(_PRIME_LIMIT)

def is_prime(n):
    """Deterministic Miller-Rabin.  Exact for n < 2^64 with 12 base primes."""
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
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
        if a >= n:
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

def _pollard_brent(n):
    """Return a non-trivial factor of composite n."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    c = random.randrange(1, n - 1)
    y = random.randrange(0, n - 1)
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
                q = (q * abs(x - y)) % n
            g = math.gcd(q, n)
            k += m
        r *= 2
    if g == n:
        g = 1
        while g == 1:
            ys = (ys * ys + c) % n
            g = math.gcd(abs(x - ys), n)
    if g == n:
        return _pollard_brent(n)
    return g

def _factor(n, factors):
    """Recursively factor n, appending prime factors to *factors*."""
    if n == 1:
        return
    for p in _SMALL_PRIMES:
        if n % p == 0:
            factors.append(p)
            _factor(n // p, factors)
            return
    if is_prime(n):
        factors.append(n)
        return
    d = _pollard_brent(n)
    _factor(d, factors)
    _factor(n // d, factors)

def all_prime_factors_leq(n, limit):
    """True iff every prime factor of n is ≤ limit.  Early-exit on violation."""
    if n <= 1:
        return True
    for p in _PRIMES:
        if p > limit:
            break
        if p * p > n:
            break
        if n % p == 0:
            while n % p == 0:
                n //= p
            if n == 1:
                return True
    if n == 1:
        return True
    if n <= limit:
        return True
    if is_prime(n):
        return False
    # If n is composite and > limit, it has a factor > limit?
    # If n has no factors <= limit, and n > limit, and n is composite,
    # then n has a factor > limit.
    return False

def construct():
    start = time.time()
    deadline = start + 110  # Safety margin

    best_k = 0
    best_p = 2

    # We will search primes p up to _PRIME_LIMIT.
    # For each p, we check k.
    # To speed up, we can randomize the order or start from larger p?
    # Larger p might have larger k? Not necessarily.
    # Let's try a mix.
    
    # First, check some small primes to get a baseline
    for p in _PRIMES[:100]:
        if time.time() > deadline:
            break
        base = p * p
        k = 0
        for i in range(1, 201):
            val = base + i
            if not all_prime_factors_leq(val, p):
                break
            k += 1
        if k > best_k:
            best_k = k
            best_p = p

    # Now random search in the prime range
    # We can't afford to check all 150k primes.
    # Let's check random samples.
    # Also, we should prioritize p where p^2+1 is smooth.
    
    # To improve efficiency, we can pre-filter:
    # p^2+1 must be p-smooth.
    # This is a strong condition.
    
    for _ in range(5000):
        if time.time() > deadline:
            break
        # Pick a random prime index
        idx = random.randint(0, len(_PRIMES) - 1)
        p = _PRIMES[idx]
        
        # Quick check for i=1
        val1 = p * p + 1
        if not all_prime_factors_leq(val1, p):
            continue
            
        k = 1
        for i in range(2, 201):
            val = p * p + i
            if not all_prime_factors_leq(val, p):
                break
            k += 1
            
        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': best_k}
