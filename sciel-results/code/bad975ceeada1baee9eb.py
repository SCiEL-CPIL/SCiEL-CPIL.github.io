"""Program for: Long runs after p^2 with no prime factor exceeding p (p <= 10^9)."""

import time
import random
import math

# ---------- Deterministic Miller-Rabin for n < 3.3e24 ----------
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def is_prime(n):
    if n < 2:
        return False
    for p in _MR_BASES:
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

def pollard_brent(n):
    """Brent's variant of Pollard's rho - faster than Floyd's."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    
    while True:
        y = random.randrange(1, n)
        c = random.randrange(1, n)
        m = random.randrange(1, 100)
        g = 1
        r = 1
        q = 1
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

def factor(n, factors=None):
    if factors is None:
        factors = []
    if n == 1:
        return factors
    if is_prime(n):
        factors.append(n)
        return factors
    d = pollard_brent(n)
    factor(d, factors)
    factor(n // d, factors)
    return factors

def sieve_primes(limit):
    """Generate list of all primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return [i for i in range(2, limit + 1) if is_p[i]]

def is_p_smooth(n, p, small_primes):
    """Check if all prime factors of n are <= p."""
    if n <= 1:
        return True
    
    v = n
    sqrt_n = math.isqrt(n)
    limit = min(p, sqrt_n)
    
    # Trial divide by small primes up to limit
    for sp in small_primes:
        if sp > limit:
            break
        if v % sp == 0:
            while v % sp == 0:
                v //= sp
            if v == 1:
                return True
    
    if v == 1:
        return True
    
    # After trial division, v has no prime factors <= limit
    if v <= p:
        return True
    
    # v > p and v has no factor <= limit = min(p, sqrt_n)
    # If limit == p: v has no factor <= p, so if v is prime, v > p -> not smooth
    # If limit == sqrt_n < p: v has no factor <= sqrt_n, so v is prime or 1
    #   In that case v is prime, and v > p -> not smooth
    # Either way, v > p means not p-smooth
    return False

def construct():
    deadline = time.time() + 110  # safety margin before 120s kill
    
    best_k = 0
    best_p = 2
    
    # Precompute small primes for trial division
    # Use primes up to 200000 for fast trial division
    small_primes = sieve_primes(200000)
    
    # We will search primes in increasing order.
    # Search up to 10^7 first, then extend if time permits.
    
    max_limit = 10000000  # 10^7
    
    # Segmented sieve to iterate through primes
    window_size = 1000000
    start = 2
    
    while start < max_limit and time.time() < deadline:
        end = min(start + window_size, max_limit)
        
        # Sieve this window
        sieve = bytearray(b'\x01') * (end - start)
        
        for sp in small_primes:
            if sp * sp > end:
                break
            first = ((start + sp - 1) // sp) * sp
            if first < sp * sp:
                first = sp * sp
            for j in range(first - start, end - start, sp):
                sieve[j] = 0
                
        for i in range(end - start):
            if time.time() > deadline:
                break
            cand = start + i
            if cand < 2:
                continue
            if sieve[i] and is_prime(cand):
                # Quick check: p^2 + 1 must be p-smooth
                # If not, skip this prime entirely
                base = cand * cand
                n1 = base + 1
                if not is_p_smooth(n1, cand, small_primes):
                    continue
                
                k = 1
                for ii in range(2, 201):
                    n = base + ii
                    if is_p_smooth(n, cand, small_primes):
                        k += 1
                    else:
                        break
                
                if k > best_k:
                    best_k = k
                    best_p = cand
                    
        start = end
    
    if best_p == 0:
        best_p = 2
        best_k = 1
        
    return {'p': best_p, 'k': best_k}
