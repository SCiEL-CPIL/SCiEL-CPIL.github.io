"""Program for: Long runs after p^2 with no prime factor exceeding p (p <= 10^9).
Uses a direct approach: search for primes p and check consecutive integers p^2+1, ..., p^2+k
for p-smoothness (all prime factors <= p). Uses efficient trial division and Pollard-Brent
with deterministic Miller-Rabin for factorization. Maximizes k within time budget."""

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

# ---------- Factorization with early termination for smoothness check ----------
def is_p_smooth(n, p, small_primes=None):
    """Check if all prime factors of n are <= p. Returns True if n is 1 or p-smooth."""
    if n <= 1:
        return True
    
    # Try small primes first with trial division
    if small_primes is None:
        small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)
    for q in small_primes:
        if q > p:
            break
        while n % q == 0:
            n //= q
        if n == 1:
            return True
    
    if n == 1:
        return True
    
    # If n is prime, check if it's <= p
    if is_prime(n):
        return n <= p
    
    # Use Pollard-Brent to find a factor, then recursively check
    # We need to find all factors and check each is <= p
    factors = []
    _factor(n, factors, p)
    for f in factors:
        if f > p:
            return False
    return True

def _factor(n, factors, p):
    """Recursively factor n, adding factors to list. Early termination if factor > p."""
    if n == 1:
        return
    if is_prime(n):
        factors.append(n)
        return
    d = pollard_brent(n)
    _factor(d, factors, p)
    _factor(n // d, factors, p)

# Precompute small primes for fast trial division
_SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499)

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

def segmented_smoothness_check(p, small_primes):
    """
    For a given prime p, check p^2+1, ..., p^2+k for p-smoothness
    using segmented sieve over the interval [p^2, p^2+200].
    Returns the maximum k such that all numbers p^2+1..p^2+k are p-smooth.
    """
    p_sq = p * p
    # We need to check p^2+1 through p^2+200
    # Create arrays to track remaining values and whether they're p-smooth
    n = 201  # indices 0..200, where index i corresponds to p^2 + i
    
    # rem[i] = remaining value after dividing out small primes
    rem = [0] * n
    for i in range(n):
        rem[i] = p_sq + i
    
    # For each small prime q <= p, divide it out of all rem[i]
    for q in small_primes:
        if q > p:
            break
        # Find the first multiple of q in [p^2, p^2+200]
        # We want to divide q out of rem[i] for all i where (p^2 + i) % q == 0
        start = p_sq
        # First index where (p_sq + i) % q == 0
        # (p_sq + i) % q == 0 => i % q == (-p_sq) % q
        first_i = (-p_sq) % q
        # But we need i >= 0 and i < n
        # Actually, let's just iterate over all i and check
        for i in range(n):
            if rem[i] % q == 0:
                while rem[i] % q == 0:
                    rem[i] //= q
                # If rem[i] becomes 1, it's fully factored by small primes
                # If rem[i] > 1, it has a prime factor > all small primes we've tried
                # But we need to check if that factor is <= p
    
    # Now for each i from 1 to 200, check if rem[i] is p-smooth
    # rem[i] should be 1 or a prime > 97 (largest small prime)
    # If rem[i] is prime, check if it's <= p
    # If rem[i] is composite, it must have a factor <= sqrt(rem[i])
    # But since we divided all small primes up to p, if rem[i] > 1 and not prime,
    # it must have a prime factor > p (contradiction) or we missed something
    
    # Actually, let's be more careful. After dividing all primes q <= p from rem[i],
    # if rem[i] > 1, then rem[i] must be prime (since any composite number has a
    # prime factor <= sqrt(rem[i]), and if that factor were <= p, we would have divided it).
    # Wait, that's not quite right. We only divided primes up to p, but rem[i] could
    # still be composite with factors all > p? No, if rem[i] is composite, it has a
    # prime factor <= sqrt(rem[i]). If sqrt(rem[i]) <= p, we would have divided it.
    # If sqrt(rem[i]) > p, then rem[i] > p^2, but rem[i] <= p^2 + 200, so this can't happen
    # for large p. For small p, we need to be careful.
    
    # Let's just check: if rem[i] == 1, it's p-smooth.
    # If rem[i] > 1, check if it's prime and <= p, or if it's composite with all factors <= p.
    # Since we divided all primes <= p, if rem[i] > 1, it must be prime (or 1).
    # Wait, that's only true if we divided ALL primes <= p. Let's verify.
    
    # For correctness, let's check rem[i] more carefully
    k = 0
    for i in range(1, n):
        r = rem[i]
        if r == 1:
            k = i
            continue
        # r > 1: check if r is prime and r <= p
        if is_prime(r):
            if r <= p:
                k = i
            else:
                break
        else:
            # r is composite. This shouldn't happen if we divided all primes <= p,
            # but let's handle it by checking if all factors are <= p
            # For safety, use the original is_p_smooth
            if not is_p_smooth(p_sq + i, p, small_primes):
                break
            k = i
    
    return k

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_k = 0
    best_p = 2
    
    # Generate primes up to a reasonable limit using sieve
    # Focus on primes where p^2+i numbers are likely to have small factors
    sieve_limit = 10_000_000
    primes = sieve_primes(sieve_limit)
    
    # For each prime p, check consecutive integers p^2+1, ..., p^2+k
    # for p-smoothness. We want to maximize k.
    
    # Strategy: 
    # 1. Try all primes up to sieve_limit with fast pre-filter
    # 2. For each, compute how far the run goes
    # 3. Keep track of best
    
    for idx, p in enumerate(primes):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        # Quick pre-filter: check if p^2+1 has any small prime factor > p
        # If p is small, most numbers p^2+i will have large prime factors
        # Only bother with detailed check if p is reasonably large
        
        if p < 100:
            # For very small p, the run will be very short, skip detailed check
            continue
            
        # Use segmented smoothness check for fast verification
        k = segmented_smoothness_check(p, _SMALL_PRIMES)
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # If time remains, try some larger primes
    if time.time() - start_time < deadline - 30:
        # Try primes in ranges larger than sieve_limit
        # Sample random larger primes
        for _ in range(500):
            if time.time() - start_time > deadline - 5:
                break
            
            # Random prime in [sieve_limit, 10^9]
            lo = max(sieve_limit + 1, 10**6)
            hi = min(10**9, 5*10**8)
            candidate = random.randrange(lo, hi + 1)
            if candidate % 2 == 0:
                candidate += 1
            
            # Find next prime
            while not is_prime(candidate):
                candidate += 2
                if candidate > hi:
                    break
            
            if candidate > hi:
                continue
            
            # Use segmented smoothness check for fast verification
            k = segmented_smoothness_check(candidate, _SMALL_PRIMES)
            
            if k > best_k:
                best_k = k
                best_p = candidate
    
    return {'p': best_p, 'k': best_k}
