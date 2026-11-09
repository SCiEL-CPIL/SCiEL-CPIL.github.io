"""Program for: Long runs after p^2 with no prime factor exceeding p (p <= 10^9).
Structural change: Use segmented sieve for prime discovery in large ranges,
early-exit factorization, and focus on larger primes where smoothness is easier."""

import time
import math
import random
from array import array

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

# ---------- Pollard-Brent ----------
def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    if n % 7 == 0:
        return 7
    if n % 11 == 0:
        return 11
    if n % 13 == 0:
        return 13
    while True:
        y = random.randrange(1, n)
        c = random.randrange(1, n)
        m = 128
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
                if g != 1:
                    break
        if g != n:
            return g

# ---------- Small primes for pre-filtering ----------
_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
                 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]

def has_large_factor(n, p):
    """Return True if n has a prime factor > p. Early exit."""
    if n <= 1:
        return False
    # Try small primes first
    for q in _SMALL_PRIMES:
        if q > p:
            break
        if n % q == 0:
            while n % q == 0:
                n //= q
    if n == 1:
        return False
    # If n itself is prime and > p, we have a large factor
    if is_prime(n):
        return n > p
    # Use Pollard-Brent to find a factor
    d = pollard_brent(n)
    if d > p:
        return True
    # Check if d has a factor > p
    if has_large_factor(d, p):
        return True
    if has_large_factor(n // d, p):
        return True
    return False

def all_factors_at_most_p(n, p):
    """Check if all prime factors of n are <= p."""
    return not has_large_factor(n, p)

# ---------- Segmented sieve for finding primes in [lo, hi] ----------
def sieve_small_primes(limit):
    """Generate primes up to limit for segmented sieve."""
    if limit < 2:
        return []
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = 0
    is_p[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return [i for i in range(2, limit + 1) if is_p[i]]

def segmented_sieve(lo, hi, small_primes):
    """Find all primes in [lo, hi] using segmented sieve."""
    if hi < 2:
        return []
    lo = max(lo, 2)
    size = hi - lo + 1
    is_p = bytearray(b'\x01') * size
    for p in small_primes:
        if p * p > hi:
            break
        # Find first multiple of p in [lo, hi]
        start = ((lo + p - 1) // p) * p
        if start == p:
            start += p
        for j in range(start - lo, size, p):
            is_p[j] = 0
    return [lo + i for i in range(size) if is_p[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_k = 0
    best_p = 2
    
    # Generate small primes for segmented sieve (up to sqrt(10^9) ~ 31623)
    sieve_limit = 40000
    small_primes = sieve_small_primes(sieve_limit)
    
    # Strategy: Search in segments from high to low, prioritizing larger primes
    # where smoothness constraints are easier to satisfy.
    # For p > 10^6, p^2 + i for i <= 200 has p^2 + i < p^2 + p (for p > 200),
    # so any factor > p must be at least... well, the number itself could be prime > p.
    # But composites with all factors <= p are more common for large p.
    
    # First, do a quick scan of smaller primes to establish a baseline
    for p in small_primes:
        if time.time() > deadline:
            break
        if p > 100000:
            break
        p_sq = p * p
        k = 0
        while k < 200:
            if time.time() > deadline:
                break
            if not all_factors_at_most_p(p_sq + k + 1, p):
                break
            k += 1
        if k > best_k:
            best_k = k
            best_p = p
    
    # Now search larger primes using segmented sieve in segments
    # Focus on ranges where primes are dense enough and p is large
    # Try segments from high to low
    segment_size = 50000
    # Start from high end and work down
    search_hi = 10**9
    search_lo = 10**6
    
    # Try several segments, prioritizing larger p values
    segments_to_try = []
    # Generate candidate segments, focusing on larger primes
    for start in range(10**9, 10**6, -segment_size):
        segments_to_try.append((start, min(start + segment_size, 10**9)))
    
    # Also add some mid-range segments
    for start in range(10**7, 10**6, -segment_size):
        segments_to_try.append((start, min(start + segment_size, 10**7)))
    
    random.shuffle(segments_to_try)
    
    for seg_lo, seg_hi in segments_to_try:
        if time.time() > deadline:
            break
        if seg_lo < 10**6:
            continue
        
        # Get primes in this segment
        primes_in_seg = segmented_sieve(seg_lo, seg_hi, small_primes)
        
        # Test primes in this segment, starting from larger ones
        for p in reversed(primes_in_seg):
            if time.time() > deadline:
                break
            if p <= best_p and best_k >= 10:
                # Only continue if we might beat the best
                pass
            
            p_sq = p * p
            k = 0
            while k < 200:
                if time.time() > deadline:
                    break
                if not all_factors_at_most_p(p_sq + k + 1, p):
                    break
                k += 1
            
            if k > best_k:
                best_k = k
                best_p = p
    
    return {'p': best_p, 'k': best_k}
