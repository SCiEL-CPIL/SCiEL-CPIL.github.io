"""Segmented-sieve smoothness check for long runs after p^2."""
# EVOLVE-BLOCK-START
import time
import math
import random

_DEADLINE = 118.0  # seconds; keep margin before 120s kill

# Deterministic Miller-Rabin bases for n < 3,317,044,064,679,887,385,961,981
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def _is_prime(n):
    """Deterministic Miller-Rabin primality test for 64-bit integers."""
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

def _sieve_primes(limit):
    """Generate all primes up to limit using simple sieve."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def _check_run_segmented(p, max_k=200):
    """
    Use a segmented sieve over [p^2+1, p^2+max_k] to find the longest prefix
    where every number has all prime factors <= p.
    
    For each number in the range, we divide out all primes <= p.
    If the remainder is 1, all factors are <= p.
    """
    base = p * p
    n = max_k
    
    # For small p, p^2 is small enough that we can just trial divide
    # For large p, use segmented sieve approach
    
    # We need primes up to sqrt(p^2 + max_k) for factoring, but we only
    # care about whether remaining factor > p.
    # Key insight: for number v in [p^2+1, p^2+max_k], if v has a prime factor
    # > p, then since v < (p+1)^2 = p^2 + 2p + 1, and v > p^2,
    # any prime factor q > p must satisfy q <= v <= p^2 + max_k.
    # But more importantly, if v has a prime factor q > p, then q must be
    # a "large" prime. Since v <= p^2 + 200, and q > p, we have v = q * m
    # where m < v/q < (p^2+200)/p = p + 200/p <= p + 1 for p >= 200.
    # So m < p + 1, meaning m <= p. Thus v = q * m where q > p and m <= p.
    # This means v is either prime (> p) or a product of one large prime and
    # small factors.
    
    # Strategy: For each v in [p^2+1, p^2+max_k], check if v is p-smooth.
    # v is p-smooth iff after removing all prime factors <= p, remainder is 1.
    # 
    # Efficient approach: precompute primes up to p (or up to sqrt(p^2+max_k)
    # if that's smaller). Then for each v, divide out all such primes.
    
    # Actually, let's think differently. The numbers p^2+1 ... p^2+max_k are
    # all in a small range. We can use a segmented sieve to mark multiples.
    
    # For the purpose of checking p-smoothness:
    # A number v is NOT p-smooth iff it has a prime factor > p.
    # Since v <= p^2 + 200 and v > p^2, any prime factor q > p of v must have
    # the cofactor m = v/q satisfying m < v/p <= p + 200/p.
    # For p >= 200, m <= p. So v = q*m with q > p prime, m composite or prime <= p.
    # This means v/q is a small number (<= p), so we can check:
    #   For each small number m in [1, p], check if v/m is prime and > p.
    # But that's O(p) per v which is too slow for large p.
    
    # Better: Use trial division with primes up to min(p, sqrt(v)).
    # For v ~ p^2, sqrt(v) ~ p. So we need primes up to p.
    # Generating primes up to p for p up to 10^9 is expensive.
    
    # Alternative: Since the range is only 200 numbers, and each number v ~ p^2,
    # we can check if v is p-smooth by:
    # 1. Trial divide by primes up to some bound B
    # 2. Check if remaining factor is 1 (smooth) or prime > p (not smooth)
    #    or composite (need further factorization)
    
    # Let's use a hybrid: trial division by small primes, then check remainder.
    
    # Precompute small primes up to a reasonable bound
    sieve_limit = min(p, 100000)
    small_primes = _sieve_primes(sieve_limit)
    
    k = 0
    for i in range(1, max_k + 1):
        v = base + i
        if v <= p:
            k = i
            continue
        
        # Trial divide by small primes
        rem = v
        for sp in small_primes:
            if sp * sp > rem:
                break
            while rem % sp == 0:
                rem //= sp
            if rem == 1:
                break
            if rem <= p:
                # All remaining factors are <= p (rem is prime or 1)
                break
        
        if rem == 1:
            k = i
            continue
        
        if rem <= p:
            # rem is a prime factor <= p, so v is p-smooth
            k = i
            continue
        
        # rem > p. Check if rem is prime.
        if _is_prime(rem):
            # rem is a prime > p, so v is NOT p-smooth
            break
        
        # rem is composite and > p. It might still be p-smooth if all its
        # prime factors are <= p. Need to check further.
        # Use Pollard-Brent or more trial division.
        # Since rem < v <= p^2 + 200, and rem > p, 
        # rem has at most one prime factor > p (since if two, their product > p^2 >= rem for rem close to p^2... 
        # actually rem could be up to p^2+200, so it could have two factors > p only if both are small,
        # but if both > p then product > p^2, and rem <= p^2+200, so at most one factor > p).
        # Wait: if rem = q1 * q2 * ... and all qi > p, then rem > p^2. But rem <= p^2 + 200.
        # So at most one prime factor of rem can be > p.
        # If rem is composite and > p, it's either:
        #   - product of primes all <= p (p-smooth)
        #   - q * m where q > p is prime and m <= p (not p-smooth)
        
        # To distinguish: check if rem has a prime factor > p.
        # Since at most one such factor exists, we can check:
        #   Is rem/q an integer for some prime q > p? 
        #   Equivalently, does rem have a divisor in (p, rem]?
        #   Since rem is composite, it has a prime factor <= sqrt(rem).
        #   If sqrt(rem) <= p, then all prime factors of rem are <= p, so it's smooth.
        #   If sqrt(rem) > p, then there might be a prime factor in (p, sqrt(rem)].
        
        if math.isqrt(rem) <= p:
            # All prime factors of rem are <= sqrt(rem) <= p
            k = i
            continue
        
        # sqrt(rem) > p. Check if rem has a prime factor in (p, sqrt(rem)].
        # Use trial division from p+1 up to sqrt(rem), but that's too slow.
        # Instead, use Pollard-Brent to find a factor.
        # If the factor found is > p, not smooth. If <= p, divide and recheck.
        
        # Try to find a factor of rem
        found_large = False
        temp = rem
        while temp > 1:
            if temp <= p:
                break
            if _is_prime(temp):
                if temp > p:
                    found_large = True
                break
            # Try to factor temp
            f = _pollard_brent(temp)
            if f == temp:
                # Should not happen
                if temp > p:
                    found_large = True
                break
            if f > p:
                found_large = True
                break
            # f <= p, divide it out
            while temp % f == 0:
                temp //= f
        
        if found_large:
            break
        k = i
    
    return k

def _pollard_brent(n):
    """Pollard's Brent cycle-finding factorization."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randrange(2, n - 1)
        c = random.randrange(1, n - 1)
        m = random.randrange(1, 100)
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
        # restart with new y, c

def construct():
    t0 = time.time()
    best_k = 0
    best_p = 2
    
    # Phase 1: Small primes exhaustive
    small_primes_list = _sieve_primes(1000)
    for p in small_primes_list:
        if time.time() - t0 > _DEADLINE - 5:
            break
        k = _check_run_segmented(p)
        if k > best_k:
            best_k = k
            best_p = p
        if best_k >= 200:
            break
    
    # Phase 2: Systematic search for medium primes using segmented sieve for primality
    if time.time() - t0 < _DEADLINE - 10:
        # Sieve primes in ranges
        start = 1000
        while start < 10**6 and time.time() - t0 < _DEADLINE - 10:
            end = min(start + 10000, 10**6)
            primes_in_range = _sieve_primes(end)
            primes_in_range = [p for p in primes_in_range if p > start]
            for p in primes_in_range:
                if time.time() - t0 > _DEADLINE - 10:
                    break
                k = _check_run_segmented(p)
                if k > best_k:
                    best_k = k
                    best_p = p
                if best_k >= 200:
                    break
            start = end
            if best_k >= 200:
                break
    
    # Phase 3: Random search for larger primes
    if time.time() - t0 < _DEADLINE - 5:
        attempts = 0
        while time.time() - t0 < _DEADLINE - 3 and attempts < 200000:
            p = random.randrange(100000, 10**9, 2)
            if _is_prime(p):
                k = _check_run_segmented(p)
                if k > best_k:
                    best_k = k
                    best_p = p
                    if best_k >= 200:
                        break
            attempts += 1
    
    return {'p': best_p, 'k': best_k}

# EVOLVE-BLOCK-END
