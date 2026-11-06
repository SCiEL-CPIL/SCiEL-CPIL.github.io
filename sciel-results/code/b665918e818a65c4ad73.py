import time
import math
import random

# ── Deterministic Miller-Rabin for n < 2^64 ──────────────────────────────────
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
                 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
                 101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
                 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]

def is_prime(n):
    """Deterministic Miller-Rabin. Exact for n < 2^64 with 12 base primes."""
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

# ── Pollard's Rho / Brent factorisation ──────────────────────────────────────

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
        # retry with different parameters
        return _pollard_brent(n)
    return g

def is_p_smooth(n, p):
    """
    Check if all prime factors of n are <= p.
    Uses the property: if n <= p, it is p-smooth.
    If n > p and prime, it is NOT p-smooth.
    If n > p and composite, check factors.
    """
    if n <= 1:
        return True
    
    # 1. Strip small primes
    for q in _SMALL_PRIMES:
        if q > p:
            break
        if n % q == 0:
            while n % q == 0:
                n //= q
            if n == 1:
                return True
            # If n <= p, remaining factors are < n <= p, so smooth
            if n <= p:
                return True
    
    if n == 1:
        return True
    
    # 2. If n <= p, it is p-smooth (factors are < n <= p)
    if n <= p:
        return True
    
    # 3. If n > p, check if prime
    if is_prime(n):
        return False  # n is a prime factor > p
    
    # 4. n is composite and > p. Find a factor.
    # We can limit the search or just use Pollard-Brent.
    # Since n ~ p^2, factors are likely <= p.
    d = _pollard_brent(n)
    
    # Recurse on factors
    if d > p:
        # d is a factor > p. If d is prime, fail. If d is composite, it has a prime factor > p?
        # Not necessarily. d could be composite with factors <= p.
        # But we need to check if d itself is p-smooth.
        pass
        
    return is_p_smooth(d, p) and is_p_smooth(n // d, p)

def _gen_primes_upto(limit):
    """Sieve of Eratosthenes to generate primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            start = i * i
            step = i
            sieve[start::step] = bytearray(len(sieve[start::step]))
    return [i for i, is_p in enumerate(sieve) if is_p]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_k = 0
    best_p = 2
    
    # Phase 1: Exhaustive search for small primes
    # Sieve up to 5,000,000
    PRIME_LIMIT = 5_000_000
    primes = _gen_primes_upto(PRIME_LIMIT)
    
    for p in primes:
        if time.time() > deadline:
            break
        base = p * p
        k = 0
        # Check up to 200
        for i in range(1, 201):
            val = base + i
            if not is_p_smooth(val, p):
                break
            k += 1
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 2: Search for larger primes
    # We can't sieve up to 10^9. We will sample.
    # Strategy: Check intervals.
    # The check is fast now.
    # Let's try to find primes in [5e6, 10e6], [10e6, 20e6], etc.
    # Or just random sampling?
    # Random sampling might miss the best p.
    # Let's do a sequential scan of a large chunk, then random.
    
    # Sequential scan from 5e6 to 10e6 (approx 300k primes)
    # We need to find primes in this range.
    # We can use a segmented sieve or just trial division/Miller-Rabin on odds.
    # Miller-Rabin is fast.
    
    current = PRIME_LIMIT + 1
    if current % 2 == 0:
        current += 1
        
    # Scan up to 10,000,000
    SCAN_LIMIT = 10_000_000
    while current <= SCAN_LIMIT:
        if time.time() > deadline:
            break
        if is_prime(current):
            p = current
            base = p * p
            k = 0
            for i in range(1, 201):
                val = base + i
                if not is_p_smooth(val, p):
                    break
                k += 1
            if k > best_k:
                best_k = k
                best_p = p
        current += 2
        
    # If time remains, random sampling up to 10^9
    # Larger p might allow longer runs?
    # Probability of p^2+i being p-smooth is related to the smoothness probability.
    # u = log(n)/log(p) = log(p^2)/log(p) = 2.
    # The probability that a number n is p-smooth where log n / log p = u is roughly 1/u^u * something?
    # Dickman function rho(u). For u=2, rho(2) = 1 - ln(2) ≈ 0.306.
    # So ~30% chance for each i.
    # Expected run length = 1 / (1 - 0.306) ≈ 1.43? No, that's geometric.
    # Wait, if probability of success is P, expected run is 1/(1-P).
    # If P=0.3, run is ~1.4.
    # But we want max k.
    # For larger p, u is still ~2.
    # However, the "small prime" stripping might be less effective if p is large?
    # No, small primes are small.
    
    # Let's just do random sampling to see if we hit a lucky streak.
    # 10^9 is large.
    while time.time() < deadline - 5:
        # Pick a random odd number in [10^7, 10^9]
        # Avoid numbers divisible by small primes to increase prime hit rate
        # Just pick random odd
        p_candidate = random.randint(10_000_000, 1_000_000_000)
        if p_candidate % 2 == 0:
            p_candidate += 1
            
        # Quick check for small factors to avoid Miller-Rabin on composites
        is_prem = True
        for q in _SMALL_PRIMES:
            if p_candidate % q == 0:
                if p_candidate == q:
                    break
                is_prem = False
                break
        if not is_prem:
            continue
            
        if is_prime(p_candidate):
            p = p_candidate
            base = p * p
            k = 0
            for i in range(1, 201):
                val = base + i
                if not is_p_smooth(val, p):
                    break
                k += 1
            if k > best_k:
                best_k = k
                best_p = p
                
    return {'p': best_p, 'k': best_k}
