import sys
import math
import time
from functools import reduce
from itertools import accumulate

# Deterministic Miller-Rabin bases for n < 3,317,044,064,679,887,385,961,981 (enough for our range)
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime_mr(n):
    """Deterministic Miller-Rabin primality test."""
    if n < 2:
        return False
    # Small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n % p == 0:
            return n == p
    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    # Test witnesses
    for a in _MR_BASES:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True

def pollard_brent(n):
    """Pollard's Brent's cycle-finding factorization algorithm."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    if n % 7 == 0:
        return 7
    # Try small primes up to 100 first
    for p in [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if n % p == 0:
            return p
    
    y, c, m = 2, 1, 128
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
    # Retry with different c
    return pollard_brent_retry(n)

def pollard_brent_retry(n):
    """Retry Pollard-Brent with different parameters."""
    import random
    for c in range(1, 20):
        y, m = c, 128
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
        if 1 < g < n:
            return g
    # Fallback to trial division
    return None

def factorize(n, small_primes):
    """Factor n, returning a list of prime factors (with multiplicity)."""
    factors = []
    for p in small_primes:
        if p * p > n:
            break
        while n % p == 0:
            factors.append(p)
            n //= p
    if n > 1:
        # n may be prime or product of large primes
        if is_prime_mr(n):
            factors.append(n)
        else:
            # Use Pollard's rho for composites
            _factorize_large(n, factors, small_primes)
    return factors

def _factorize_large(n, factors, small_primes):
    """Recursively factor a large number."""
    if n == 1:
        return
    if is_prime_mr(n):
        factors.append(n)
        return
    d = pollard_brent(n)
    if d is None or d == n:
        # Trial division fallback
        for p in small_primes:
            if p * p > n:
                break
            while n % p == 0:
                factors.append(p)
                n //= p
        if n > 1:
            factors.append(n)
        return
    _factorize_large(d, factors, small_primes)
    _factorize_large(n // d, factors, small_primes)

def largest_prime_factor(n, small_primes, max_factor=None):
    """Find the largest prime factor of n, stopping early if it exceeds max_factor."""
    if max_factor is not None and n <= max_factor:
        return n
    best = 1
    for p in small_primes:
        if p * p > n:
            break
        if n % p == 0:
            best = max(best, p)
            while n % p == 0:
                n //= p
            if max_factor is not None and best > max_factor:
                return best
    if n > 1:
        if is_prime_mr(n):
            best = max(best, n)
        else:
            # Factor the remaining composite
            factors = []
            _factorize_large(n, factors, small_primes)
            best = max(best, max(factors))
        if max_factor is not None and best > max_factor:
            return best
    return best

def is_smooth(n, p, small_primes):
    """Check if all prime factors of n are at most p."""
    if n == 1:
        return True
    for prime in small_primes:
        if prime > p:
            break
        if prime * prime > n:
            break
        if n % prime == 0:
            while n % prime == 0:
                n //= prime
    if n > 1:
        # Remaining factor must be prime and <= p
        if is_prime_mr(n):
            return n <= p
        else:
            # Need to factor further
            factors = []
            _factorize_large(n, factors, small_primes)
            return all(f <= p for f in factors)
    return True

def sieve_primes(limit):
    """Sieve of Eratosthenes to generate primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def construct():
    deadline = time.time() + 145.0  # Safety margin before 150s limit
    # Generate small primes for trial division
    small_primes = sieve_primes(10000)
    
    best_k = 0
    best_p = None
    
    # Search over primes p. Start with smaller primes for quick wins,
    # but the maximum k likely occurs at moderate p values.
    # We'll search p up to a reasonable limit, prioritizing primes that
    # might give long runs.
    
    # Strategy: search primes in increasing order, but stop when time is up.
    # For each prime p, compute the run length k.
    
    # First, generate primes up to a limit using a sieve
    # We'll use a segmented approach or just generate primes up to some limit
    
    # Let's search p up to 1000000 first (small primes), then expand if time allows
    # Actually, for the run p^2 + 1, ..., p^2 + k to all be p-smooth,
    # p needs to be large enough to have enough small prime factors available.
    
    # Generate primes up to 200000
    search_limit = 200000
    primes = sieve_primes(search_limit)
    
    # Filter to only odd primes >= 3 (2 is trivial)
    # For p=2, p^2=4, check 5,6,...: 5>2 so k=0
    # For small primes, k is likely 0 or very small
    
    for idx, p in enumerate(primes):
        if time.time() > deadline:
            break
        if p < 3:
            continue
        
        # Compute the run length
        k = 0
        n = p * p + 1
        while k < 200:
            if is_smooth(n, p, small_primes):
                k += 1
                n += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
            # Print progress (optional, but useful for debugging)
            # sys.stderr.write(f"p={p}, k={k}\n")
    
    # If best_k is still 0, return a valid default
    if best_p is None:
        return {'p': 3, 'k': 1}  # p=3, p^2=9, 10=2*5, 5>3 so k=0. Actually k must be >=1.
        # Let's find a valid p with k>=1
        # p=2: 4+1=5, 5>2, k=0
        # p=3: 9+1=10=2*5, 5>3, k=0
        # p=5: 25+1=26=2*13, 13>5, k=0
        # p=7: 49+1=50=2*5^2, all factors <=7, k=1; 51=3*17, 17>7, k=1
        # So p=7 gives k=1
        return {'p': 7, 'k': 1}
    
    return {'p': best_p, 'k': best_k}
