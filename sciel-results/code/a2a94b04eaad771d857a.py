"""Program for: Long runs after p^2 with no prime factor exceeding p (p <= 10^9).
Uses Pollard's rho + Miller-Rabin for fast factorization, and searches primes
efficiently to maximize k within the time budget."""

import time
import random
import math

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

# ---------- Pollard's Rho ----------
def pollard_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        x = random.randrange(2, n)
        y = x
        c = random.randrange(1, n)
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d

# ---------- Factorization ----------
def factor(n, factors=None):
    if factors is None:
        factors = []
    if n == 1:
        return factors
    if is_prime(n):
        factors.append(n)
        return factors
    d = pollard_rho(n)
    factor(d, factors)
    factor(n // d, factors)
    return factors

def all_factors_at_most_p(n, p):
    """Check if all prime factors of n are <= p."""
    if n <= 1:
        return True
    # First try small primes
    for q in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        while n % q == 0:
            if q > p:
                return False
            n //= q
    if n == 1:
        return True
    # Use Pollard-Rho for the rest
    factors = factor(n)
    for f in factors:
        if f > p:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_k = 0
    best_p = 2
    
    # Precompute small primes for trial division
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Search primes p
    # For small p, we can try many. For large p, factorization gets harder but runs might be longer.
    # Strategy: sweep primes in increasing order, but also try some larger ones if time permits.
    
    def check_prime(n):
        if n < 2:
            return False
        if n in small_primes:
            return True
        if n % 2 == 0:
            return False
        for q in small_primes[1:]:
            if q * q > n:
                break
            if n % q == 0:
                return False
        return is_prime(n)
    
    # Generate primes up to some limit using a sieve for speed, then extend with trial division
    # Actually, let's just iterate and check primality.
    
    p = 2
    # Try p=2 first
    k = 0
    while k < 200 and all_factors_at_most_p(4 + k + 1, 2):
        k += 1
    if k > best_k:
        best_k = k
        best_p = 2
    
    # Now iterate odd numbers and check if prime
    for p in range(3, 10**6, 2):  # Start with smaller primes
        if time.time() > deadline:
            break
        if not check_prime(p):
            continue
        
        k = 0
        p_sq = p * p
        while k < 200:
            if time.time() > deadline:
                break
            if not all_factors_at_most_p(p_sq + k + 1, p):
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # If we have time, try larger primes
    # Larger p means p^2 is large, so numbers p^2+i are large.
    # But if p is large, it's more likely that factors are <= p.
    # However, factorization of large numbers is slow.
    # Let's try a few larger primes if time remains.
    
    # We can try random large primes or just continue from where we left off.
    # For now, let's just return the best found.
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
