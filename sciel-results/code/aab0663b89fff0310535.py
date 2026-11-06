"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def _largest_prime_factor(x, primes):
    """Compute the largest prime factor of x via trial division using precomputed primes."""
    if x <= 1:
        return 1
    max_pf = 1
    for p in primes:
        if p * p > x:
            break
        while x % p == 0:
            max_pf = p
            x //= p
    if x > 1:
        max_pf = x
    return max_pf

def _sieve(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(limit + 1) if is_prime[i]]

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    LO, HI = 10000, 1000000
    
    # Precompute primes up to sqrt(HI) for efficient trial division
    prime_limit = int(math.isqrt(HI)) + 1
    primes = _sieve(prime_limit)
    
    best_n = LO
    best_lpf = float('inf')
    
    # First pass: scan all n in [LO, HI] but compute LPF only for n(n+1)
    # To speed up, we can first check if n or n+1 has a large prime factor quickly
    # A number x has a prime factor > P iff after dividing by all primes <= P, x > 1
    # But we need the actual largest prime factor value, not just whether it's large.
    
    # Strategy: For each n, compute L = n*(n+1). Since gcd(n, n+1)=1,
    # LPF(L) = max(LPF(n), LPF(n+1))
    # So we can compute LPF(n) and LPF(n+1) separately, which is faster than LPF(n*(n+1)).
    
    # Even better: precompute LPF for all numbers up to HI using a sieve-like approach.
    # LPF[x] = largest prime factor of x. We can compute this efficiently.
    
    # Compute LPF array for all x in [0, HI]
    lpf = [0] * (HI + 1)
    for i in range(2, HI + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, HI + 1, i):
                lpf[j] = i  # i is the largest prime factor seen so far for j
    
    # Now scan for the n that minimizes max(lpf[n], lpf[n+1])
    for n in range(LO, HI + 1):
        if time.time() > deadline:
            break
        val = max(lpf[n], lpf[n+1])
        if val < best_lpf:
            best_lpf = val
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
