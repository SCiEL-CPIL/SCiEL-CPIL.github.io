"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def _factorize(n):
    """Return list of prime factors of n with multiplicity."""
    factors = []
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    f = 3
    while f * f <= n:
        while n % f == 0:
            factors.append(f)
            n //= f
        f += 2
    if n > 1:
        factors.append(n)
    return factors

def _phi_from_factors(factors):
    """Compute phi from prime factorization (with multiplicity)."""
    if not factors:
        return 1
    # Group by prime
    from collections import Counter
    cnt = Counter(factors)
    result = 1
    for p, k in cnt.items():
        result *= (p - 1) * (p ** (k - 1))
    return result

def _compute_phi(n):
    """Compute Euler's totient of n using trial division."""
    if n <= 0:
        return 0
    result = n
    temp = n
    # Factor out 2s
    if temp % 2 == 0:
        result -= result // 2
        while temp % 2 == 0:
            temp //= 2
    # Odd factors
    f = 3
    while f * f <= temp:
        if temp % f == 0:
            result -= result // f
            while temp % f == 0:
                temp //= f
        f += 2
    if temp > 1:
        result -= result // temp
    return result

def construct():
    """
    Return the smallest n in [1, 1000000] such that phi(n) is divisible by 120.
    Uses direct computation of phi(n) via trial division for small n,
    checking in order until the first valid n is found.
    """
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin for safety
    
    # Check n from 1 upward
    for n in range(1, 1000001):
        if time.time() > deadline:
            break
        if _compute_phi(n) % 120 == 0:
            return n
    
    # Fallback
    return 366
# EVOLVE-BLOCK-END
