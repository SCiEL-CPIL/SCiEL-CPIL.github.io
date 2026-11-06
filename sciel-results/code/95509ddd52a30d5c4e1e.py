"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import isqrt

def _factorize(n):
    """Return dict of prime->exponent for n."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _divisors_from_factors(factors):
    """Generate all divisors from prime factorization dict."""
    divs = [1]
    for p, e in factors.items():
        new_divs = []
        pe = 1
        for i in range(e + 1):
            for d in divs:
                new_divs.append(d * pe)
            pe *= p
        divs = new_divs
    return divs

def construct():
    deadline = time.time() + 85
    results = [None] * 254
    BOUND = 10**12
    
    # Precompute factorizations for small numbers up to ~50000
    # b = n*x, n<=256, x<=~192, so b <= ~49152
    max_b = 256 * 192 + 100
    factor_cache = {}
    for i in range(2, max_b + 1):
        factor_cache[i] = _factorize(i)
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        
        x_lo = n // 4 + 1
        x_hi = (3 * n + 3) // 4
        
        for x in range(x_lo, x_hi + 1):
            a = 4 * x - n
            b = n * x
            if a <= 0:
                continue
            
            # (ay-b)(az-b) = b^2
            # U = ay-b, V = az-b, UV = b^2
            # y = (U+b)/a, z = (V+b)/a
            # Need: a | (U+b), a | (V+b), y > x, z > y, all distinct, <= BOUND
            
            b_sq = b * b
            
            min_U = 2 * x * (2 * x - n) + 1  # U must be > ax - b
            if min_U < 1:
                min_U = 1
            
            found_for_x = None
            
            # Get divisors of b_sq
            # Factor b first, then square exponents for b_sq
            b_factors = factor_cache.get(b)
            if b_factors is None:
                b_factors = _factorize(b)
            b_sq_factors = {}
            for p, e in b_factors.items():
                b_sq_factors[p] = e * 2
            
            divs = _divisors_from_factors(b_sq_factors)
            
            for U in divs:
                if U < min_U:
                    continue
                V = b_sq // U
                if V < min_U:
                    continue
                # Check y and z are integers
                if (U + b) % a != 0:
                    continue
                if (V + b) % a != 0:
                    continue
                y = (U + b) // a
                z = (V + b) // a
                # Ensure y <= z for ordering
                if y > z:
                    y, z = z, y
                # Check distinctness and bounds
                if y <= x or z <= y:
                    continue
                if y > BOUND or z > BOUND:
                    continue
                if x == y or y == z or x == z:
                    continue
                found_for_x = [x, y, z]
                break
            
            if found_for_x:
                found = found_for_x
                break
        
        results[idx] = found
    
    return results
# EVOLVE-BLOCK-END
