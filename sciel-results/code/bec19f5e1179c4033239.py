"""Combined program: hybrid divisor + y-scan for 4/n unit fractions."""
# EVOLVE-BLOCK-START
import time
import math
from math import gcd

def _factorize(n):
    """Return dict of prime -> exponent for n."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _divisors_from_factors(factors):
    """Generate all divisors from prime factorization dict."""
    divs = [1]
    for p, e in factors.items():
        new_divs = []
        pe = 1
        for _ in range(e + 1):
            for d in divs:
                new_divs.append(d * pe)
            pe *= p
        divs = new_divs
    return divs

def _try_y_scan(x, n, deadline):
    """Try to find y, z by scanning y values directly. Returns [y, z] or None."""
    A = 4 * x - n
    if A <= 0:
        return None
    B = n * x
    
    y_min = max(x + 1, B // A + 1)
    y_max = 2 * B // A
    
    if y_max < y_min:
        return None
    
    # Limit scan to avoid infinite loops; cap at reasonable range
    scan_limit = min(y_max, y_min + 50000)
    
    for y in range(y_min, scan_limit + 1):
        if time.time() > deadline:
            return None
        denom = A * y - B
        if denom <= 0:
            continue
        if (B * y) % denom == 0:
            z = (B * y) // denom
            if z > y and z <= 10**12:
                return [y, z]
    return None

def _try_divisor_method(x, n, deadline):
    """Try to find y, z using divisor enumeration of B^2. Returns [y, z] or None."""
    A = 4 * x - n
    if A <= 0:
        return None
    B = n * x
    
    y_min = max(x + 1, B // A + 1)
    y_max = 2 * B // A
    
    if y_max < y_min:
        return None
    
    # Factor B^2
    fb = _factorize(B)
    fb2 = {p: 2 * e for p, e in fb.items()}
    
    # Check if number of divisors is manageable
    num_divs = 1
    for e in fb2.values():
        num_divs *= (e + 1)
    
    if num_divs > 200000:
        return None  # Too many divisors, skip
    
    B2 = B * B
    divs = _divisors_from_factors(fb2)
    
    for d in divs:
        if time.time() > deadline:
            return None
        if d >= B:
            continue
        num_y = d + B
        if num_y % A != 0:
            continue
        y = num_y // A
        if y <= x or y < y_min or y > y_max:
            continue
        num_z = B2 // d + B
        if num_z % A != 0:
            continue
        z = num_z // A
        if z <= y or z > 10**12:
            continue
        return [y, z]
    
    return None

def _find_solution_for_n(n, deadline):
    """Find distinct positive integer denominators x,y,z with 1/x+1/y+1/z=4/n."""
    x_min = n // 4 + 1
    x_max = (3 * n) // 4
    
    for x in range(x_min, x_max + 1):
        if time.time() > deadline:
            return None
        
        A = 4 * x - n
        if A <= 0:
            continue
        
        # Try y-scan first (faster for small ranges)
        result = _try_y_scan(x, n, deadline)
        if result is not None:
            return [x] + result
        
        # Fall back to divisor method
        result = _try_divisor_method(x, n, deadline)
        if result is not None:
            return [x] + result
    
    return None

def construct():
    deadline = time.time() + 85
    results = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        results[idx] = _find_solution_for_n(n, deadline)
    
    return results
# EVOLVE-BLOCK-END
