import time
from math import isqrt
from functools import lru_cache

def _factorize(m):
    """Return dict of prime -> exponent for m."""
    factors = {}
    d = 2
    while d * d <= m:
        while m % d == 0:
            factors[d] = factors.get(d, 0) + 1
            m //= d
        d += 1
    if m > 1:
        factors[m] = factors.get(m, 0) + 1
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

def _solve_one(n, deadline):
    """Find distinct x,y,z with 1/x+1/y+1/z=4/n. Returns [x,y,z] or None."""
    # Try x = n + k for k = 1..10
    for k in range(1, 11):
        if time.time() > deadline:
            return None
        x = n + k
        a = 3 * n + 4 * k  # numerator of (1/y + 1/z)
        b = n * x           # denominator of (1/y + 1/z)
        # (ay - b)(az - b) = b^2
        # d = ay - b must divide b^2, and d > b (so y > x roughly)
        # y = (d + b) / a, z = (b^2/d + b) / a
        # Need: a | (d+b), a | (b^2/d + b), y != x, z != x, y != z, all positive, <= 10^12
        b2 = b * b
        # Factorize b to get divisors of b^2
        fac_b = _factorize(b)
        # b^2 has each exponent doubled
        fac_b2 = {p: e * 2 for p, e in fac_b.items()}
        divs = _divisors_from_factors(fac_b2)
        # For each divisor d of b^2 with d > b:
        for d in divs:
            if d <= b:
                continue
            if d + b < a:
                continue
            # Check a | (d + b)
            if (d + b) % a != 0:
                continue
            y = (d + b) // a
            # Check y > 0, y != x
            if y <= 0 or y == x:
                continue
            # z = (b^2/d + b) / a
            rem = b2 // d  # since d divides b^2
            if (rem + b) % a != 0:
                continue
            z = (rem + b) // a
            if z <= 0 or z == x or z == y:
                continue
            if y > 1000000000000 or z > 1000000000000:
                continue
            # Verify: 1/x + 1/y + 1/z == 4/n via cross multiplication
            # n*(y*z + x*z + x*y) == 4*x*y*z
            lhs = n * (y * z + x * z + x * y)
            rhs = 4 * x * y * z
            if lhs == rhs:
                return [x, y, z]
    return None

def construct():
    deadline = time.time() + 50.0  # safety margin before 60s
    answers = []
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        result = _solve_one(n, deadline)
        answers.append(result)
    return answers
