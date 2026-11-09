import time
from fractions import Fraction
from math import isqrt

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256
DEADLINE = 55.0

def _get_divisors(n):
    """Return list of divisors of n."""
    if n == 0:
        return []
    divs = []
    for i in range(1, isqrt(n) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return divs

def _solve_2_terms(p, q, max_d, exclude):
    """
    Find a, b such that 1/a + 1/b = p/q, a < b, a,b <= max_d, a,b not in exclude.
    Uses (pa-q)(pb-q) = q^2.
    """
    if p == 0 or q == 0:
        return None
    if p < 0 or q < 0:
        return None
    
    # 1/a + 1/b = p/q => q(a+b) = pab => p^2 ab - pq(a+b) = 0 => (pa-q)(pb-q) = q^2
    # Let X = pa-q, Y = pb-q. XY = q^2.
    # a = (X+q)/p, b = (Y+q)/p.
    # a, b > 0 => X, Y > 0.
    # a < b => X < Y.
    # X < q < Y since XY=q^2 and X<Y.
    
    divs = _get_divisors(q * q)
    for X in divs:
        if X >= q:
            break
        # Check divisibility for a
        if (X + q) % p != 0:
            continue
        a = (X + q) // p
        Y = (q * q) // X
        if (Y + q) % p != 0:
            continue
        b = (Y + q) // p
        
        if a >= b:
            continue
        if a > max_d or b > max_d:
            continue
        if a in exclude or b in exclude:
            continue
        if a == b:
            continue
            
        return [a, b]
    return None

def _search_for_M(M, deadline):
    """
    Check if 5/31 can be represented as sum of distinct unit fractions
    with denominators in [1, M].
    Returns list of denominators or None.
    """
    target = Fraction(5, 31)
    
    # Try 2 terms
    res = _solve_2_terms(5, 31, M, set())
    if res:
        return res

    # Try 3 terms: 1/c + 1/a + 1/b = 5/31
    # 1/a + 1/b = 5/31 - 1/c = (5c-31)/(31c)
    for c in range(1, M + 1):
        if time.time() > deadline:
            return None
        p = 5 * c - 31
        q = 31 * c
        if p <= 0:
            continue
        # Simplify p/q? _solve_2_terms works with any p,q, but smaller is better for divisors.
        from math import gcd
        g = gcd(p, q)
        p //= g
        q //= g
        
        res = _solve_2_terms(p, q, M, {c})
        if res:
            return [c] + res

    # Try 4 terms: 1/d + 1/c + 1/a + 1/b = 5/31
    # 1/a + 1/b = 5/31 - 1/c - 1/d
    # Iterate c, d. To avoid duplicates, c < d.
    for c in range(1, M + 1):
        if time.time() > deadline:
            return None
        rem_after_c = target - Fraction(1, c)
        if rem_after_c <= 0:
            continue
        
        # We need 1/d + 1/a + 1/b = rem_after_c
        # 1/a + 1/b = rem_after_c - 1/d
        # Iterate d > c
        for d in range(c + 1, M + 1):
            if time.time() > deadline:
                return None
            rem_after_cd = rem_after_c - Fraction(1, d)
            if rem_after_cd <= 0:
                continue
            
            p = rem_after_cd.numerator
            q = rem_after_cd.denominator
            
            res = _solve_2_terms(p, q, M, {c, d})
            if res:
                return [c, d] + res

    return None

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Known solution: [8, 31, 248], max = 248
    best_solution = [8, 31, 248]
    best_max = 248
    
    # Try to find solutions with max < best_max
    # Iterate M downwards from best_max - 1
    for M in range(best_max - 1, 1, -1):
        if time.time() > deadline:
            break
        
        res = _search_for_M(M, deadline)
        if res is not None:
            # Validate
            if len(set(res)) == len(res) and all(1 <= d <= MAX_DENOM for d in res) and sum(Fraction(1, d) for d in res) == TARGET:
                return res
                
    return best_solution
