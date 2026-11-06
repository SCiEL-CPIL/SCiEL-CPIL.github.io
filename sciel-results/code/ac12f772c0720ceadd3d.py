import time
from fractions import Fraction
from math import isqrt

def construct():
    deadline = time.time() + 55
    target = Fraction(5, 31)
    best_solution = [8, 31, 248]
    best_max = 248

    def find_2_terms(rem, max_denom):
        """Find a, b < max_denom, a != b, such that 1/a + 1/b = rem.
        Returns (a, b) or None."""
        if rem <= 0:
            return None
        
        # 1/a + 1/b = p/q
        p = rem.numerator
        q = rem.denominator
        # (qa - p)(qb - p) = p^2
        # Let X = qa - p, Y = qb - p
        # X * Y = p^2
        # X must be a divisor of p^2
        # Also X = qa - p => X + p must be divisible by q
        # And a = (X + p) / q < max_denom
        # Similarly for b.
        
        # Iterate divisors of p^2
        # p^2 can be large if rem is complex, but rem is derived from 5/31 - 1/c
        # rem = (5c - 31) / (31c)
        # p = 5c - 31, q = 31c
        # p^2 is roughly 25c^2. For c < 248, p^2 < 25 * 60000 ~ 1.5M.
        # Factoring or iterating divisors is fast.
        
        p2 = p * p
        # Get divisors
        divisors = []
        for i in range(1, isqrt(p2) + 1):
            if p2 % i == 0:
                divisors.append(i)
                if i != p2 // i:
                    divisors.append(p2 // i)
        
        for X in divisors:
            if (X + p) % q != 0:
                continue
            a = (X + p) // q
            if a <= 0 or a >= max_denom:
                continue
            # Calculate Y
            Y = p2 // X
            if (Y + p) % q != 0:
                continue
            b = (Y + p) // q
            if b <= 0 or b >= max_denom:
                continue
            if a == b:
                continue
            return (min(a, b), max(a, b))
        return None

    def find_3_terms(rem, max_denom, exclude=None):
        """Find a, b, c < max_denom, distinct, such that 1/a + 1/b + 1/c = rem.
        Returns (a, b, c) or None."""
        if rem <= 0:
            return None
        
        # 1/a + 1/b + 1/c = rem
        # Iterate a from 1 to max_denom-1
        # rem' = rem - 1/a
        # Find 1/b + 1/c = rem'
        
        # Pruning: 1/a < rem => a > 1/rem
        # Also 3/a >= rem => a <= 3/rem
        a_min = max(1, int(1 / float(rem)) + 1)
        a_max = min(max_denom - 1, int(3 / float(rem)))
        
        for a in range(a_min, a_max + 1):
            if exclude is not None and a in exclude:
                continue
            rem_prime = rem - Fraction(1, a)
            if rem_prime <= 0:
                continue
            # Check if rem_prime can be 2 terms with denom < max_denom
            # Also b, c must be distinct from a
            res = find_2_terms(rem_prime, max_denom)
            if res:
                b, c = res
                if b != a and c != a:
                    return (a, b, c)
        return None

    def find_4_terms(rem, max_denom, exclude=None):
        """Find 4 distinct terms < max_denom summing to rem."""
        if rem <= 0:
            return None
        
        a_min = max(1, int(1 / float(rem)) + 1)
        a_max = min(max_denom - 1, int(4 / float(rem)))
        
        for a in range(a_min, a_max + 1):
            if exclude is not None and a in exclude:
                continue
            rem_prime = rem - Fraction(1, a)
            if rem_prime <= 0:
                continue
            # Find 3 terms for rem_prime with denom < max_denom, excluding a
            res = find_3_terms(rem_prime, max_denom, exclude={a} if exclude is None else exclude | {a})
            if res:
                b, c, d = res
                if a not in (b, c, d):
                    return (a, b, c, d)
        return None

    # Search for solutions with max denominator < 248
    # We iterate C from 1 to 247.
    # If a solution exists with max C, it contains C.
    # So we check if 5/31 - 1/C can be formed by distinct denominators < C.
    
    for C in range(1, 248):
        if time.time() > deadline:
            break
        
        rem = target - Fraction(1, C)
        if rem <= 0:
            continue
        
        # Try 2 terms (total 3 terms including C)
        res = find_2_terms(rem, C)
        if res:
            a, b = res
            # Check distinctness
            if a != C and b != C and a != b:
                return [a, b, C]
        
        # Try 3 terms (total 4 terms including C)
        # Only if time permits? 2 terms is very fast. 3 terms is slower.
        # Let's try 3 terms only for small C or if we have time.
        if C < 100: # Heuristic: 3-term search is more expensive
             res = find_3_terms(rem, C, exclude={C})
             if res:
                a, b, d = res
                if C not in (a, b, d):
                    return [a, b, d, C]
        
        # Try 4 terms (total 5 terms including C) - only for very small C
        if C < 30:
             res = find_4_terms(rem, C, exclude={C})
             if res:
                a, b, d, e = res
                if C not in (a, b, d, e):
                    return [a, b, d, e, C]

    # If no better solution found, return the known one
    return best_solution
