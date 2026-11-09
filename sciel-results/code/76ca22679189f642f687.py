import time
from fractions import Fraction
from math import isqrt

def construct():
    """Find Egyptian fraction of 5/31 minimizing the largest denominator.
    
    Uses number-theoretic divisor enumeration to solve the Diophantine
    constraints exactly. For 2-term and 3-term solutions, the problem reduces
    to finding divisors of specific integers, which can be done in O(sqrt(N)).
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    target = Fraction(5, 31)
    
    # Fallback known solution
    fallback = [8, 31, 248]
    
    def find_two_terms_fast(rem, min_d, max_d):
        """Find distinct b, c in [min_d, max_d] with 1/b + 1/c = rem.
        Uses the formula: if rem = p/q in lowest terms, then
        b = q*(k+1)/p, c = q*(k+q)/p for divisors k of q^2.
        Actually, let's use: 1/b + 1/c = p/q => q(b+c) = pq => 
        (pb - q)(pc - q) = q^2.
        So we need divisors of q^2.
        """
        # rem = p/q in lowest terms
        p = rem.numerator
        q = rem.denominator
        
        # (p*b - q) * (p*c - q) = q^2
        # Let x = p*b - q, y = p*c - q
        # x * y = q^2
        # b = (x + q) / p, c = (y + q) / p
        # Need x, y to be positive divisors of q^2 with x <= y (for b <= c)
        # And (x + q) % p == 0, (y + q) % p == 0
        # And min_d <= b < c <= max_d
        
        q_sq = q * q
        sqrt_q_sq = isqrt(q_sq)
        
        for x in range(1, sqrt_q_sq + 1):
            if q_sq % x != 0:
                continue
            y = q_sq // x
            
            # x <= y so b <= c
            if x > y:
                continue
            
            # Check divisibility
            if (x + q) % p != 0:
                continue
            if (y + q) % p != 0:
                continue
            
            b = (x + q) // p
            c = (y + q) // p
            
            if b < min_d:
                continue
            if c > max_d:
                continue
            if b >= c:
                continue
            if b < min_d or c > max_d:
                continue
            
            return (b, c)
        
        return None
    
    def find_three_terms_fast(rem, min_d, max_d):
        """Find distinct a, b, c in [min_d, max_d] with 1/a + 1/b + 1/c = rem.
        Fix c (the largest), then find a, b with 1/a + 1/b = rem - 1/c.
        We iterate c from max_d down to min_d+2, and for each c, solve the 2-term problem.
        Actually, we want to find ANY solution, so we can iterate c from min_d+2 to max_d.
        But to minimize the largest denominator, we should check if a solution exists for a given max_d.
        
        Better approach: iterate c from max_d down to 2*min_d+1.
        For each c, compute rem2 = rem - 1/c. If rem2 <= 0, break.
        Then find a, b with 1/a + 1/b = rem2 where min_d <= a < b < c.
        """
        for c in range(max_d, min_d + 1, -1):
            rem2 = rem - Fraction(1, c)
            if rem2 <= 0:
                break
            
            # Find a, b with 1/a + 1/b = rem2, min_d <= a < b < c
            pair = find_two_terms_fast(rem2, min_d, c - 1)
            if pair is not None:
                a, b = pair
                # Verify a < b < c
                if a < b < c:
                    return (a, b, c)
        
        return None
    
    # Check 1-term (impossible for 5/31 > 1)
    # Check 2-term
    sol = find_two_terms_fast(target, 1, 10000000)
    if sol is not None:
        a, b = sol
        if sum(Fraction(1, x) for x in sol) == target:
            return list(sol)
    
    # Check 3-term: iterate M from 7 upward, but use the fast method
    # For a given M, check if there's a 3-term solution with max denominator <= M
    # We need to be smarter: iterate M from 7 to some limit
    
    best_solution = None
    
    # For 3-term solutions, the largest denominator c must be at least 7
    # Let's search for the minimal M such that a 3-term solution exists with max <= M
    
    # Strategy: iterate c (the largest denominator) from 7 upward
    # For each c, check if 1/a + 1/b = 5/31 - 1/c has a solution with 1 <= a < b < c
    # The first c for which this works gives us the minimal largest denominator
    
    for c in range(7, 2000):
        if time.time() > deadline:
            break
        
        rem2 = target - Fraction(1, c)
        if rem2 <= 0:
            continue
        
        # Find a, b with 1/a + 1/b = rem2, 1 <= a < b < c
        pair = find_two_terms_fast(rem2, 1, c - 1)
        if pair is not None:
            a, b = pair
            if a < b < c:
                sol = [a, b, c]
                if sum(Fraction(1, x) for x in sol) == target:
                    return sol
    
    # Check 4-term solutions with small max denominator
    def find_four_terms_fast(rem, min_d, max_d):
        """Find distinct a, b, c, d in [min_d, max_d] with sum of reciprocals = rem.
        Fix d (largest), then find a, b, c with 1/a + 1/b + 1/c = rem - 1/d.
        """
        for d in range(max_d, min_d + 1, -1):
            if time.time() > deadline:
                return None
            rem2 = rem - Fraction(1, d)
            if rem2 <= 0:
                break
            
            # Find a, b, c with 1/a + 1/b + 1/c = rem2, min_d <= a < b < c < d
            # Fix c, then find a, b
            for c in range(d - 1, min_d + 1, -1):
                if time.time() > deadline:
                    return None
                rem3 = rem2 - Fraction(1, c)
                if rem3 <= 0:
                    continue
                
                pair = find_two_terms_fast(rem3, min_d, c - 1)
                if pair is not None:
                    a, b = pair
                    if a < b < c < d:
                        return (a, b, c, d)
        
        return None
    
    # Check 4-term solutions
    for M in range(7, 100):
        if time.time() > deadline:
            break
        sol = find_four_terms_fast(target, 1, M)
        if sol is not None:
            if sum(Fraction(1, x) for x in sol) == target:
                return list(sol)
    
    # Check 5-term solutions with small max denominator
    def find_five_terms_fast(rem, min_d, max_d):
        """Find distinct a, b, c, d, e in [min_d, max_d] with sum of reciprocals = rem."""
        for e in range(max_d, min_d + 1, -1):
            if time.time() > deadline:
                return None
            rem2 = rem - Fraction(1, e)
            if rem2 <= 0:
                break
            
            # Find a, b, c, d with 1/a + 1/b + 1/c + 1/d = rem2, min_d <= a < b < c < d < e
            sol4 = find_four_terms_fast(rem2, min_d, e - 1)
            if sol4 is not None:
                a, b, c, d = sol4
                if a < b < c < d < e:
                    return (a, b, c, d, e)
        
        return None
    
    for M in range(7, 50):
        if time.time() > deadline:
            break
        sol = find_five_terms_fast(target, 1, M)
        if sol is not None:
            if sum(Fraction(1, x) for x in sol) == target:
                return list(sol)
    
    return fallback
