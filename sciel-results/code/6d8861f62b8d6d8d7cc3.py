"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import math

def construct():
    deadline = time.time() + 55.0  # safety margin before 60s kill
    target = Fraction(5, 31)
    best_solution = None
    best_max_den = float('inf')
    
    # 1. Find best 3-term solution using factorization of (5a-31)(5b-31) = 1116
    # 5/31 = 1/a + 1/b + 1/c
    # 1/a + 1/b = 5/31 - 1/c = (5c-31)/(31c)
    # (a+b)/(ab) = (5c-31)/(31c)
    # 31c(a+b) = ab(5c-31)
    # This path is complex. Use the derived condition:
    # If 5/31 = 1/a + 1/b + 1/c, then for fixed a,b, c is determined.
    # Alternatively, solve for 2-term expansion of remainder.
    # We found analytically: (8, 31, 248) and (7, 62, 434).
    # Let's verify and add them.
    
    candidate_3_terms = [
        [8, 31, 248],
        [7, 62, 434]
    ]
    
    for sol in candidate_3_terms:
        if len(set(sol)) == len(sol): # distinct
            s = sum(Fraction(1, x) for x in sol)
            if s == target:
                mx = max(sol)
                if mx < best_max_den:
                    best_max_den = mx
                    best_solution = sol
    
    # 2. Search for 4-term solutions to potentially improve max denominator
    # 5/31 = 1/a + 1/b + 1/c + 1/d
    # Iterate a, b, c and solve for d.
    # 1/d = 5/31 - 1/a - 1/b - 1/c
    # d = 1 / (5/31 - 1/a - 1/b - 1/c)
    # We need d to be an integer and d > c (to keep order, though not strictly required for validity, 
    # but helps in search bounds and distinctness check).
    
    # Bounds for a: 1/a < 5/31 => a > 6.2 => a >= 7
    # To keep max denominator small, we shouldn't pick a too large? 
    # Actually, if a is large, 1/a is small, remainder is large, requiring larger b,c,d.
    # So a should be relatively small.
    # Let's try a up to 20.
    
    max_a = 20
    for a in range(7, max_a + 1):
        if time.time() > deadline:
            break
        rem1 = target - Fraction(1, a)
        if rem1 <= 0:
            continue
        
        # Bounds for b: 1/b < rem1 => b > 1/rem1
        # Also b >= a (wlog)
        min_b = max(a, (rem1.numerator + rem1.denominator - 1) // rem1.denominator)
        # Upper bound for b? 
        # If b is too large, rem2 = rem1 - 1/b is close to rem1.
        # Then we need 1/c + 1/d = rem2.
        # Max(c,d) will be at least 2/rem2.
        # We want max(c,d) < best_max_den.
        # So 2/rem2 < best_max_den => rem2 > 2/best_max_den.
        # rem1 - 1/b > 2/best_max_den
        # 1/b < rem1 - 2/best_max_den
        # b > 1 / (rem1 - 2/best_max_den)
        # This gives a lower bound.
        # We also need b not too large such that d becomes huge.
        # Let's just iterate b up to a reasonable limit, e.g., 200.
        max_b = 200
        
        for b in range(min_b, max_b + 1):
            if time.time() > deadline:
                break
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            
            # Bounds for c: 1/c < rem2 => c > 1/rem2
            # Also c >= b
            min_c = max(b, (rem2.numerator + rem2.denominator - 1) // rem2.denominator)
            # Upper bound for c?
            # Similarly, 1/d = rem2 - 1/c.
            # d = 1 / (rem2 - 1/c).
            # For d to be integer and d > c (wlog), we need rem2 - 1/c > 1/d_max?
            # Actually, just iterate c.
            max_c = 200
            
            for c in range(min_c, max_c + 1):
                rem3 = rem2 - Fraction(1, c)
                if rem3 <= 0:
                    continue
                
                # Calculate d
                # d = 1 / rem3
                # rem3 = n/denom. d = denom/n.
                # d must be integer.
                if rem3.denominator % rem3.numerator != 0:
                    continue
                
                d = rem3.denominator // rem3.numerator
                
                # Check validity
                if d <= c:
                    continue
                if d > 10000000:
                    continue
                
                sol = [a, b, c, d]
                if len(set(sol)) == 4:
                    # Verify sum
                    if sum(Fraction(1, x) for x in sol) == target:
                        mx = max(sol)
                        if mx < best_max_den:
                            best_max_den = mx
                            best_solution = sol
                            # If we found a very good one, we can stop
                            if best_max_den < 100:
                                return best_solution
    
    # 3. Fallback to greedy if nothing better found (shouldn't happen given 3-term solutions)
    if best_solution is None:
        r = target
        greedy_vals = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            greedy_vals.append(d)
            r -= Fraction(1, d)
        if r == 0 and len(greedy_vals) <= 256 and max(greedy_vals) <= 10000000:
            return greedy_vals
        # If greedy fails or is invalid, return the best 3-term one we found earlier?
        # best_solution should be set.
    
    return best_solution
# EVOLVE-BLOCK-END
