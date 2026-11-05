"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 55.0
    target = Fraction(5, 31)
    
    # Start with known baseline
    baseline = [8, 31, 248]
    best_max = 248
    best_solution = baseline
    
    def verify(sol):
        if len(set(sol)) != len(sol):
            return False
        s = sum(Fraction(1, x) for x in sol)
        return s == target and all(1 <= x <= 10000000 for x in sol)
    
    # Verify baseline
    if verify(best_solution):
        pass
    else:
        # Fallback if baseline somehow invalid (shouldn't happen)
        return [8, 31, 248]

    # Strategy: Direct modular arithmetic search for 3-term and 4-term solutions
    # For 3 terms: 5/31 = 1/a + 1/b + 1/c
    # => 1/c = 5/31 - 1/a - 1/b = (5ab - 31a - 31b) / (31ab)
    # => c = 31ab / (5ab - 31a - 31b)
    # For c to be integer: (5ab - 31a - 31b) | 31ab
    
    # For 4 terms: 5/31 = 1/a + 1/b + 1/c + 1/d
    # We search for small a, b, c and compute d directly
    
    # Phase 1: Search for 3-term solutions with max denominator < 248
    # a < b < c, all < 248
    # 1/a < 5/31 => a >= 7
    # 1/a + 1/b < 5/31 (since 1/c > 0)
    # So 1/b < 5/31 - 1/a
    # b > a, and 1/b < 5/31 - 1/a
    
    for a in range(7, 100):
        if time.time() > deadline:
            break
        if a >= best_max:
            break
        # 1/a < 5/31 required
        if Fraction(1, a) >= target:
            continue
        
        rem1 = target - Fraction(1, a)
        # 1/b < rem1 => b > 1/rem1
        # b > a
        min_b = max(a + 1, int(1 / float(rem1)) + 1)
        # b < best_max
        for b in range(min_b, best_max):
            if time.time() > deadline:
                break
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            # c = 1/rem2
            # c must be integer, c > b, c < best_max
            if rem2.numerator != 1:
                # rem2 = p/q, c = q/p, integer only if p|q
                if rem2.denominator % rem2.numerator != 0:
                    continue
                c = rem2.denominator // rem2.numerator
            else:
                c = rem2.denominator
            
            if c <= b:
                continue
            if c >= best_max:
                continue
            
            sol = [a, b, c]
            if verify(sol):
                best_max = c
                best_solution = sol
                if best_max < 100:
                    return best_solution

    # Phase 2: Search for 4-term solutions with max denominator < best_max
    # 5/31 = 1/a + 1/b + 1/c + 1/d, a < b < c < d < best_max
    # 1/d = 5/31 - 1/a - 1/b - 1/c
    # d must be integer, d > c
    
    for a in range(7, min(50, best_max)):
        if time.time() > deadline:
            break
        if Fraction(1, a) >= target:
            continue
        rem1 = target - Fraction(1, a)
        
        min_b = max(a + 1, int(1 / float(rem1)) + 1)
        for b in range(min_b, min(80, best_max)):
            if time.time() > deadline:
                break
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            
            min_c = max(b + 1, int(1 / float(rem2)) + 1)
            for c in range(min_c, min(150, best_max)):
                if time.time() > deadline:
                    break
                rem3 = rem2 - Fraction(1, c)
                if rem3 <= 0:
                    continue
                
                # d = 1/rem3
                if rem3.numerator != 1:
                    if rem3.denominator % rem3.numerator != 0:
                        continue
                    d = rem3.denominator // rem3.numerator
                else:
                    d = rem3.denominator
                
                if d <= c:
                    continue
                if d >= best_max:
                    continue
                
                sol = [a, b, c, d]
                if verify(sol):
                    best_max = d
                    best_solution = sol
                    if best_max < 100:
                        return best_solution

    # Phase 3: If we found a better 3 or 4 term solution, try to find even better
    # by doing a more exhaustive search with tighter bounds
    # Let's also try 5-term solutions if best_max is still high
    if best_max > 150:
        # Try 5-term: 1/a+1/b+1/c+1/d+1/e
        # This is more complex; skip for now unless we have time
        pass

    return best_solution
# EVOLVE-BLOCK-END
