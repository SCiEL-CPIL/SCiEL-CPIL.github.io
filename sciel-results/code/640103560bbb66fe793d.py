"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 55.0
    target = Fraction(5, 31)
    best_solution = None
    best_max_den = float('inf')

    # Known 3-term solutions with max denominator 248
    # Verify them first as a baseline
    baseline_solutions = [
        [8, 31, 248],
        [7, 62, 434]
    ]
    for sol in baseline_solutions:
        if len(set(sol)) == len(sol):
            s = sum(Fraction(1, x) for x in sol)
            if s == target:
                mx = max(sol)
                if mx < best_max_den:
                    best_max_den = mx
                    best_solution = sol

    # Strategy: Search for 4-term solutions 5/31 = 1/a + 1/b + 1/c + 1/d
    # where a < b < c < d to enforce distinctness and ordering
    # We want to minimize d (the largest denominator)
    
    # For 4 terms with a < b < c < d:
    # 1/d = 5/31 - 1/a - 1/b - 1/c
    # d = 1 / (5/31 - 1/a - 1/b - 1/c)
    # For d to be an integer > c, we need:
    # 5/31 - 1/a - 1/b - 1/c = 1/d where d is integer
    
    # Key insight: For small max denominator, we need a, b, c to be small
    # Let's bound a: 1/a < 5/31 => a > 6.2 => a >= 7
    # For 4 terms, we can have a as small as 7
    
    # To minimize the largest denominator d:
    # 1/d = 5/31 - 1/a - 1/b - 1/c > 0
    # So 1/a + 1/b + 1/c < 5/31
    # And d = 1/(5/31 - 1/a - 1/b - 1/c)
    # For d to be small, 5/31 - 1/a - 1/b - 1/c should be large
    # This means 1/a + 1/b + 1/c should be as small as possible (but still less than 5/31)
    
    # Let's enumerate a, b, c with a < b < c and check if d is integer and d > c
    # We want to find solutions where d < 248 (to beat current best)
    
    max_a = 15
    for a in range(7, max_a + 1):
        if time.time() > deadline:
            break
        rem1 = target - Fraction(1, a)
        if rem1 <= 0:
            continue
        
        # b > a, and 1/b < rem1
        min_b = a + 1
        max_b = 50  # keep reasonable
        
        for b in range(min_b, max_b + 1):
            if time.time() > deadline:
                break
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            
            # c > b, and 1/c < rem2
            min_c = b + 1
            max_c = 100
            
            for c in range(min_c, max_c + 1):
                rem3 = rem2 - Fraction(1, c)
                if rem3 <= 0:
                    continue
                
                # d = 1 / rem3 = rem3.denominator / rem3.numerator
                # For d to be integer: rem3.numerator must divide rem3.denominator
                if rem3.denominator % rem3.numerator != 0:
                    continue
                
                d = rem3.denominator // rem3.numerator
                
                # Check d > c (ordering) and d is integer
                if d <= c:
                    continue
                if d > 10000000:
                    continue
                
                sol = [a, b, c, d]
                # Verify distinctness (guaranteed by a<b<c<d) and sum
                if sum(Fraction(1, x) for x in sol) == target:
                    mx = max(sol)
                    if mx < best_max_den:
                        best_max_den = mx
                        best_solution = sol
                        if best_max_den < 100:
                            return best_solution

    # Also search for 3-term solutions more systematically to see if any beat 248
    # 5/31 = 1/a + 1/b + 1/c with a < b < c
    # 1/c = 5/31 - 1/a - 1/b
    # c = 1/(5/31 - 1/a - 1/b)
    
    for a in range(7, 20):
        if time.time() > deadline:
            break
        rem1 = target - Fraction(1, a)
        if rem1 <= 0:
            continue
        
        min_b = a + 1
        for b in range(min_b, 100):
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            
            if rem2.denominator % rem2.numerator != 0:
                continue
            
            c = rem2.denominator // rem2.numerator
            if c <= b:
                continue
            if c > 10000000:
                continue
            
            sol = [a, b, c]
            if sum(Fraction(1, x) for x in sol) == target:
                mx = max(sol)
                if mx < best_max_den:
                    best_max_den = mx
                    best_solution = sol

    # Fallback
    if best_solution is None:
        # Use known solution
        return [8, 31, 248]
    
    return best_solution
# EVOLVE-BLOCK-END
