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
    
    def update_best(sol):
        nonlocal best_solution, best_max_den
        if not sol:
            return
        # Check distinctness
        if len(set(sol)) != len(sol):
            return
        # Check bounds
        if any(d < 1 or d > 10000000 for d in sol):
            return
        # Check sum
        s = sum(Fraction(1, x) for x in sol)
        if s == target:
            mx = max(sol)
            if mx < best_max_den:
                best_max_den = mx
                best_solution = list(sol)

    def solve_2_terms(rem, min_den, existing):
        """
        Solve 1/c + 1/d = rem.
        Returns list of valid (c, d) pairs.
        """
        if rem <= 0:
            return []
        
        N = rem.numerator
        D = rem.denominator
        K = D * D
        
        solutions = []
        # Iterate divisors of K
        # K can be large, but D is at most 31 * a * b * c ...
        # For 4-term, rem is based on a,b. D is manageable.
        # For 5-term, rem is based on a,b,c.
        
        sqrt_k = int(math.isqrt(K))
        for x in range(1, sqrt_k + 1):
            if time.time() > deadline:
                break
            if K % x != 0:
                continue
            
            y = K // x
            
            # Check congruences: N*c - D = x => N*c = x + D => x + D must be divisible by N
            if (x + D) % N != 0:
                continue
            c = (x + D) // N
            
            if (y + D) % N != 0:
                continue
            d = (y + D) // N
            
            # Ensure c, d are positive integers
            if c <= 0 or d <= 0:
                continue
            
            # Ensure distinct and greater than min_den
            # And distinct from existing
            if c in existing or d in existing:
                continue
            if c == d:
                continue
            if c < min_den or d < min_den:
                continue
            
            # Sort to keep order, though not strictly required for set, 
            # but helps with min_den logic if we were doing DFS. 
            # Here we just return the pair.
            solutions.append((c, d))
            
            # Also check the swapped case if x != y, but the loop covers divisors.
            # If x < y, we get one pair. If x > y, we get the same pair swapped.
            # Since we iterate x up to sqrt, we only get x <= y.
            # c and d are derived from x and y.
            # If x < y, c < d.
            # We should also consider if x > y is possible? No, x is the smaller divisor.
            # But what if the valid solution has c > d?
            # The equation is symmetric. If (c,d) is a solution, (d,c) is too.
            # Our derivation maps x to c and y to d.
            # If we find a valid pair, it's valid.
            
        return solutions

    # 1. 2-term solutions
    # 5/31 = 1/a + 1/b
    sols_2 = solve_2_terms(target, 1, set())
    for c, d in sols_2:
        update_best([c, d])

    # 2. 3-term solutions
    # 5/31 = 1/a + 1/b + 1/c
    # a > 31/5 = 6.2 => a >= 7
    max_a_3 = 100
    for a in range(7, max_a_3 + 1):
        if time.time() > deadline:
            break
        rem1 = target - Fraction(1, a)
        if rem1 <= 0:
            continue
        
        sols = solve_2_terms(rem1, a + 1, {a})
        for b, c in sols:
            update_best([a, b, c])

    # 3. 4-term solutions
    # 5/31 = 1/a + 1/b + 1/c + 1/d
    # a >= 7
    max_a_4 = 50
    max_b_4 = 200
    for a in range(7, max_a_4 + 1):
        if time.time() > deadline:
            break
        rem1 = target - Fraction(1, a)
        if rem1 <= 0:
            continue
        
        # b must be > a
        # 1/b < rem1 => b > 1/rem1
        min_b = max(a + 1, (rem1.denominator + rem1.numerator - 1) // rem1.numerator)
        
        for b in range(min_b, max_b_4 + 1):
            if time.time() > deadline:
                break
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            
            sols = solve_2_terms(rem2, b + 1, {a, b})
            for c, d in sols:
                update_best([a, b, c, d])

    # 4. 5-term solutions
    # 5/31 = 1/a + 1/b + 1/c + 1/d + 1/e
    # a >= 7
    max_a_5 = 30
    max_b_5 = 100
    max_c_5 = 200
    for a in range(7, max_a_5 + 1):
        if time.time() > deadline:
            break
        rem1 = target - Fraction(1, a)
        if rem1 <= 0:
            continue
        
        min_b = max(a + 1, (rem1.denominator + rem1.numerator - 1) // rem1.numerator)
        
        for b in range(min_b, max_b_5 + 1):
            if time.time() > deadline:
                break
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            
            min_c = max(b + 1, (rem2.denominator + rem2.numerator - 1) // rem2.numerator)
            
            for c in range(min_c, max_c_5 + 1):
                if time.time() > deadline:
                    break
                rem3 = rem2 - Fraction(1, c)
                if rem3 <= 0:
                    continue
                
                sols = solve_2_terms(rem3, c + 1, {a, b, c})
                for d, e in sols:
                    update_best([a, b, c, d, e])

    # 5. 6-term solutions (just in case)
    max_a_6 = 20
    max_b_6 = 50
    max_c_6 = 100
    max_d_6 = 150
    for a in range(7, max_a_6 + 1):
        if time.time() > deadline:
            break
        rem1 = target - Fraction(1, a)
        if rem1 <= 0:
            continue
        
        min_b = max(a + 1, (rem1.denominator + rem1.numerator - 1) // rem1.numerator)
        for b in range(min_b, max_b_6 + 1):
            if time.time() > deadline:
                break
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            
            min_c = max(b + 1, (rem2.denominator + rem2.numerator - 1) // rem2.numerator)
            for c in range(min_c, max_c_6 + 1):
                if time.time() > deadline:
                    break
                rem3 = rem2 - Fraction(1, c)
                if rem3 <= 0:
                    continue
                
                min_d = max(c + 1, (rem3.denominator + rem3.numerator - 1) // rem3.numerator)
                for d in range(min_d, max_d_6 + 1):
                    if time.time() > deadline:
                        break
                    rem4 = rem3 - Fraction(1, d)
                    if rem4 <= 0:
                        continue
                    
                    sols = solve_2_terms(rem4, d + 1, {a, b, c, d})
                    for e, f in sols:
                        update_best([a, b, c, d, e, f])

    if best_solution is not None:
        return best_solution
    
    # Fallback: Greedy
    r = target
    greedy_vals = []
    while r > 0:
        d = (r.denominator + r.numerator - 1) // r.numerator
        greedy_vals.append(d)
        r -= Fraction(1, d)
    if r == 0 and len(greedy_vals) <= 256 and max(greedy_vals) <= 10000000:
        return greedy_vals
        
    # Absolute fallback
    return [8, 31, 248]
# EVOLVE-BLOCK-END
