from fractions import Fraction
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 45
    
    target = Fraction(5, 31)
    best_max = float('inf')
    best_sol = None
    
    def find_two_term_solutions(n, d, min_val):
        """Find all pairs (b,c) with b <= c, b >= min_val, such that 1/b + 1/c = n/d.
        Uses factorization: (nb-d)(nc-d) = d^2
        """
        solutions = []
        d_sq = d * d
        
        # Enumerate divisors of d^2
        # For each divisor X of d^2 where X = nb - d, we get b = (X + d) / n
        # and Y = d^2 / X, c = (Y + d) / n
        # Need X > 0, Y > 0, b and c integers, b >= min_val, b <= c
        
        # Find all divisors of d^2
        sqrt_d_sq = int(math.isqrt(d_sq))
        
        for X in range(1, sqrt_d_sq + 1):
            if d_sq % X != 0:
                continue
            
            Y = d_sq // X
            
            # Case 1: X = nb - d, Y = nc - d
            if (X + d) % n == 0:
                b = (X + d) // n
                if (Y + d) % n == 0:
                    c = (Y + d) // n
                    if b >= min_val and b <= c:
                        solutions.append((b, c))
            
            # Case 2: X = nc - d, Y = nb - d (swap)
            if X != Y:
                if (Y + d) % n == 0:
                    b = (Y + d) // n
                    if (X + d) % n == 0:
                        c = (X + d) // n
                        if b >= min_val and b <= c:
                            solutions.append((b, c))
        
        return solutions
    
    # 3-term search: 1/a + 1/b + 1/c = 5/31
    # For each a, remainder R_a = 5/31 - 1/a
    # Then find (b,c) such that 1/b + 1/c = R_a with b > a
    
    for a in range(7, 100):
        if time.time() > deadline:
            break
        
        R_a = target - Fraction(1, a)
        if R_a <= 0:
            continue
        
        n_a, d_a = R_a.numerator, R_a.denominator
        
        # Find all (b,c) with b > a, b <= c
        solutions = find_two_term_solutions(n_a, d_a, a + 1)
        
        for b, c in solutions:
            if c < best_max:
                best_max = c
                best_sol = [a, b, c]
    
    # 4-term search: 1/a + 1/b + 1/c + 1/d = 5/31
    # For each (a,b), remainder R_ab = 5/31 - 1/a - 1/b
    # Then find (c,d) such that 1/c + 1/d = R_ab with c > b
    
    # Bound a: 1/a < 5/31 => a > 31/5 = 6.2, so a >= 7
    # For 4 terms, a should be small enough that remaining 3 terms can sum to R_a
    # 3 * (1/(a+1)) > R_a for feasibility
    
    a_max_4 = 200  # Conservative bound
    for a in range(7, a_max_4):
        if time.time() > deadline:
            break
        
        R_a = target - Fraction(1, a)
        if R_a <= 0:
            continue
        
        # b > a, and 1/b + 1/c + 1/d = R_a
        # b < 1/R_a * 3 (since 1/b < R_a/3 for the other two to be positive)
        # Actually b_min = floor(1/R_a) + 1, b_max = floor(3/R_a)
        
        n_a, d_a = R_a.numerator, R_a.denominator
        b_min = d_a // n_a + 1
        if b_min < a + 1:
            b_min = a + 1
        b_max = 3 * d_a // n_a
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            
            R_ab = R_a - Fraction(1, b)
            if R_ab <= 0:
                continue
            
            n_ab, d_ab = R_ab.numerator, R_ab.denominator
            
            # Find (c,d) with c > b
            solutions = find_two_term_solutions(n_ab, d_ab, b + 1)
            
            for c, d in solutions:
                if d < best_max:
                    best_max = d
                    best_sol = [a, b, c, d]
    
    if best_sol is None:
        return [8, 31, 248]
        
    return best_sol
