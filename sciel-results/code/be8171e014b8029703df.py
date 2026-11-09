"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import math
    
    target = Fraction(5, 31)
    
    # We want to find distinct denominators a, b, c with a < b < c
    # such that 1/a + 1/b + 1/c = 5/31, minimizing c (the largest denominator).
    
    # For each candidate a, we solve for b, c:
    # 1/b + 1/c = 5/31 - 1/a = r
    # Let r = N/D in lowest terms.
    # Then (Nb - D)(Nc - D) = D^2
    # Let X = Nb - D, Y = Nc - D. Then X*Y = D^2.
    # b = (X + D) / N, c = (Y + D) / N.
    # Since b < c, we have X < Y, so X < D.
    # We need X to be a divisor of D^2 such that:
    # 1. (X + D) % N == 0
    # 2. (Y + D) % N == 0 where Y = D^2 // X
    # 3. b > a and c > b
    
    # a must satisfy 1/a < 5/31, so a > 31/5 = 6.2, thus a >= 7.
    # Also, since we want to minimize the largest denominator c, and c > b > a,
    # we should search for small a first, but also check if a larger a could
    # yield a smaller c. However, typically smaller a leads to larger r, 
    # which might lead to different b,c.
    
    # Let's iterate a from 7 upwards.
    # For each a, find the minimal c such that there exists b with a < b < c
    # and 1/a + 1/b + 1/c = 5/31.
    
    best_max_denom = float('inf')
    best_solution = None
    
    # Upper bound for a: Since 1/a + 1/b + 1/c = 5/31 and a < b < c,
    # 3/a > 5/31 => a < 93/5 = 18.6, so a <= 18.
    # Wait, that's not right. 1/a < 5/31 => a > 31/5 = 6.2.
    # Also 1/a + 1/b + 1/c < 3/a (since b > a, c > a) => 5/31 < 3/a => a < 93/5 = 18.6.
    # So a can be at most 18? No, that's if all were equal to a.
    # Actually, 1/a + 1/b + 1/c = 5/31. Since b > a, 1/b < 1/a.
    # So 5/31 < 1/a + 1/a + 1/a = 3/a => a < 93/5 = 18.6.
    # So a is in [7, 18].
    
    for a in range(7, 19):
        r = target - Fraction(1, a)
        if r <= 0:
            continue
            
        N = r.numerator
        D = r.denominator
        
        D_sq = D * D
        
        # Find divisors X of D^2 such that X < D
        # and (X + D) % N == 0 and (D_sq//X + D) % N == 0
        # and resulting b > a, c > b
        
        # To minimize c, we want to maximize b (since c = D^2/(N*b - D) ... wait)
        # Actually, from 1/b + 1/c = r, we have c = 1 / (r - 1/b).
        # For fixed a, r is fixed. As b increases towards 1/r, c increases.
        # As b decreases towards a, c decreases.
        # But b must be > a.
        # Let's just enumerate all valid (b, c) pairs for this a and pick the one with min c.
        
        min_c_for_a = float('inf')
        best_b_for_a = None
        best_c_for_a = None
        
        # Iterate through possible b values? Or divisors?
        # Using divisors is more efficient if D^2 is not too large.
        # D is the denominator of 5/31 - 1/a = (5a - 31) / (31a).
        # So D = 31a / gcd(5a-31, 31a).
        
        # Let's iterate divisors of D^2.
        # D^2 can be up to (31*18)^2 = 558^2 ~ 311364. That's small.
        
        for X in range(1, D):
            if D_sq % X != 0:
                continue
            
            if (X + D) % N != 0:
                continue
            
            b = (X + D) // N
            if b <= a:
                continue
            
            Y = D_sq // X
            if (Y + D) % N != 0:
                continue
            
            c = (Y + D) // N
            if c <= b:
                continue
            
            # We have a valid solution (a, b, c)
            if c < min_c_for_a:
                min_c_for_a = c
                best_b_for_a = b
                best_c_for_a = c
        
        if best_b_for_a is not None:
            if min_c_for_a < best_max_denom:
                best_max_denom = min_c_for_a
                best_solution = [a, best_b_for_a, best_c_for_a]
    
    if best_solution:
        return best_solution
    else:
        # Fallback (should not happen)
        r, values = Fraction(5, 31) - Fraction(1, 8), [8]
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
# EVOLVE-BLOCK-END
