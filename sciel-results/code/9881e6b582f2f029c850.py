"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import math
    
    target = Fraction(5, 31)
    
    # We want to find distinct positive integers a < b < c such that
    # 1/a + 1/b + 1/c = 5/31
    # and minimize c (the largest denominator).
    
    # Strategy: iterate c from smallest possible value upward.
    # For each c, we need 1/a + 1/b = 5/31 - 1/c = (5c - 31)/(31c)
    # Let R = (5c - 31)/(31c). We need 1/a + 1/b = R.
    # This gives (5c-31)(ab) = 31c(a+b)
    # Let N = 5c - 31, D = 31c.
    # N*ab = D*(a+b)
    # N*ab - D*a - D*b = 0
    # (Na - D)(Nb - D) = D^2
    # So we need divisors of D^2 = (31c)^2.
    # Let X = Na - D, Y = Nb - D, with X*Y = D^2, X < Y (since a < b).
    # a = (X + D)/N, b = (Y + D)/N.
    # We need a, b to be positive integers, a < b < c, and a, b, c distinct.
    # Also a >= 1, b >= a+1, c >= b+1.
    
    # Lower bound for c: since 1/a + 1/b + 1/c = 5/31 and a < b < c,
    # 3/c < 5/31, so c > 93/5 = 18.6, so c >= 19.
    # Also 1/a + 1/b + 1/c <= 1/1 + 1/2 + 1/3 = 11/6 > 5/31, no issue.
    # More tightly: 1/a + 1/b + 1/c >= 3/c (since a,b,c <= c, but that's wrong direction).
    # Actually 1/a >= 1/c, 1/b >= 1/c, so 5/31 >= 3/c, c >= 93/5 = 18.6, c >= 19.
    # But we also need a < b < c, so a <= c-2, b <= c-1.
    # 1/a + 1/b + 1/c >= 1/(c-2) + 1/(c-1) + 1/c.
    # For small c this is large, so we need 1/(c-2) + 1/(c-1) + 1/c <= 5/31.
    # Let's just iterate c from 19 upward and check.
    
    # To minimize c, iterate c from 19 upward, and for each c check if a valid (a,b) exists.
    # Return the first valid solution found (smallest c).
    
    for c in range(19, 10000001):
        N = 5 * c - 31
        D = 31 * c
        
        if N <= 0:
            continue
            
        D_sq = D * D
        
        # Find divisors of D_sq that are less than D (since X < D for a < b < c constraint... 
        # actually X = Na - D, and a < c means Na < Nc = 5c^2 - 31c, so X < 5c^2 - 31c - 31c = 5c^2 - 62c.
        # But more simply, since a < b, we have X < Y, so X < D (since X*Y = D^2, X < D < Y).
        
        # We need to find divisors X of D_sq such that X < D, (X+D) % N == 0, 
        # and a = (X+D)//N satisfies 1 <= a < b < c where b = (D_sq//X + D)//N.
        
        # Iterate over divisors of D_sq up to sqrt(D_sq)
        limit = math.isqrt(D_sq)
        found = False
        
        for X in range(1, limit + 1):
            if D_sq % X != 0:
                continue
            Y = D_sq // X
            # We want X < Y, which is guaranteed since X <= sqrt(D_sq)
            # But we also need a < b, which means X < Y, so this is fine.
            # Check a = (X + D) / N is a positive integer
            if (X + D) % N != 0:
                continue
            a = (X + D) // N
            if a <= 0:
                continue
            # Check b = (Y + D) / N is a positive integer
            if (Y + D) % N != 0:
                continue
            b = (Y + D) // N
            if b <= 0:
                continue
            # Check ordering: a < b < c
            if a >= b or b >= c:
                continue
            # Check distinctness (already guaranteed by a < b < c)
            # Verify the sum
            if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                return [a, b, c]
    
    # Fallback (should not be reached)
    return [8, 31, 31]
# EVOLVE-BLOCK-END
