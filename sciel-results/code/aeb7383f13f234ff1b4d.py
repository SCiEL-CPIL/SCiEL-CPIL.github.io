def construct():
    from fractions import Fraction
    target = Fraction(5, 31)
    best = None
    best_max = float('inf')
    
    # Upper bound for a. We know a solution with max 1736 exists.
    # If a > 1736, then b > a > 1736 and c > b > 1736, so max > 1736.
    # So we only need to search a <= 1736.
    upper_bound_a = 1736
    
    for a in range(1, upper_bound_a + 1):
        N_a = 5 * a - 31
        if N_a <= 0:
            continue
        D_a = 31 * a
        D_a_sq = D_a * D_a
        
        # Iterate potential divisors X of D_a^2 such that X < D_a
        # X = N_a * b - D_a
        # We need X to be a divisor of D_a^2
        # And X == -D_a (mod N_a)
        
        # Optimization: 
        # X must be positive.
        # b > a => N_a * b - D_a > N_a * a - D_a = 5a^2 - 31a - 31a = 5a^2 - 62a.
        # Let min_X = max(1, 5*a*a - 62*a)
        # Also X < D_a
        
        min_X = 5 * a * a - 62 * a
        if min_X < 1:
            min_X = 1
            
        # We iterate i from min_X to D_a - 1
        # But we only care if i divides D_a_sq
        # Iterating all integers is slow if range is large.
        # Range size approx D_a - min_X.
        # For a=100, D_a=3100, min_X=50000-6200=43800. min_X > D_a.
        # So for large a, min_X > D_a, no solutions?
        # Let's check.
        # If min_X >= D_a, then no X in [min_X, D_a) exists.
        # So we can break early if min_X >= D_a.
        
        if min_X >= D_a:
            # As a increases, min_X grows as 5a^2, D_a as 31a.
            # 5a^2 > 31a => a > 6.2.
            # So for a >= 7, min_X might be >= D_a?
            # a=7: min_X = 5*49 - 62*7 = 245 - 434 = -189. min_X=1. D_a=217.
            # a=10: min_X = 500 - 620 = -120. min_X=1. D_a=310.
            # a=20: min_X = 2000 - 1240 = 760. D_a=620. min_X > D_a.
            # So for a >= 20, no solutions with b > a?
            # Let's verify.
            # If b > a, then 1/b < 1/a.
            # 1/b + 1/c = 5/31 - 1/a.
            # If a is large, RHS is close to 5/31.
            # Max value of 1/b + 1/c given b > a is 1/(a+1) + 1/(a+2) approx 2/a.
            # We need 2/a >= 5/31 - 1/a = (5a-31)/31a.
            # 2/a >= (5a-31)/31a => 62 >= 5a - 31 => 93 >= 5a => a <= 18.6.
            # So indeed, for a >= 19, it's impossible to have b > a such that the sum reaches 5/31?
            # Wait, 1/b + 1/c < 2/b < 2/a.
            # We need 1/b + 1/c = (5a-31)/31a.
            # So we need (5a-31)/31a < 2/a => 5a-31 < 62 => 5a < 93 => a <= 18.
            # So we only need to check a up to 18!
            
            continue

        # So a is in [7, 18]
        # This is extremely fast.
        
        for X in range(min_X, D_a):
            if D_a_sq % X != 0:
                continue
            
            # Check X == -D_a (mod N_a)
            if (X + D_a) % N_a != 0:
                continue
            
            b = (X + D_a) // N_a
            # b must be integer, which is guaranteed by mod check
            # b must be > a
            if b <= a:
                continue
            
            Y = D_a_sq // X
            c = (Y + D_a) // N_a
            # c must be integer
            # Check c > b
            if c <= b:
                continue
            
            if c > 10000000:
                continue
            
            # Verify
            if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) != target:
                continue
                
            m = c # since a < b < c
            if m < best_max:
                best_max = m
                best = [a, b, c]
                
    if best is not None:
        return best
        
    # Fallback
    r, values = Fraction(5, 31) - Fraction(1, 8), [8]
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
