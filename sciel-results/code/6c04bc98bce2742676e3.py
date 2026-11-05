def construct():
    from fractions import Fraction
    target = Fraction(5, 31)
    best = None
    best_max = float('inf')
    
    # Search for 3-term solutions: 1/a + 1/b + 1/c = 5/31
    # Assume a <= b <= c
    # Bounds for a:
    # 3/a >= 5/31 => a <= 18
    # 1/a < 5/31 => a > 6 => a >= 7
    for a in range(7, 19):
        N = 5 * a - 31
        D = 31 * a
        if N <= 0:
            continue
        
        # 1/b + 1/c = N/D
        # b > D/N and b <= 2D/N
        b_min = D // N + 1
        b_max = (2 * D) // N
        
        # Enforce a <= b
        if b_min < a:
            b_min = a
            
        for b in range(b_min, b_max + 1):
            # 1/c = (Nb - D) / (Db)
            num_c = N * b - D
            den_c = D * b
            if num_c <= 0:
                continue
            if den_c % num_c != 0:
                continue
            c = den_c // num_c
            if c < b:
                continue
            if c > 10000000:
                continue
            
            # Verify
            s = Fraction(1, a) + Fraction(1, b) + Fraction(1, c)
            if s == target:
                m = max(a, b, c)
                if m < best_max:
                    best_max = m
                    best = [a, b, c]
                    
    if best is not None:
        return best
        
    # Fallback: use the original greedy approach if no 3-term solution found
    r, values = Fraction(5, 31) - Fraction(1, 8), [8]
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
