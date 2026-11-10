def construct():
    from fractions import Fraction
    
    target = Fraction(5, 31)
    
    # Try 3-term solutions
    best = None
    best_max = float('inf')
    
    for a in range(7, 500):
        for b in range(a + 1, 10000):
            c_den = 5 * a * b - 31 * a - 31 * b
            if c_den <= 0:
                continue
            if (31 * a * b) % c_den != 0:
                continue
            c = (31 * a * b) // c_den
            if c <= b:
                continue
            if c > 10000000:
                continue
            if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                if c < best_max:
                    best_max = c
                    best = [a, b, c]
    
    if best is not None:
        return best
    
    # Fallback
    r, values = target - Fraction(1, 8), [8]
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
