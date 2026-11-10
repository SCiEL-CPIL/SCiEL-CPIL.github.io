def construct():
    from fractions import Fraction
    target = Fraction(5, 31)
    best = None
    best_max = float('inf')
    
    # Try different first denominators
    for d1 in range(7, 500):
        r = target - Fraction(1, d1)
        if r <= 0:
            continue
        if r == 0:
            vals = [d1]
            mx = d1
        else:
            # Greedy completion
            vals = [d1]
            rem = r
            mx = d1
            valid = True
            for _ in range(255):
                if rem == 0:
                    break
                d = (rem.denominator + rem.numerator - 1) // rem.numerator
                if d > 10000000 or d in vals:
                    valid = False
                    break
                vals.append(d)
                mx = max(mx, d)
                rem -= Fraction(1, d)
                if rem < 0:
                    valid = False
                    break
            if not valid or rem != 0:
                continue
        if mx < best_max:
            best_max = mx
            best = vals[:]
    
    if best is None:
        # Fallback to original
        r, values = target - Fraction(1, 8), [8]
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
    return best
