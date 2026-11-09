def construct():
    import time
    from math import gcd
    from fractions import Fraction
    
    deadline = time.time() + 85  # safety margin before 90s kill
    
    def two_term_solve(a, b):
        """Solve 1/x + 1/y = a/b, return list of (x,y) with x<=y, x!=y, x,y>=1."""
        # (x-b)(y-b) = b^2, so enumerate divisors of b^2
        b2 = b * b
        results = []
        i = 1
        while i * i <= b2:
            if b2 % i == 0:
                j = b2 // i
                x = i + b
                y = j + b
                if x >= 1 and y >= 1 and x != y:
                    results.append((min(x,y), max(x,y)))
            i += 1
        return results
    
    # 2-term: 1/x + 1/y = 5/31
    # (x-31)(y-31) = 31^2 = 961
    best_max = 10000001
    best_vals = None
    
    # Try 2-term
    pairs = two_term_solve(5, 31)
    for x, y in pairs:
        mx = max(x, y)
        if mx < best_max:
            best_max = mx
            best_vals = [x, y]
    
    # Try 3-term: 1/d + 1/x + 1/y = 5/31
    # r = 5/31 - 1/d = (5d - 31)/(31d)
    # Need 5d - 31 > 0, so d >= 7
    # Then solve 1/x + 1/y = (5d-31)/(31d)
    # Reduce: let g = gcd(5d-31, 31d), a = (5d-31)/g, b = 31d/g
    # (x-b)(y-b) = b^2
    for d in range(7, 2000):
        if time.time() > deadline:
            break
        num = 5 * d - 31
        den = 31 * d
        if num <= 0:
            continue
        g = gcd(num, den)
        a = num // g
        b = den // g
        # Solve 1/x + 1/y = a/b
        b2 = b * b
        i = 1
        while i * i <= b2:
            if b2 % i == 0:
                j = b2 // i
                x = i + b
                y = j + b
                if x >= 1 and y >= 1 and x != y and x != d and y != d:
                    mx = max(d, x, y)
                    if mx < best_max:
                        best_max = mx
                        best_vals = sorted([d, x, y])
            i += 1
    
    # Try 4-term: 1/d1 + 1/d2 + 1/x + 1/y = 5/31
    # Pick d1, d2, then solve 2-term for remainder
    # r = 5/31 - 1/d1 - 1/d2
    # This is more expensive; try small d1, d2
    if time.time() < deadline - 5:
        for d1 in range(7, 100):
            if time.time() > deadline:
                break
            for d2 in range(d1 + 1, 500):
                if time.time() > deadline:
                    break
                # r = 5/31 - 1/d1 - 1/d2
                num = 5 * d1 * d2 - 31 * d2 - 31 * d1
                den = 31 * d1 * d2
                if num <= 0:
                    continue
                g = gcd(num, den)
                a = num // g
                b = den // g
                b2 = b * b
                i = 1
                while i * i <= b2:
                    if b2 % i == 0:
                        j = b2 // i
                        x = i + b
                        y = j + b
                        if x >= 1 and y >= 1 and x != y and x != d1 and y != d1 and x != d2 and y != d2:
                            mx = max(d1, d2, x, y)
                            if mx < best_max:
                                best_max = mx
                                best_vals = sorted([d1, d2, x, y])
                    i += 1
    
    # Try 5-term: 1/d1 + 1/d2 + 1/d3 + 1/x + 1/y = 5/31
    if time.time() < deadline - 3:
        for d1 in range(7, 50):
            if time.time() > deadline:
                break
            for d2 in range(d1 + 1, 100):
                if time.time() > deadline:
                    break
                for d3 in range(d2 + 1, 200):
                    if time.time() > deadline:
                        break
                    num = 5 * d1 * d2 * d3 - 31 * (d2 * d3 + d1 * d3 + d1 * d2)
                    den = 31 * d1 * d2 * d3
                    if num <= 0:
                        continue
                    g = gcd(num, den)
                    a = num // g
                    b = den // g
                    b2 = b * b
                    i = 1
                    while i * i <= b2:
                        if b2 % i == 0:
                            j = b2 // i
                            x = i + b
                            y = j + b
                            if x >= 1 and y >= 1 and x != y:
                                if x != d1 and x != d2 and x != d3 and y != d1 and y != d2 and y != d3:
                                    mx = max(d1, d2, d3, x, y)
                                    if mx < best_max:
                                        best_max = mx
                                        best_vals = sorted([d1, d2, d3, x, y])
                        i += 1
    
    if best_vals is None:
        # Fallback: greedy
        from fractions import Fraction
        r = Fraction(5, 31)
        values = []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
    
    return best_vals
