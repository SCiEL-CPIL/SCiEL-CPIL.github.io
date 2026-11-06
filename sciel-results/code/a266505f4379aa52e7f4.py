def construct():
    from fractions import Fraction

    target = Fraction(5, 31)
    best_max = 10**9
    best_solution = None

    # Try 3-term expansions: 1/a + 1/b + 1/c = 5/31, a < b < c
    for a in range(1, 100):
        for b in range(a + 1, 2000):
            r = target - Fraction(1, a) - Fraction(1, b)
            if r <= 0:
                break
            if r.numerator == 1:
                c = r.denominator
                if c > b and c <= 10000000:
                    max_den = c
                    if max_den < best_max:
                        best_max = max_den
                        best_solution = [a, b, c]

    if best_solution is not None:
        return best_solution

    # Try 4-term expansions: 1/a + 1/b + 1/c + 1/d = 5/31, a < b < c < d
    for a in range(1, 50):
        for b in range(a + 1, 300):
            r_ab = target - Fraction(1, a) - Fraction(1, b)
            if r_ab <= 0:
                break
            for c in range(b + 1, 800):
                r = r_ab - Fraction(1, c)
                if r <= 0:
                    break
                if r.numerator == 1:
                    d = r.denominator
                    if d > c and d <= 10000000:
                        max_den = d
                        if max_den < best_max:
                            best_max = max_den
                            best_solution = [a, b, c, d]

    if best_solution is not None:
        return best_solution

    # Fallback to greedy
    r = target
    values = []
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
