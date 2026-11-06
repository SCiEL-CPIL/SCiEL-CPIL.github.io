"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from fractions import Fraction

    deadline = time.time() + 80.0
    target = Fraction(5, 31)

    # Try 2-term: 1/a + 1/b = 5/31
    # 31(a+b) = 5ab => (5a-31)(5b-31) = 961
    # 961 = 31^2, factors: 1, 31, 961
    # None give integer a, so no 2-term solution.

    # Exhaustive 3-term search: 1/a + 1/b + 1/c = 5/31, a < b < c
    # a >= 7 (since 3/a > 5/31 => a < 18.6, and a >= ceil(31/5)=7)
    # For each a, r = 5/31 - 1/a, then solve 1/b + 1/c = r
    best_max = float('inf')
    best_sol = None

    for a in range(7, 19):
        r = target - Fraction(1, a)
        if r <= 0:
            continue
        n, d = r.numerator, r.denominator
        # (nb-d)(nc-d) = d^2
        # X = nb - d, Y = nc - d, XY = d^2
        # b = (X+d)/n, c = (Y+d)/n
        # To minimize c = (d^2/X + d)/n, maximize X
        d2 = d * d
        # Find largest divisor X of d2 such that (X+d)%n==0 and (d^2/X + d)%n==0
        # and b > a, c > b
        for X in range(int(d2**0.5), 0, -1):
            if d2 % X != 0:
                continue
            Y = d2 // X
            if (X + d) % n != 0 or (Y + d) % n != 0:
                continue
            b = (X + d) // n
            c = (Y + d) // n
            if b > a and c > b:
                if c < best_max:
                    best_max = c
                    best_sol = [a, b, c]
                break  # largest X gives smallest c for this a

    if best_sol:
        return best_sol

    # Fallback: 4-term search with bounded c, d
    # 1/a + 1/b + 1/c + 1/d = 5/31
    # a in [7,18], b > a, c > b, d > c
    # For fixed a,b,c, d = 1/(5/31 - 1/a - 1/b - 1/c)
    # We bound d to keep search tractable
    best_max4 = float('inf')
    best_sol4 = None

    for a in range(7, 19):
        r1 = target - Fraction(1, a)
        if r1 <= 0:
            continue
        # b > a, and 2/b > r1 => b < 2/r1
        b_max = int(2 / r1) + 1
        for b in range(a + 1, min(b_max, 100)):
            r2 = r1 - Fraction(1, b)
            if r2 <= 0:
                continue
            c_max = int(2 / r2) + 1
            for c in range(b + 1, min(c_max, 200)):
                r3 = r2 - Fraction(1, c)
                if r3 <= 0:
                    continue
                d_val = 1 / r3
                if d_val.is_integer() and int(d_val) > c:
                    d_int = int(d_val)
                    if d_int < best_max4:
                        best_max4 = d_int
                        best_sol4 = [a, b, c, d_int]
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    if best_sol4:
        return best_sol4

    # Last resort
    return [7, 62, 434]
# EVOLVE-BLOCK-END
