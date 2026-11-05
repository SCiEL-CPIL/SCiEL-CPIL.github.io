"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from math import gcd

    start = time.time()
    deadline = 85.0

    def check_valid(denoms):
        """Verify the denominators sum reciprocally to 5/31."""
        if not denoms:
            return False
        if len(denoms) != len(set(denoms)):
            return False
        for d in denoms:
            if d < 1 or d > 10000000:
                return False
        # Compute sum using common denominator
        from math import prod
        # Use incremental LCM approach
        from math import gcd
        num = 0
        den = 1
        for d in denoms:
            g = gcd(den, d)
            lcm = den * d // g
            num = num * (lcm // den) + lcm // d
            den = lcm
        g = gcd(num, den)
        num //= g
        den //= g
        return num == 5 and den == 31

    best_solution = None
    best_max_den = float('inf')

    def update_best(denoms):
        nonlocal best_solution, best_max_den
        mx = max(denoms)
        if mx < best_max_den:
            best_max_den = mx
            best_solution = list(denoms)

    # Search for 3-term solutions: 1/a + 1/b + 1/c = 5/31
    # For fixed a, b: c = b*(5a-31)*31 / (b*(5a-31) - 31*a) must be positive integer
    # a must satisfy 5a > 31, so a >= 7
    # Also a < 31/5 = 6.2 for 1/a < 5/31, so a >= 7
    # For 3 terms with a <= b <= c, we need 3/a >= 5/31, so a <= 93/5 = 18.6, so a <= 18
    # And 1/a + 1/b >= 5/31 - 1/c, but for bounding, 2/b + 1/a >= 5/31 (since b<=c)
    
    for a in range(7, 19):
        if time.time() > deadline:
            break
        # 1/b + 1/c = 5/31 - 1/a = (5a-31)/(31a)
        num_bc = 5 * a - 31
        den_bc = 31 * a
        if num_bc <= 0:
            continue
        
        # c = b * den_bc / (b * num_bc - den_bc)
        # For c > 0: b * num_bc > den_bc, so b > den_bc / num_bc
        # For c >= b: b * den_bc >= b * (b * num_bc - den_bc)
        #   b * den_bc >= b^2 * num_bc - b * den_bc
        #   2 * b * den_bc >= b^2 * num_bc
        #   2 * den_bc >= b * num_bc
        #   b <= 2 * den_bc / num_bc
        
        min_b = den_bc // num_bc + 1
        max_b = min(2 * den_bc // num_bc, 10000000)
        
        # Also need b >= a for ordering
        min_b = max(min_b, a)
        
        for b in range(min_b, max_b + 1):
            if time.time() > deadline:
                break
            # c = b * den_bc / (b * num_bc - den_bc)
            denom_c = b * num_bc - den_bc
            if denom_c <= 0:
                continue
            num_c = b * den_bc
            if num_c % denom_c != 0:
                continue
            c = num_c // denom_c
            if c < b:
                continue
            if c > 10000000:
                continue
            # Check distinctness
            if a == b or b == c or a == c:
                continue
            denoms = [a, b, c]
            if check_valid(denoms):
                update_best(denoms)
                if best_max_den <= c:
                    break

    # If no 3-term solution found, try 4-term solutions
    if best_solution is None or best_max_den > 1000:
        # For 4 terms: 1/a + 1/b + 1/c + 1/d = 5/31
        # Try small a, b, then solve for c, d similarly
        for a in range(7, 15):
            if time.time() > deadline:
                break
            rem_a = (5 * a - 31) / (31 * a)  # as float for estimation
            if rem_a <= 0:
                continue
            # 1/b + 1/c + 1/d = (5a-31)/(31a)
            num_ab = 5 * a - 31
            den_ab = 31 * a
            if num_ab <= 0:
                continue
            for b in range(a + 1, 50):
                if time.time() > deadline:
                    break
                # 1/c + 1/d = (5a-31)/(31a) - 1/b = (b(5a-31) - 31a) / (31ab)
                num_cd = b * num_ab - den_ab
                den_cd = 31 * a * b
                if num_cd <= 0:
                    continue
                # d = c * den_cd / (c * num_cd - den_cd)
                min_c = den_cd // num_cd + 1
                max_c = min(2 * den_cd // num_cd, 10000000)
                min_c = max(min_c, b + 1)
                
                for c in range(min_c, min(max_c + 1, 10000)):
                    if time.time() > deadline:
                        break
                    denom_d = c * num_cd - den_cd
                    if denom_d <= 0:
                        continue
                    num_d = c * den_cd
                    if num_d % denom_d != 0:
                        continue
                    d = num_d // denom_d
                    if d < c:
                        continue
                    if d > 10000000:
                        continue
                    if len({a, b, c, d}) != 4:
                        continue
                    denoms = [a, b, c, d]
                    if check_valid(denoms):
                        update_best(denoms)
                        if best_max_den <= d:
                            break

    # Fallback: greedy completion
    if best_solution is None:
        from fractions import Fraction
        r = Fraction(5, 31)
        values = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d in values:
                d += 1
            values.append(d)
            r -= Fraction(1, d)
            if len(values) > 256:
                break
        best_solution = values

    return best_solution
# EVOLVE-BLOCK-END
