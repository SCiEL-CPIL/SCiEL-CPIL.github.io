def construct():
    import time
    start_time = time.time()
    DEADLINE = 80

    from math import isqrt, gcd
    from fractions import Fraction

    def divisors(n):
        """Return all divisors of n."""
        if n <= 0:
            return []
        small = []
        large = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                small.append(i)
                if i != n // i:
                    large.append(n // i)
            i += 1
        large.reverse()
        return small + large

    def find_3term(a, max_bound):
        """Find b,c such that 1/a+1/b+1/c=5/31 with b,c distinct from a and each other, b,c<=max_bound."""
        results = []
        num = 5 * a - 31
        if num <= 0:
            return results
        den = 31 * a
        g = gcd(num, den)
        n = num // g
        d = den // g
        d2 = d * d
        for u in divisors(d2):
            v = d2 // u
            bu = u + d
            bv = v + d
            if bu % n == 0 and bv % n == 0:
                b = bu // n
                c = bv // n
                if b <= 0 or c <= 0:
                    continue
                if b == a or c == a or b == c:
                    continue
                if b > max_bound or c > max_bound:
                    continue
                # Verify
                if 31 * b * c + 31 * a * c + 31 * a * b == 5 * a * b * c:
                    results.append((a, b, c))
        return results

    def find_4term(a, b, max_bound):
        """Find c,d such that 1/a+1/b+1/c+1/d=5/31 with c,d distinct from a,b and each other."""
        results = []
        # 1/c + 1/d = 5/31 - 1/a - 1/b = (5ab - 31b - 31a)/(31ab)
        num = 5 * a * b - 31 * b - 31 * a
        if num <= 0:
            return results
        den = 31 * a * b
        g = gcd(num, den)
        n = num // g
        d = den // g
        d2 = d * d
        for u in divisors(d2):
            v = d2 // u
            bu = u + d
            bv = v + d
            if bu % n == 0 and bv % n == 0:
                c = bu // n
                dd = bv // n
                if c <= 0 or dd <= 0:
                    continue
                if c == a or c == b or c == dd:
                    continue
                if dd == a or dd == b:
                    continue
                if c > max_bound or dd > max_bound:
                    continue
                # Verify
                lhs = 31 * b * c * dd + 31 * a * c * dd + 31 * a * b * dd + 31 * a * b * c
                rhs = 5 * a * b * c * dd
                if lhs == rhs:
                    results.append((a, b, c, dd))
        return results

    best_max_den = 10000001
    best_solution = None

    # Phase 1: 3-term search using divisor enumeration
    for a in range(7, 200):
        if time.time() - start_time > DEADLINE:
            break
        if a >= best_max_den:
            break
        sols = find_3term(a, best_max_den - 1)
        for sol in sols:
            m = max(sol)
            if m < best_max_den:
                best_max_den = m
                best_solution = list(sol)

    # Phase 2: 4-term search - enumerate pairs (a,b) and solve for (c,d)
    if best_solution is None or best_max_den > 20:
        for a in range(7, 50):
            if time.time() - start_time > DEADLINE:
                break
            if a >= best_max_den:
                break
            for b in range(a + 1, best_max_den):
                if time.time() - start_time > DEADLINE:
                    break
                sols = find_4term(a, b, best_max_den - 1)
                for sol in sols:
                    m = max(sol)
                    if m < best_max_den:
                        best_max_den = m
                        best_solution = list(sol)

    # Phase 3: Meet-in-the-middle for 4-term solutions
    # Precompute all sums of two distinct unit fractions 1/b + 1/c where b < c
    # Key by the rational value, store the pair (b, c)
    # Then for each a, check if 5/31 - 1/a exists in the map
    # This allows finding 4-term solutions more efficiently
    
    if best_solution is None or best_max_den > 15:
        # Build map of 1/b + 1/c for b < c, with b,c < best_max_den
        # To keep memory reasonable, limit b and c to a reasonable range
        # Since we're looking for max_den < best_max_den, let's limit to best_max_den - 1
        limit = min(best_max_den - 1, 500)  # Cap to avoid too much memory
        
        pair_map = {}
        for b in range(2, limit + 1):
            if time.time() - start_time > DEADLINE:
                break
            for c in range(b + 1, limit + 1):
                val = Fraction(1, b) + Fraction(1, c)
                # Store the pair; if multiple pairs give same val, keep the one with smallest max
                key = (val.numerator, val.denominator)
                if key not in pair_map or c < pair_map[key][1]:
                    pair_map[key] = (b, c)
        
        # Now for each a, check if 5/31 - 1/a is in pair_map
        for a in range(2, limit + 1):
            if time.time() - start_time > DEADLINE:
                break
            if a >= best_max_den:
                break
            target_val = Fraction(5, 31) - Fraction(1, a)
            if target_val <= 0:
                continue
            key = (target_val.numerator, target_val.denominator)
            if key in pair_map:
                b, c = pair_map[key]
                # Check distinctness
                if a != b and a != c and b != c:
                    sol = [a, b, c]
                    m = max(sol)
                    if m < best_max_den:
                        best_max_den = m
                        best_solution = sol

    if best_solution is not None:
        # Validate the solution
        if 0 < len(best_solution) <= 256:
            if all(1 <= d <= 10000000 for d in best_solution):
                if len(set(best_solution)) == len(best_solution):
                    s = sum(Fraction(1, d) for d in best_solution)
                    if s == Fraction(5, 31):
                        return best_solution

    # Fallback: Hardcoded valid solution for 5/31
    # 1/8 + 1/28 + 1/1736 = 5/31
    return [8, 28, 1736]
