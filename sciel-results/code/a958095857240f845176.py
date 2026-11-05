"""
Find Egyptian fraction of 5/31 minimizing the largest denominator.
Uses divisor-based pair enumeration for efficiency.
"""
import time
from math import gcd
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin before 60s limit

    target = Fraction(5, 31)

    # Precompute a good initial solution via greedy
    def greedy(target):
        r = target
        denoms = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            denoms.append(d)
            r -= Fraction(1, d)
        return denoms

    best = greedy(target)
    best_max = max(best)

    def time_ok():
        return time.time() < deadline

    # For 1/x + 1/y = p/q (reduced), find all pairs x < y with y <= bound
    # (p*x - q)(p*y - q) = q^2
    # Let d = p*x - q, then d | q^2, x = (d+q)/p, y = (q^2/d + q)/p
    def find_pairs(p, q, x_min, x_max, bound):
        """Find pairs (x,y) with x_min <= x < y <= bound, x,y integers, 1/x+1/y = p/q"""
        if p <= 0 or q <= 0:
            return []
        g = gcd(p, q)
        p //= g
        q //= g

        q_sq = q * q
        pairs = []

        # x must satisfy: x_min <= x, and 1/x < p/q => x > q/p
        # Also 2/x > p/q => x < 2q/p
        # And y = (q^2/d + q)/p where d = p*x - q
        # d ranges over divisors of q^2 with d = p*x - q
        # x = (d + q) / p must be integer, x in [x_min, x_max]
        # y = (q^2/d + q) / p must be integer, y > x, y <= bound

        # d = p*x - q, so d >= p*x_min - q and d <= p*x_max - q
        # Also d > 0 (since x > q/p)
        d_min = max(1, p * x_min - q)
        d_max = p * x_max - q

        # Iterate divisors of q^2 in [d_min, d_max]
        # For each divisor d, check if (d+q) % p == 0 and (q^2//d + q) % p == 0

        # Efficient: iterate i up to sqrt(q_sq), collect divisors
        # But q_sq can be large. Alternative: iterate d from d_min to d_max checking divisibility
        # That's too slow. Instead, factor q and enumerate divisors.

        # Factor q
        def factorize(n):
            factors = {}
            d = 2
            while d * d <= n:
                while n % d == 0:
                    factors[d] = factors.get(d, 0) + 1
                    n //= d
                d += 1
            if n > 1:
                factors[n] = factors.get(n, 0) + 1
            return factors

        q_factors = factorize(q)
        # q^2 factors: each exponent doubled
        q2_factors = {p: e * 2 for p, e in q_factors.items()}

        # Generate all divisors of q^2
        def gen_divisors(factors):
            primes = list(factors.keys())
            exponents = [factors[p] for p in primes]
            divisors = [1]
            for i, p in enumerate(primes):
                new_divs = []
                for d in divisors:
                    for e in range(exponents[i] + 1):
                        new_divs.append(d * (p ** e))
                divisors = new_divs
            return sorted(divisors)

        divisors = gen_divisors(q2_factors)

        for d in divisors:
            if d < d_min or d > d_max:
                continue
            if (d + q) % p != 0:
                continue
            x = (d + q) // p
            if x < x_min or x > x_max:
                continue
            # y = (q^2 // d + q) / p
            if q_sq % d != 0:
                continue
            d2 = q_sq // d
            if (d2 + q) % p != 0:
                continue
            y = (d2 + q) // p
            if y <= x or y > bound:
                continue
            # Verify
            if Fraction(1, x) + Fraction(1, y) == Fraction(p, q):
                pairs.append((x, y))

        return pairs

    # Search for 3-term: 1/a + 1/b + 1/c = 5/31, a < b < c
    # a >= 7 (since 1/a < 5/31 => a > 6.2)
    # 3/a > 5/31 => a < 93/5 = 18.6, so a <= 18 for 3 terms
    # But with more terms a can be larger; for 3 terms specifically a in [7, 18]
    def search_3_terms(bound):
        solutions = []
        for a in range(7, 19):
            if not time_ok():
                break
            r = target - Fraction(1, a)
            if r <= 0:
                continue
            # 1/b + 1/c = r = p/q
            p, q = r.numerator, r.denominator
            # b > a, b < 2q/p (since 2/b > r)
            b_min = a + 1
            b_max = min(2 * q // p + 1, bound)
            pairs = find_pairs(p, q, b_min, b_max, bound)
            for b, c in pairs:
                solutions.append((a, b, c))
        return solutions

    # Search for 4-term: 1/a + 1/b + 1/c + 1/d = 5/31
    # a in [7, 24] (4/a > 5/31 => a < 24.8)
    def search_4_terms(bound):
        solutions = []
        for a in range(7, 25):
            if not time_ok():
                break
            r1 = target - Fraction(1, a)
            if r1 <= 0:
                continue
            # 1/b + 1/c + 1/d = r1
            # b > a, 3/b > r1 => b < 3/r1
            b_min = a + 1
            b_max = min(int(3 / r1) + 1, bound)
            for b in range(b_min, b_max + 1):
                if not time_ok():
                    break
                r2 = r1 - Fraction(1, b)
                if r2 <= 0:
                    continue
                # 1/c + 1/d = r2 = p/q
                p, q = r2.numerator, r2.denominator
                c_min = b + 1
                c_max = min(2 * q // p + 1, bound)
                pairs = find_pairs(p, q, c_min, c_max, bound)
                for c, d in pairs:
                    solutions.append((a, b, c, d))
        return solutions

    # Search for 5-term
    def search_5_terms(bound):
        solutions = []
        for a in range(7, 21):
            if not time_ok():
                break
            r1 = target - Fraction(1, a)
            if r1 <= 0:
                continue
            b_min = a + 1
            b_max = min(int(4 / r1) + 1, bound)
            for b in range(b_min, b_max + 1):
                if not time_ok():
                    break
                r2 = r1 - Fraction(1, b)
                if r2 <= 0:
                    continue
                c_min = b + 1
                c_max = min(int(3 / r2) + 1, bound)
                for c in range(c_min, c_max + 1):
                    if not time_ok():
                        break
                    r3 = r2 - Fraction(1, c)
                    if r3 <= 0:
                        continue
                    # 1/d + 1/e = r3 = p/q
                    p, q = r3.numerator, r3.denominator
                    d_min = c + 1
                    d_max = min(2 * q // p + 1, bound)
                    pairs = find_pairs(p, q, d_min, d_max, bound)
                    for d, e in pairs:
                        solutions.append((a, b, c, d, e))
        return solutions

    # Try increasing bounds
    bounds = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    for bound in bounds:
        if not time_ok():
            break

        # 3-term
        sols3 = search_3_terms(bound)
        for sol in sols3:
            m = max(sol)
            if m < best_max:
                best_max = m
                best = list(sol)

        # 4-term
        if best_max > bound:
            sols4 = search_4_terms(bound)
            for sol in sols4:
                m = max(sol)
                if m < best_max:
                    best_max = m
                    best = list(sol)

        # 5-term
        if best_max > bound:
            sols5 = search_5_terms(bound)
            for sol in sols5:
                m = max(sol)
                if m < best_max:
                    best_max = m
                    best = list(sol)

        if best_max <= 31:
            break

    # Final verification
    s = sum(Fraction(1, d) for d in best)
    if s != target or len(best) > 256 or len(set(best)) != len(best) or any(d < 1 or d > 10000000 for d in best):
        best = greedy(target)

    return best
