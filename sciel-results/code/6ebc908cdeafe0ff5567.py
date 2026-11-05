"""Combined program: exact divisor-based pair search for Egyptian fraction of 5/31.
Minimizes the largest denominator."""
import time
from fractions import Fraction
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    target = Fraction(5, 31)

    # Greedy fallback solution
    def greedy_solution(target):
        denominators = []
        r = target
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            denominators.append(d)
            r -= Fraction(1, d)
        return denominators

    best_solution = greedy_solution(target)
    best_max = max(best_solution)

    def time_ok():
        return time.time() < deadline

    def factorize(n):
        """Return prime factorization as dict {prime: exponent}."""
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

    def gen_divisors(factors):
        """Generate all divisors from prime factorization dict."""
        if not factors:
            return [1]
        primes = list(factors.keys())
        exponents = [factors[p] for p in primes]
        divisors = [1]
        for i, p in enumerate(primes):
            new_divs = []
            p_powers = [p ** e for e in range(exponents[i] + 1)]
            for d in divisors:
                for pp in p_powers:
                    new_divs.append(d * pp)
            divisors = new_divs
        return sorted(divisors)

    # Cache for divisors of q^2
    _divisor_cache = {}

    def get_divisors_of_q2(q):
        """Get sorted divisors of q^2, with caching."""
        if q in _divisor_cache:
            return _divisor_cache[q]
        q_factors = factorize(q)
        q2_factors = {p: e * 2 for p, e in q_factors.items()}
        divs = gen_divisors(q2_factors)
        _divisor_cache[q] = divs
        return divs

    def find_pairs(p, q, x_min, x_max, bound):
        """Find all pairs (x, y) with x_min <= x < y <= bound such that 1/x + 1/y = p/q.
        Uses divisor enumeration of q^2 for exactness."""
        if p <= 0 or q <= 0:
            return []
        g = gcd(p, q)
        p //= g
        q //= g

        q_sq = q * q
        pairs = []

        # d = p*x - q must be a positive divisor of q^2
        # x = (d + q) / p must be integer
        # y = (q_sq/d + q) / p must be integer, y > x, y <= bound
        d_min = max(1, p * x_min - q)
        d_max = p * x_max - q

        if d_min > d_max:
            return []

        divisors = get_divisors_of_q2(q)

        for d in divisors:
            if d < d_min or d > d_max:
                continue
            if (d + q) % p != 0:
                continue
            x = (d + q) // p
            if x < x_min or x > x_max:
                continue
            d2 = q_sq // d
            if (d2 + q) % p != 0:
                continue
            y = (d2 + q) // p
            if y <= x or y > bound:
                continue
            # Verify
            if Fraction(1, x) + Fraction(1, y) != Fraction(p, q):
                continue
            pairs.append((x, y))

        return pairs

    # Search for 3-term: 1/a + 1/b + 1/c = 5/31, a < b < c
    # 1/a < 5/31 => a > 31/5 = 6.2, so a >= 7
    # 3/a > 5/31 => a < 93/5 = 18.6, so a <= 18
    def search_3_terms(bound):
        nonlocal best_solution, best_max
        for a in range(7, 19):
            if not time_ok():
                return
            r = target - Fraction(1, a)
            if r <= 0:
                continue
            p, q = r.numerator, r.denominator
            # b > a, and 2/b > r => b < 2q/p
            b_min = a + 1
            b_max = min(2 * q // p + 1, bound)
            if b_max < b_min:
                continue
            pairs = find_pairs(p, q, b_min, b_max, bound)
            for b, c in pairs:
                if c < best_max:
                    best_max = c
                    best_solution = [a, b, c]

    # Search for 4-term: 1/a + 1/b + 1/c + 1/d = 5/31
    # 4/a > 5/31 => a < 124/5 = 24.8, so a <= 24
    def search_4_terms(bound):
        nonlocal best_solution, best_max
        for a in range(7, 25):
            if not time_ok():
                return
            r1 = target - Fraction(1, a)
            if r1 <= 0:
                continue
            # 3/b > r1 => b < 3/r1
            b_min = a + 1
            b_max = min(int(3 / r1) + 1, bound)
            for b in range(b_min, b_max + 1):
                if not time_ok():
                    return
                r2 = r1 - Fraction(1, b)
                if r2 <= 0:
                    continue
                p, q = r2.numerator, r2.denominator
                c_min = b + 1
                c_max = min(2 * q // p + 1, bound)
                if c_max < c_min:
                    continue
                pairs = find_pairs(p, q, c_min, c_max, bound)
                for c, d in pairs:
                    if d < best_max:
                        best_max = d
                        best_solution = [a, b, c, d]

    # Search for 5-term: 1/a + 1/b + 1/c + 1/d + 1/e = 5/31
    # 5/a > 5/31 => a < 31, so a <= 30
    def search_5_terms(bound):
        nonlocal best_solution, best_max
        for a in range(7, 31):
            if not time_ok():
                return
            r1 = target - Fraction(1, a)
            if r1 <= 0:
                continue
            b_min = a + 1
            b_max = min(int(4 / r1) + 1, bound)
            for b in range(b_min, b_max + 1):
                if not time_ok():
                    return
                r2 = r1 - Fraction(1, b)
                if r2 <= 0:
                    continue
                c_min = b + 1
                c_max = min(int(3 / r2) + 1, bound)
                for c in range(c_min, c_max + 1):
                    if not time_ok():
                        return
                    r3 = r2 - Fraction(1, c)
                    if r3 <= 0:
                        continue
                    p, q = r3.numerator, r3.denominator
                    d_min = c + 1
                    d_max = min(2 * q // p + 1, bound)
                    if d_max < d_min:
                        continue
                    pairs = find_pairs(p, q, d_min, d_max, bound)
                    for d, e in pairs:
                        if e < best_max:
                            best_max = e
                            best_solution = [a, b, c, d, e]

    # Search for 6-term (in case 5-term doesn't find something better)
    def search_6_terms(bound):
        nonlocal best_solution, best_max
        for a in range(7, 31):
            if not time_ok():
                return
            r1 = target - Fraction(1, a)
            if r1 <= 0:
                continue
            b_min = a + 1
            b_max = min(int(5 / r1) + 1, bound)
            for b in range(b_min, b_max + 1):
                if not time_ok():
                    return
                r2 = r1 - Fraction(1, b)
                if r2 <= 0:
                    continue
                c_min = b + 1
                c_max = min(int(4 / r2) + 1, bound)
                for c in range(c_min, c_max + 1):
                    if not time_ok():
                        return
                    r3 = r2 - Fraction(1, c)
                    if r3 <= 0:
                        continue
                    d_min = c + 1
                    d_max = min(int(3 / r3) + 1, bound)
                    for d in range(d_min, d_max + 1):
                        if not time_ok():
                            return
                        r4 = r3 - Fraction(1, d)
                        if r4 <= 0:
                            continue
                        p, q = r4.numerator, r4.denominator
                        e_min = d + 1
                        e_max = min(2 * q // p + 1, bound)
                        if e_max < e_min:
                            continue
                        pairs = find_pairs(p, q, e_min, e_max, bound)
                        for e, f in pairs:
                            if f < best_max:
                                best_max = f
                                best_solution = [a, b, c, d, e, f]

    # Systematic search with increasing bounds
    for bound in [20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 500, 1000, 2000, 5000, 10000]:
        if not time_ok():
            break

        search_3_terms(bound)

        if best_max > bound:
            search_4_terms(bound)

        if best_max > bound:
            search_5_terms(bound)

        if best_max > bound:
            search_6_terms(bound)

        # Early termination if we found a very small max
        if best_max <= 31:
            break

    # Final verification
    if best_solution is not None:
        s = sum(Fraction(1, d) for d in best_solution)
        if s != target:
            best_solution = greedy_solution(target)
            best_max = max(best_solution)

        # Ensure constraints
        if (len(best_solution) > 256 or len(set(best_solution)) != len(best_solution)
                or any(d < 1 or d > 10000000 for d in best_solution)):
            best_solution = greedy_solution(target)
            best_max = max(best_solution)

    return best_solution
