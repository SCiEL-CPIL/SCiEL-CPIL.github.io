def construct():
    from fractions import Fraction
    from math import gcd
    import time

    deadline = time.time() + 85

    target = Fraction(5, 31)

    best_solution = None
    best_max = 10000001

    def update_best(sol):
        nonlocal best_solution, best_max
        if not sol:
            return
        if len(set(sol)) != len(sol):
            return
        if any(d < 1 or d > 10000000 for d in sol):
            return
        s = sum(Fraction(1, d) for d in sol)
        if s == target:
            mx = max(sol)
            if mx < best_max:
                best_max = mx
                best_solution = sorted(sol)

    def get_divisors(n):
        """Generate all divisors of n using prime factorization."""
        if n == 0:
            return []
        divisors = [1]
        i = 2
        while i * i <= n:
            if n % i == 0:
                exp = 0
                while n % i == 0:
                    n //= i
                    exp += 1
                new_divs = []
                for d in divisors:
                    power = 1
                    for _ in range(exp + 1):
                        new_divs.append(d * power)
                        power *= i
                divisors = new_divs
            i += 1
        if n > 1:
            new_divs = []
            for d in divisors:
                new_divs.append(d)
                new_divs.append(d * n)
            divisors = new_divs
        return sorted(divisors)

    def solve_2_term(p, q):
        """
        Find pairs (x, y) with x < y such that 1/x + 1/y = p/q.
        p, q are in lowest terms.
        Equation: (px - q)(py - q) = q^2.
        Let u = px - q, v = py - q. Then uv = q^2.
        """
        if p == 0 or q == 0:
            return []
        q2 = q * q
        divs = get_divisors(q2)
        solutions = []
        for u in divs:
            if q2 % u != 0:
                continue
            v = q2 // u
            if (u + q) % p == 0 and (v + q) % p == 0:
                x = (u + q) // p
                y = (v + q) // p
                if x < 1 or y < 1:
                    continue
                if x == y:
                    continue
                if x > y:
                    x, y = y, x
                solutions.append((x, y))
        return solutions

    # Phase 1: 3-term search using divisor method
    # 1/a + 1/b + 1/c = 5/31
    # a >= 7 (since 1/6 > 5/31)
    # 3/a > 5/31 => a < 93/5 = 18.6 => a <= 18
    for a in range(7, 19):
        if time.time() > deadline:
            break
        rem = target - Fraction(1, a)
        if rem <= 0:
            continue
        p = rem.numerator
        q = rem.denominator
        g = gcd(p, q)
        p //= g
        q //= g
        pairs = solve_2_term(p, q)
        for b, c in pairs:
            if b > a and c > b:
                update_best([a, b, c])

    # Phase 2: 4-term search using divisor method
    # 1/a + 1/b + 1/c + 1/d = 5/31
    for a in range(7, 15):
        if time.time() > deadline:
            break
        for b in range(a + 1, 30):
            rem_ab = target - Fraction(1, a) - Fraction(1, b)
            if rem_ab <= 0:
                continue
            p = rem_ab.numerator
            q = rem_ab.denominator
            g = gcd(p, q)
            p //= g
            q //= g
            pairs = solve_2_term(p, q)
            for c, d in pairs:
                if c > b and d > c:
                    update_best([a, b, c, d])

    # Phase 3: 5-term search using divisor method
    for a in range(7, 12):
        if time.time() > deadline:
            break
        for b in range(a + 1, 20):
            if time.time() > deadline:
                break
            rem_ab = target - Fraction(1, a) - Fraction(1, b)
            if rem_ab <= 0:
                continue
            for c in range(b + 1, 40):
                rem_abc = rem_ab - Fraction(1, c)
                if rem_abc <= 0:
                    continue
                p = rem_abc.numerator
                q = rem_abc.denominator
                g = gcd(p, q)
                p //= g
                q //= g
                pairs = solve_2_term(p, q)
                for d, e in pairs:
                    if d > c and e > d:
                        update_best([a, b, c, d, e])

    if best_solution is not None:
        return best_solution

    # Fallback: Greedy algorithm (should not be needed, but just in case)
    r = target
    values = []
    used = set()
    while r and len(values) < 256:
        d = (r.denominator + r.numerator - 1) // r.numerator
        if d in used:
            d += 1
            while d in used:
                d += 1
        if d > 10000000:
            break
        values.append(d)
        used.add(d)
        r -= Fraction(1, d)
    if r == 0 and values:
        return values

    return [8, 31, 248]
