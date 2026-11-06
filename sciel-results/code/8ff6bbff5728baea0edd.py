"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from fractions import Fraction

    deadline = time.time() + 85.0  # safety margin before the 90s kill
    target = Fraction(5, 31)

    def greedy_complete(r, start, D):
        vals = list(start)
        maxd = max(vals) if vals else 0
        while r > 0:
            if time.time() > deadline:
                return None
            # smallest d >= maxd+1 (distinct) with 1/d <= r
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d <= maxd:
                d = maxd + 1
            if d > D or d > 10000000:
                return None
            vals.append(d)
            maxd = d
            r -= Fraction(1, d)
            if len(vals) > 256:
                return None
        return vals

    best = None
    best_max = None

    def consider(vals):
        nonlocal best, best_max
        if vals is None:
            return
        m = max(vals)
        if best_max is None or m < best_max:
            best_max = m
            best = vals

    # 1) single starting denominator
    for d in range(1, 5000):
        if time.time() > deadline:
            break
        if d >= 10000000:
            break
        r = target - Fraction(1, d)
        if r <= 0:
            continue
        consider(greedy_complete(r, [d], 10000000))

    # 2) pair starting denominators (bounded, distinct, ascending)
    for d1 in range(1, 300):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, d1)
        if r1 <= 0:
            continue
        for d2 in range(d1 + 1, 3000):
            if time.time() > deadline:
                break
            r = r1 - Fraction(1, d2)
            if r <= 0:
                continue
            consider(greedy_complete(r, [d1, d2], 10000000))

    # 3) triple starting denominators (small)
    for d1 in range(1, 60):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, d1)
        if r1 <= 0:
            continue
        for d2 in range(d1 + 1, 300):
            if time.time() > deadline:
                break
            r2 = r1 - Fraction(1, d2)
            if r2 <= 0:
                continue
            for d3 in range(d2 + 1, 1500):
                if time.time() > deadline:
                    break
                r = r2 - Fraction(1, d3)
                if r <= 0:
                    continue
                consider(greedy_complete(r, [d1, d2, d3], 10000000))

    # 4) plain greedy from the start (no fixed first term)
    consider(greedy_complete(target, [], 10000000))

    # 5) if nothing found, fall back to a guaranteed valid expansion
    if best is None:
        r = target
        vals = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            vals.append(d)
            r -= Fraction(1, d)
        best = vals

    # safety: ensure distinct and in range
    seen = set()
    out = []
    for v in best:
        if v not in seen and 1 <= v <= 10000000:
            seen.add(v)
            out.append(v)
    return out
# EVOLVE-BLOCK-END
