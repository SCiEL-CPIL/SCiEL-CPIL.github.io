    def greedy_complete(r0):
        vals = []
        r = r0
        while r > 0 and len(vals) < MAX_TERMS:
            num, den = r.numerator, r.denominator
            d = (den + num - 1) // num
            if d > MAX_DEN:
                return None
            vals.append(d)
            r -= Fraction(1, d)
        if r != 0:
            return None
        return vals

    # quick heuristic to get initial best
    for first in range(2, 30):
        rem = target - Fraction(1, first)
        if rem <= 0:
            continue
        g = greedy_complete(rem)
        if g:
            cand = [first] + g
            if len(set(cand)) == len(cand) and max(cand) < best_max:
                best_max = max(cand)
                best = cand
