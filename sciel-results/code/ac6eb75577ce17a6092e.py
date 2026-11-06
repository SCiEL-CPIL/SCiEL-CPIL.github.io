"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

_MAX_DEN = 10000000


def _max_fill(last, count):
    # Max reciprocal sum achievable with `count` distinct denominators > last.
    s = Fraction(0)
    for j in range(1, count + 1):
        s += Fraction(1, last + j)
    return s


def _search(k, deadline):
    # Backtracking for exactly k distinct denominators >= 5 summing to 1.
    def rec(remaining, last, chosen, S):
        if time.time() > deadline:
            return None
        if remaining == 0:
            return chosen[:] if S == 1 else None
        if S > 1:
            return None
        R = 1 - S  # sum still to be filled by `remaining` terms
        if R <= 0:
            return None
        if remaining == 1:
            inv = Fraction(1) / R
            if inv.denominator == 1:
                d = inv.numerator
                if d > last and d >= 5 and d <= _MAX_DEN:
                    return chosen + [d]
            return None
        # remaining >= 2: choose next denominator d > last with 1/d < R
        d_low = max(last + 1, R.denominator // R.numerator + 1)
        d = d_low
        guard = 0
        while d <= _MAX_DEN:
            guard += 1
            if guard > 2000000 or time.time() > deadline:
                return None
            one_d = Fraction(1, d)
            if one_d >= R:
                d += 1
                continue
            R_after = R - one_d
            # If even the smallest available denominators can't fill R_after,
            # larger d only worsens both sides -> prune the whole tail.
            if R_after > _max_fill(d, remaining - 1):
                break
            res = rec(remaining - 1, d, chosen + [d], S + one_d)
            if res is not None:
                return res
            d += 1
        return None

    return rec(k, 4, [], Fraction(0))


def _fallback():
    # Guaranteed-valid baseline (subset DP over divisors of 2520).
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    raise RuntimeError("no baseline representation")


def construct():
    # k < 8 is impossible (max sum of 7 largest allowed reciprocals < 1).
    deadline = time.time() + 50.0
    for k in range(8, 257):
        sol = _search(k, deadline)
        if sol is not None:
            return sol
    return _fallback()
# EVOLVE-BLOCK-END
