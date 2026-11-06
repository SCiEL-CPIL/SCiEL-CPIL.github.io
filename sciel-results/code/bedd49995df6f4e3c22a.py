import time
from fractions import Fraction


def construct():
    deadline = time.time() + 80.0
    for max_terms in range(5, 257):
        result = _search(Fraction(1, 1), 5, max_terms, deadline)
        if result is not None:
            return result
    raise RuntimeError("no solution")


def _max_sum_consecutive(d1, n):
    s = Fraction(0, 1)
    for i in range(n):
        s += Fraction(1, d1 + i)
    return s


def _ceil_frac(a, b):
    num = a * b.denominator
    den = b.numerator
    return (num + den - 1) // den


def _search(remaining, min_d, terms_left, deadline):
    if time.time() > deadline:
        return None
    if terms_left == 0:
        return [] if remaining == 0 else None
    if remaining <= 0:
        return None
    d_low = max(min_d, _ceil_frac(1, remaining))
    if _max_sum_consecutive(d_low, terms_left) < remaining:
        return None
    lo, hi = d_low, 10000000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _max_sum_consecutive(mid, terms_left) >= remaining:
            lo = mid
        else:
            hi = mid - 1
    d_high = lo
    if d_low > d_high:
        return None
    for d in range(d_low, d_high + 1):
        if time.time() > deadline:
            return None
        new_rem = remaining - Fraction(1, d)
        if new_rem < 0:
            continue
        if terms_left - 1 == 0:
            if new_rem == 0:
                return [d]
            continue
        sub = _search(new_rem, d + 1, terms_left - 1, deadline)
        if sub is not None:
            return [d] + sub
    return None
