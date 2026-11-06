"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
import time

# EVOLVE-BLOCK-START

def _search(k, target, min_den, max_den, deadline):
    if time.time() > deadline:
        return None
    if k == 0:
        return [] if target == 0 else None
    if k == 1:
        if target.denominator == 1:
            d = target.numerator
            if min_den <= d <= max_den:
                return [d]
        return None
    if k == 2:
        p, q = target.numerator, target.denominator
        if p == 0:
            return None
        q2 = q * q
        i = 1
        while i * i <= q2:
            if q2 % i == 0:
                f1, f2 = i, q2 // i
                for fa, fb in [(f1, f2), (f2, f1)]:
                    na, nb = fa + q, fb + q
                    if na % p == 0 and nb % p == 0:
                        a, b = na // p, nb // p
                        if min_den <= a < b <= max_den:
                            return [a, b]
                        if min_den <= b < a <= max_den:
                            return [b, a]
            i += 1
        return None
    if target <= 0:
        return None

    a_start = max(min_den, int(1 / float(target)) + 1)
    a_end = min(max_den, int(k / float(target)) + 2)
    if a_end < a_start:
        return None

    for a in range(a_start, a_end + 1):
        if time.time() > deadline:
            return None
        rem = target - Fraction(1, a)
        if rem <= 0:
            continue
        if k <= 20:
            max_sum = sum(Fraction(1, a + i) for i in range(1, k))
            if rem > max_sum:
                continue
        result = _search(k - 1, rem, a + 1, max_den, deadline)
        if result is not None:
            return [a] + result
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 75

    for k in range(8, 11):
        if time.time() > deadline - 3:
            break
        sol = _search(k, Fraction(1, 1), 5, 10000000, deadline)
        if sol is not None:
            return sol

    return [5, 6, 7, 8, 9, 10, 14, 15, 63, 2520]
# EVOLVE-BLOCK-END
