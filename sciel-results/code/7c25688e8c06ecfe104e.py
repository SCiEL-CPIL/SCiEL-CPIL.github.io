"""Egyptian fraction of 5/31 with minimal largest denominator."""
import time
from fractions import Fraction

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256

def _verify(denoms):
    """Verify a set of denominators sums to TARGET and meets constraints."""
    if not denoms:
        return False
    if len(denoms) > MAX_TERMS:
        return False
    if len(set(denoms)) != len(denoms):
        return False
    if any(d < 1 or d > MAX_DENOM for d in denoms):
        return False
    s = sum(Fraction(1, d) for d in denoms)
    return s == TARGET

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Known solution: 5/31 = 1/8 + 1/31 + 1/248
    known = [8, 31, 248]
    if _verify(known):
        return known

    # Try to find a solution with smaller largest denominator
    # For 3-term: 1/a + 1/b + 1/c = 5/31, minimize c
    # For each a from 6 upward, b from a+1 upward, solve for c
    # 1/c = 5/31 - 1/a - 1/b = (5ab - 31b - 31a) / (31ab)
    # So c = 31ab / (5ab - 31a - 31b), must be integer > b

    best_result = known[:]
    best_max_denom = 248

    # Search for 3-term solutions with c < 248
    # a must be at least ceil(31/5) = 7, but also 1/a < 5/31 so a > 31/5 = 6.2, so a >= 7
    # Actually 1/6 = 0.1666 > 5/31 = 0.16129, so a must be >= 7
    for a in range(7, 100):
        if time.time() > deadline:
            break
        for b in range(a + 1, 200):
            if time.time() > deadline:
                break
            # c = 31*a*b / (5*a*b - 31*a - 31*b)
            num = 31 * a * b
            den = 5 * a * b - 31 * a - 31 * b
            if den <= 0:
                continue
            if num % den != 0:
                continue
            c = num // den
            if c <= b:
                continue
            if c >= best_max_denom:
                continue
            # Verify
            s = Fraction(1, a) + Fraction(1, b) + Fraction(1, c)
            if s == TARGET and c <= MAX_DENOM:
                best_result = [a, b, c]
                best_max_denom = c

    # Also try 2-term solutions: 1/a + 1/b = 5/31
    # 1/b = 5/31 - 1/a = (5a - 31)/(31a)
    # b = 31a / (5a - 31)
    for a in range(7, 200):
        if time.time() > deadline:
            break
        den = 5 * a - 31
        if den <= 0:
            continue
        num = 31 * a
        if num % den != 0:
            continue
        b = num // den
        if b <= a:
            continue
        if b >= best_max_denom:
            continue
        s = Fraction(1, a) + Fraction(1, b)
        if s == TARGET and b <= MAX_DENOM:
            best_result = [a, b]
            best_max_denom = b

    # Try 4-term solutions with max denominator < best_max_denom
    # 1/a + 1/b + 1/c + 1/d = 5/31
    # For each a, b, c, solve for d
    if best_max_denom > 100 and time.time() < deadline - 5:
        for a in range(7, 50):
            if time.time() > deadline:
                break
            for b in range(a + 1, 80):
                if time.time() > deadline:
                    break
                for c in range(b + 1, 120):
                    if time.time() > deadline:
                        break
                    # 1/d = 5/31 - 1/a - 1/b - 1/c
                    num = 31 * a * b * c
                    den = 5 * a * b * c - 31 * b * c - 31 * a * c - 31 * a * b
                    if den <= 0:
                        continue
                    if num % den != 0:
                        continue
                    d = num // den
                    if d <= c:
                        continue
                    if d >= best_max_denom:
                        continue
                    s = Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d)
                    if s == TARGET and d <= MAX_DENOM:
                        best_result = [a, b, c, d]
                        best_max_denom = d

    return best_result
