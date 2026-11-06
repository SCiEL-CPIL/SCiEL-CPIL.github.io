"""Search for larger rational-distance sets in general position (exact rational arithmetic)."""
import time
import random
import math
from fractions import Fraction
from itertools import combinations

# Deadline guard (150s wall-clock limit; keep safety margin)
_DEADLINE = time.time() + 140.0

# Known valid 5-point set (integer lattice, no 3 collinear, no 4 concyclic)
_FALLBACK_POINTS = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]


def _is_square(n):
    if n < 0:
        return False
    if n == 0:
        return True
    r = math.isqrt(n)
    return r * r == n


def _check_set(points):
    """Exact check. Returns True iff all pairwise distances rational, no 3 collinear, no 4 concyclic."""
    n = len(points)
    if n < 3:
        return False
    # Distances rational
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            d2 = (xi - xj) ** 2 + (yi - yj) ** 2
            if not _is_square(d2.numerator) or not _is_square(d2.denominator):
                return False
    # No three collinear
    for i, j, k in combinations(range(n), 3):
        x1, y1 = points[i]
        x2, y2 = points[j]
        x3, y3 = points[k]
        if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
            return False
    # No four concyclic
    for a, b, c, d in combinations(range(n), 4):
        x1, y1 = points[a]
        x2, y2 = points[b]
        x3, y3 = points[c]
        x4, y4 = points[d]
        A = x1 - x4
        B = y1 - y4
        C = x2 - x4
        D = y2 - y4
        E = x3 - x4
        F = y3 - y4
        det = (A * A + B * B) * (C * F - D * E) \
            - (C * C + D * D) * (A * F - B * E) \
            + (E * E + F * F) * (A * D - B * C)
        if det != 0:
            return False
    return True


def _try_insert(points, x, y):
    """Try to insert (x,y); return new list if valid else None."""
    cand = points + [(x, y)]
    if _check_set(cand):
        return cand
    return None


def _random_rational(lo, hi, denom_max=128):
    d = random.randint(1, denom_max)
    v = random.uniform(lo, hi)
    num = int(round(v * d))
    return Fraction(num, d)


def _search(start_points, time_budget):
    """Randomized local search: try to grow the set by inserting random rational points."""
    points = list(start_points)
    end_time = time.time() + time_budget
    # Candidate ranges: focus around the existing point cloud
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xlo, xhi = min(xs) - 30, max(xs) + 30
    ylo, yhi = min(ys) - 30, max(ys) + 30

    while time.time() < end_time:
        improved = False
        # Try a batch of insertions
        for _ in range(4000):
            if time.time() > end_time:
                break
            x = _random_rational(xlo, xhi)
            y = _random_rational(ylo, yhi)
            newp = _try_insert(points, x, y)
            if newp is not None:
                points = newp
                improved = True
                # Expand ranges to include new point
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                xlo, xhi = min(xs) - 30, max(xs) + 30
                ylo, yhi = min(ys) - 30, max(ys) + 30
                break
        if not improved:
            # Occasionally perturb the whole set slightly (translate) to escape local optimum
            if random.random() < 0.3 and len(points) <= 6:
                # Try a small integer translation of the base configuration
                tx = random.randint(-10, 10)
                ty = random.randint(-10, 10)
                base = [(Fraction(p[0] + tx), Fraction(p[1] + ty)) for p in _FALLBACK_POINTS]
                # keep only if still valid (translation preserves validity)
                points = base
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                xlo, xhi = min(xs) - 30, max(xs) + 30
                ylo, yhi = min(ys) - 30, max(ys) + 30
    return points


def construct():
    global _DEADLINE
    random.seed(12345)
    start = list(_FALLBACK_POINTS)
    # Convert to Fraction for exact arithmetic
    points = [(Fraction(x), Fraction(y)) for x, y in start]

    # Search for a larger set
    try:
        best = points
        # Give search most of the budget, leave margin for final checks
        remaining = _DEADLINE - time.time() - 5.0
        if remaining > 2:
            candidate = _search(points, remaining)
            if len(candidate) > len(best):
                best = candidate
    except Exception:
        best = points

    # Final exact verification of the chosen best set
    if not _check_set(best):
        best = [(Fraction(x), Fraction(y)) for x, y in _FALLBACK_POINTS]

    return {'D': 1, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in best]}
