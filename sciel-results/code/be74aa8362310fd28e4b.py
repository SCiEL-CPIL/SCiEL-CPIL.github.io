"""Integral point sets in general position - search for 6+ points."""
import time
from math import isqrt, gcd
from itertools import combinations

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _check_valid(pts):
    """Check all constraints for a set of integer points."""
    n = len(pts)
    if n < 3:
        return False
    if len(set(pts)) != n:
        return False
    # Check all pairwise distances are integers
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            d2 = dx * dx + dy * dy
            if not _is_square(d2):
                return False
    # Check no three collinear
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                if (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) == 0:
                    return False
    # Check no four concyclic
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = pts[i], pts[j], pts[k], pts[l]
                    a11 = x1 - x2
                    a12 = y1 - y2
                    b1 = (x2*x2 + y2*y2) - (x1*x1 + y1*y1)
                    a21 = x1 - x3
                    a22 = y1 - y3
                    b2 = (x3*x3 + y3*y3) - (x1*x1 + y1*y1)
                    det = a11 * a22 - a21 * a12
                    if det == 0:
                        continue
                    A_num = b1 * a22 - b2 * a12
                    B_num = a11 * b2 - a21 * b1
                    C_num = -(x1*x1 + y1*y1) * det - x1 * A_num - y1 * B_num
                    lhs = x4 * A_num + y4 * B_num + C_num
                    rhs = -(x4*x4 + y4*y4) * det
                    if lhs == rhs:
                        return False
    return True

def _generate_pythagorean(max_c):
    """Generate all Pythagorean pairs (a, b, c) with a^2+b^2=c^2, c<=max_c, a,b>0."""
    result = []
    for c in range(5, max_c + 1):
        for a in range(1, c):
            b2 = c * c - a * a
            b = isqrt(b2)
            if b * b == b2 and b > 0:
                result.append((a, b, c))
    return result

def _search_y_symmetric(pythagorean, deadline):
    """Search for 6 points with y-axis symmetry: (±a1,b1), (±a2,b2), (±a3,b3)."""
    n = len(pythagorean)
    for i in range(n):
        if time.time() > deadline:
            return None
        a1, b1, c1 = pythagorean[i]
        for j in range(i + 1, n):
            if time.time() > deadline:
                return None
            a2, b2, c2 = pythagorean[j]
            d2_same = (a1 - a2)**2 + (b1 - b2)**2
            if not _is_square(d2_same):
                continue
            d2_cross = (a1 + a2)**2 + (b1 - b2)**2
            if not _is_square(d2_cross):
                continue
            for k in range(j + 1, n):
                a3, b3, c3 = pythagorean[k]
                d2_13_same = (a1 - a3)**2 + (b1 - b3)**2
                if not _is_square(d2_13_same):
                    continue
                d2_13_cross = (a1 + a3)**2 + (b1 - b3)**2
                if not _is_square(d2_13_cross):
                    continue
                d2_23_same = (a2 - a3)**2 + (b2 - b3)**2
                if not _is_square(d2_23_same):
                    continue
                d2_23_cross = (a2 + a3)**2 + (b2 - b3)**2
                if not _is_square(d2_23_cross):
                    continue
                pts = [(a1, b1), (-a1, b1), (a2, b2), (-a2, b2), (a3, b3), (-a3, b3)]
                if _check_valid(pts):
                    return pts
    return None

def _greedy_base_search(deadline):
    """Greedy search starting from base points (0,0) and (1,0)."""
    base = [(0, 0), (1, 0)]
    candidates = []
    for a in range(1, 120):
        for b in range(1, 120):
            if a % 2 == b % 2:
                continue
            x_num = a * a - b * b + 1
            if x_num % 2 != 0:
                continue
            x = x_num // 2
            y2 = a * a - x * x
            if y2 < 0:
                continue
            y = isqrt(y2)
            if y * y != y2 or y == 0:
                continue
            candidates.append((x, y))
    candidates = list(set(candidates))
    best = list(base)
    for pt in candidates:
        if time.time() > deadline:
            break
        new_set = best + [pt]
        if _check_valid(new_set):
            best = new_set
    return best

def _try_extend(pts, pythagorean_all, deadline):
    """Try to add one more point to an existing set."""
    current = list(pts)
    existing = set(current)
    for a, b, c in pythagorean_all:
        if time.time() > deadline:
            break
        for sx in [1, -1]:
            for sy in [1, -1]:
                pt = (sx * a, sy * b)
                if pt in existing:
                    continue
                new_pts = current + [pt]
                if _check_valid(new_pts):
                    current = new_pts
                    existing = set(current)
                    break
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    # Seed 5-point set
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best = seed_points if _check_valid(seed_points) else None

    # Generate Pythagorean pairs
    max_c = 250
    pythagorean = _generate_pythagorean(max_c)
    pythagorean.sort(key=lambda x: x[2])

    # Strategy 1: Y-symmetric search for 6 points
    found = _search_y_symmetric(pythagorean, deadline)
    if found:
        best = found

    # Strategy 2: Greedy base search if we don't have 6+ points
    if not best or len(best) < 6:
        greedy_result = _greedy_base_search(deadline)
        if len(greedy_result) > (len(best) if best else 0):
            best = greedy_result

    # Try to extend to 7, 8 points
    if best and len(best) >= 5 and time.time() < deadline - 10:
        best = _try_extend(best, pythagorean, deadline)

    # If still only 5, try harder y-symmetric with larger range
    if best and len(best) < 6 and time.time() < deadline - 20:
        pythagorean_large = _generate_pythagorean(400)
        pythagorean_large.sort(key=lambda x: x[2])
        found2 = _search_y_symmetric(pythagorean_large, deadline)
        if found2 and len(found2) > len(best):
            best = found2

    if best is None:
        best = seed_points

    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best]}
