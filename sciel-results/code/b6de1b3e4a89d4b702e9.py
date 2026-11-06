"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from math import isqrt, gcd
from fractions import Fraction
import time
import random

# EVOLVE-BLOCK-START

def _is_square(n):
    """Check if n is a perfect square."""
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n


def _dist_sq_rational(p1, p2, D):
    """Check if squared distance between two points is a rational square."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x1 - x2
    dy = y1 - y2
    sq = dx * dx + D * dy * dy
    if sq < 0:
        return False
    num = sq.numerator
    den = sq.denominator
    return _is_square(num) and _is_square(den)


def _collinear(p1, p2, p3):
    """Check if three points are collinear."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    det = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    return det == 0


def _concyclic(p1, p2, p3, p4, D):
    """Check if four points are concyclic."""
    # Circle: x^2 + y^2*D + ax + by + c = 0
    # Build 4x4 matrix
    M = []
    for (x, y) in [p1, p2, p3, p4]:
        M.append([x * x + y * y * D, x, y, 1])
    return _det4(M) == 0


def _det4(M):
    """Compute determinant of a 4x4 matrix of Fractions using Gaussian elimination."""
    M = [row[:] for row in M]
    n = 4
    det = Fraction(1, 1)
    for col in range(n):
        pivot = -1
        for row in range(col, n):
            if M[row][col] != 0:
                pivot = row
                break
        if pivot == -1:
            return Fraction(0, 1)
        if pivot != col:
            M[col], M[pivot] = M[pivot], M[col]
            det = -det
        det *= M[col][col]
        for row in range(col + 1, n):
            if M[row][col] != 0:
                factor = M[row][col] / M[col][col]
                for k in range(col, n):
                    M[row][k] -= factor * M[col][k]
    return det


def _check_can_add(current_pts, new_pt, D):
    """Fast check if new_pt can be added to current_pts."""
    n = len(current_pts)
    # Check distances to all existing points
    for p in current_pts:
        if not _dist_sq_rational(new_pt, p, D):
            return False
    # Check no three collinear with new point
    for i in range(n):
        for j in range(i + 1, n):
            if _collinear(current_pts[i], current_pts[j], new_pt):
                return False
    # Check no four concyclic with new point
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _concyclic(current_pts[i], current_pts[j], current_pts[k], new_pt, D):
                    return False
    return True


def _full_validate(pts, D):
    """Full validation of a point set."""
    n = len(pts)
    if n < 3:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if not _dist_sq_rational(pts[i], pts[j], D):
                return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear(pts[i], pts[j], pts[k]):
                    return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _concyclic(pts[i], pts[j], pts[k], pts[l], D):
                        return False
    return True


def _search_integer_points(max_coord=400, max_points=8, time_limit=100):
    """
    Search for sets of integer points with rational distances, no three collinear, no four concyclic.
    Uses incremental validation for speed.
    """
    start_time = time.time()

    # Start with the known 5-point set
    base_points = [(Fraction(0), Fraction(0)), 
                   (Fraction(-56), Fraction(33)), 
                   (Fraction(-16), Fraction(63)), 
                   (Fraction(0), Fraction(33)), 
                   (Fraction(16), Fraction(63))]

    best = list(base_points)
    D = 1

    # Generate candidate points - focus on points that form rational distances with base
    candidates = []
    base_set = set()
    for (x, y) in base_points:
        base_set.add((x.numerator, x.denominator, y.numerator, y.denominator))

    # Generate candidates more efficiently - only check distance to first point first
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            fp = (Fraction(x), Fraction(y))
            # Quick check: distance to origin must be rational
            if not _is_square(x * x + y * y):
                continue
            candidates.append(fp)

    # Shuffle for variety
    random.seed(42)
    random.shuffle(candidates)

    current = list(base_points)
    for cand in candidates:
        if time.time() - start_time > time_limit:
            break
        if _check_can_add(current, cand, D):
            current.append(cand)
            if len(current) > len(best):
                best = list(current)
            if len(best) >= max_points:
                break

    # Try multiple random restarts with different starting points
    for attempt in range(30):
        if time.time() - start_time > time_limit:
            break
        random.seed(attempt * 7 + 13)
        
        # Pick 4 random valid starting points
        start_pts = []
        used = set()
        attempts = 0
        while len(start_pts) < 4 and attempts < 1000:
            attempts += 1
            x = random.randint(-max_coord, max_coord)
            y = random.randint(-max_coord, max_coord)
            fp = (Fraction(x), Fraction(y))
            key = (x, y)
            if key in used:
                continue
            # Check if this point works with existing start points
            valid = True
            for sp in start_pts:
                if not _dist_sq_rational(fp, sp, D) or _collinear(fp, sp, start_pts[0]) if len(start_pts) >= 2 else True:
                    valid = False
                    break
            if valid:
                start_pts.append(fp)
                used.add(key)
        
        if len(start_pts) < 4:
            continue
        if not _full_validate(start_pts, D):
            continue

        current = list(start_pts)
        for cand in candidates:
            if time.time() - start_time > time_limit:
                break
            if any(cand == cp for cp in current):
                continue
            if _check_can_add(current, cand, D):
                current.append(cand)
                if len(current) > len(best):
                    best = list(current)
                if len(best) >= max_points:
                    break

    return [(x.numerator, x.denominator, y.numerator, y.denominator) for (x, y) in best]


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # Start with the known 5-point solution
    base_points = [(0, 1, 0, 1), (-56, 1, 33, 1), (-16, 1, 63, 1), (0, 1, 33, 1), (16, 1, 63, 1)]
    best = list(base_points)
    best_D = 1

    # Try to extend with more points using integer search
    if time.time() < deadline - 5:
        extended = _search_integer_points(
            max_coord=400,
            max_points=8,
            time_limit=min(110, deadline - time.time() - 5)
        )
        if len(extended) > len(best):
            best = extended
            best_D = 1

    # Verify the best solution
    pts_frac = [(Fraction(xn, xd), Fraction(yn, yd)) for (xn, xd, yn, yd) in best]
    if not _full_validate(pts_frac, best_D):
        # Fallback to base
        best = list(base_points)
        best_D = 1

    return {
        'D': best_D,
        'points': [[[x, d], [y, dy]] for (x, d, y, dy) in best]
    }
# EVOLVE-BLOCK-END
