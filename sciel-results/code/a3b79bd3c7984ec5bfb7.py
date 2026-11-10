"""Improved program: systematic search for larger rational-distance point sets."""
import time
import math
import random
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_square(n):
    """Check if n (int) is a perfect square."""
    if not isinstance(n, int):
        n = int(n)
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n


def _is_squarefree(n):
    """Check if n is squarefree."""
    if not isinstance(n, int):
        n = int(n)
    if n <= 1:
        return True
    for p in range(2, math.isqrt(n) + 1):
        if n % (p * p) == 0:
            return False
    return True


def _sq_dist_sq(p1, p2, D):
    """Return squared distance as a Fraction between (x1,y1) and (x2,y2) in the (x, y*sqrt(D)) space."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + D * dy * dy


def _is_rational_square(fr):
    """Check if Fraction fr is a perfect square of a rational number."""
    if fr < 0:
        return False
    num = fr.numerator
    den = fr.denominator
    return _is_square(num) and _is_square(den)


def _check_all_distances_rational(points, D):
    """Check that all pairwise squared distances are perfect squares of rationals."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            d2 = _sq_dist_sq(points[i], points[j], D)
            if not _is_rational_square(d2):
                return False
    return True


def _check_no_three_collinear(points):
    """Check no three points are collinear using exact arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                det = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                if det == 0:
                    return False
    return True


def _check_no_four_concyclic(points, D):
    """Check no four points are concyclic using exact arithmetic."""
    n = len(points)
    for quad in combinations(range(n), 4):
        i, j, k, l = quad
        x1, y1 = points[i]
        x2, y2 = points[j]
        x3, y3 = points[k]
        x4, y4 = points[l]
        
        # Translate so p1 is at origin
        x2p = x2 - x1
        y2p = y2 - y1
        x3p = x3 - x1
        y3p = y3 - y1
        x4p = x4 - x1
        y4p = y4 - y1
        
        # Determinant of:
        # [x2p, y2p, -(x2p^2 + D*y2p^2)]
        # [x3p, y3p, -(x3p^2 + D*y3p^2)]
        # [x4p, y4p, -(x4p^2 + D*y4p^2)]
        q2 = x2p * x2p + D * y2p * y2p
        q3 = x3p * x3p + D * y3p * y3p
        q4 = x4p * x4p + D * y4p * y4p
        
        det = (x2p * (y3p * (-q4) - y4p * (-q3))
               - y2p * (x3p * (-q4) - x4p * (-q3))
               + (-q2) * (x3p * y4p - x4p * y3p))
        
        if det == 0:
            return False
    return True


def _validate(points_frac, D):
    """Full validation of a point set."""
    if len(points_frac) < 3:
        return False
    if not _is_squarefree(D):
        return False
    # Check distinctness
    seen = set()
    for p in points_frac:
        key = (p[0], p[1])
        if key in seen:
            return False
        seen.add(key)
    if not _check_all_distances_rational(points_frac, D):
        return False
    if not _check_no_three_collinear(points_frac):
        return False
    if not _check_no_four_concyclic(points_frac, D):
        return False
    return True


def _build_known_5points():
    """Return the known 5-point configuration with D=1."""
    D = 1
    pts = [(Fraction(0), Fraction(0)),
           (Fraction(-56), Fraction(33)),
           (Fraction(-16), Fraction(63)),
           (Fraction(0), Fraction(33)),
           (Fraction(16), Fraction(63))]
    return pts, D


def _try_add_point(current_points, D, candidates):
    """Try to add one candidate point to current set. Returns new list or None."""
    for cand in candidates:
        if cand in current_points:
            continue
        new_set = current_points + [cand]
        if _validate(new_set, D):
            return new_set
    return None


def _generate_candidates_near(points, D, radius, seed):
    """Generate random integer candidates near the existing points."""
    rng = random.Random(seed)
    candidates = []
    xs = [p[0].numerator for p in points]
    ys = [p[1].numerator for p in points]
    min_x = min(xs) - radius
    max_x = max(xs) + radius
    min_y = min(ys) - radius
    max_y = max(ys) + radius
    for _ in range(300):
        x = rng.randint(min_x, max_x)
        y = rng.randint(min_y, max_y)
        candidates.append((Fraction(x), Fraction(y)))
    return candidates


def _search_expand(start_points, D, deadline, max_extra=7):
    """Greedily try to expand the point set by adding points one at a time."""
    current = list(start_points)
    
    for step in range(max_extra):
        if time.time() > deadline:
            break
        
        # Generate candidates with increasing radius
        found = False
        for radius in [30, 50, 80, 120, 200]:
            if time.time() > deadline:
                break
            for seed in range(50):
                if time.time() > deadline:
                    break
                cands = _generate_candidates_near(current, D, radius, 1000 + step * 1000 + seed)
                new_set = _try_add_point(current, D, cands)
                if new_set is not None:
                    current = new_set
                    found = True
                    break
            if found:
                break
        
        if not found:
            break
    
    return current


def _try_pythagorean_expansion(points, D, deadline):
    """
    Try to find a 6th point by looking for integer points (a,b) such that
    (a-xi)^2 + D*(b-yi)^2 is a perfect square for all existing points (xi, yi).
    This is a system of Diophantine equations. We search over small integer (a,b).
    """
    n = len(points)
    # Precompute for each existing point
    best = None
    
    # Search over a range of integer coordinates
    xs = [p[0].numerator for p in points]
    ys = [p[1].numerator for p in points]
    center_x = sum(xs) // n
    center_y = sum(ys) // n
    spread_x = max(xs) - min(xs)
    spread_y = max(ys) - min(ys)
    radius = max(spread_x, spread_y) + 60
    
    # Try a grid of candidates
    for dx in range(-radius, radius + 1, 1):
        if time.time() > deadline:
            break
        a = center_x + dx
        for dy in range(-radius, radius + 1, 1):
            if time.time() > deadline:
                break
            b = center_y + dy
            cand = (Fraction(a), Fraction(b))
            if cand in points:
                continue
            
            # Quick check: all squared distances must be perfect squares
            ok = True
            for p in points:
                d2 = _sq_dist_sq(cand, p, D)
                if not _is_rational_square(d2):
                    ok = False
                    break
            if not ok:
                continue
            
            # Check collinearity with all pairs
            collinear = False
            for i in range(n):
                for j in range(i + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    det = (cand[0] - x1) * (y2 - y1) - (cand[1] - y1) * (x2 - x1)
                    if det == 0:
                        collinear = True
                        break
                if collinear:
                    break
            if collinear:
                continue
            
            # Check concyclicity with all triples
            concyclic = False
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        x1, y1 = points[i]
                        x2, y2 = points[j]
                        x3, y3 = points[k]
                        # Check if cand, p1, p2, p3 are concyclic
                        # Use the same determinant trick
                        # Translate so cand is at origin
                        x1p = x1 - cand[0]
                        y1p = y1 - cand[1]
                        x2p = x2 - cand[0]
                        y2p = y2 - cand[1]
                        x3p = x3 - cand[0]
                        y3p = y3 - cand[1]
                        
                        q1 = x1p * x1p + D * y1p * y1p
                        q2 = x2p * x2p + D * y2p * y2p
                        q3 = x3p * x3p + D * y3p * y3p
                        
                        det = (x1p * (y2p * (-q3) - y3p * (-q2))
                               - y1p * (x2p * (-q3) - x3p * (-q2))
                               + (-q1) * (x2p * y3p - x3p * y2p))
                        if det == 0:
                            concyclic = True
                            break
                    if concyclic:
                        break
                if concyclic:
                    break
            if concyclic:
                continue
            
            # All checks passed!
            best = points + [cand]
            break
        if best is not None:
            break
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Start with known 5-point set
    base_points, D = _build_known_5points()
    
    if not _validate(base_points, D):
        raise RuntimeError("Base configuration failed validation")
    
    best = list(base_points)
    
    # Strategy 1: Pythagorean expansion (grid search for 6th point)
    result = _try_pythagorean_expansion(best, D, deadline)
    if result is not None and len(result) > len(best):
        best = result
    
    # Strategy 2: Random expansion for 7th, 8th points
    expanded = _search_expand(best, D, deadline, max_extra=3)
    if len(expanded) > len(best):
        best = expanded
    
    # Strategy 3: Try different D values with small point sets
    # For D > 1, the geometry is different. Try D=2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 46, 47, 51, 53, 54, 55, 57, 58, 59, 61, 62, 65, 66, 67, 69, 70, 71, 73, 74, 77, 78, 79, 82, 83, 85, 86, 87, 89, 90, 93, 94, 95, 97
    # But this is complex. For now, focus on D=1.
    
    # Try a few more random attempts with D=1
    if time.time() < deadline - 5:
        rng = random.Random(999)
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            # Try to find a different 5-point base and expand
            # Or just try random 6-point sets
            pts = []
            for _ in range(6):
                x = rng.randint(-100, 100)
                y = rng.randint(-100, 100)
                pts.append((Fraction(x), Fraction(y)))
            if _validate(pts, D):
                if len(pts) > len(best):
                    best = pts
                    break
    
    # Convert to required format
    result_points = []
    for x, y in best:
        result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': D, 'points': result_points}

# EVOLVE-BLOCK-END
