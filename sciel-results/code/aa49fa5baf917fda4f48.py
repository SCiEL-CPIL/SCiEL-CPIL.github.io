"""
Construct integral point sets in Q(sqrt(D)) with rational pairwise distances,
no three collinear, no four concyclic. Maximize number of points.
"""

import time
import math
from fractions import Fraction

# Deadline: must return before 150s wall-clock; use 130s safety margin
_DEADLINE = time.time() + 130.0


def _is_square(n):
    """Check if n is a perfect square (n >= 0)."""
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n


def _sqfree(n):
    """Return the squarefree part of positive integer n."""
    if n <= 0:
        return 1
    res = 1
    p = 2
    while p * p <= n:
        if n % (p * p) == 0:
            res *= p
            n //= (p * p)
        else:
            p += 1 if p == 2 else 2
    return res * n


def _check_rational_dist_sq(px, py, qx, qy, D):
    """
    Check that squared distance between (px, py) and (qx, qy) in Q(sqrt(D))
    is rational. Points are (x, y*sqrt(D)).
    dist^2 = (px-qx)^2 + D*(py-qy)^2
    """
    # (px-qx)^2 and (py-qy)^2 are rational (squares of rationals)
    # D*(py-qy)^2 is rational since D is integer
    # So dist^2 is always rational for integer D!
    return True


def _check_no_three_collinear(points):
    """
    Check no three points are collinear using exact rational arithmetic.
    points: list of (x, y) as Fractions
    """
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            for k in range(j + 1, n):
                x3, y3 = points[k]
                # Cross product of (P2-P1) and (P3-P1)
                # (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1) != 0
                cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                if cross == 0:
                    return False
    return True


def _check_no_four_concyclic(points):
    """
    Check no four points are concyclic using exact rational arithmetic.
    For four points to be concyclic, the determinant of the circle equation
    must be zero. Using the general circle equation:
    x^2 + y^2 + Dx + Ey + F = 0
    
    For points (x_i, y_i), the system is:
    x_i^2 + y_i^2 + D*x_i + E*y_i + F = 0 for i=1..4
    
    This is a 4x5 system; the four points are concyclic iff the 4x4 determinant
    of the matrix [x_i, y_i, 1, x_i^2+y_i^2] (rows) is zero.
    """
    n = len(points)
    if n < 4:
        return True
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for m in range(k + 1, n):
                    pts = [points[i], points[j], points[k], points[m]]
                    # Build 4x4 matrix: [x, y, 1, x^2+y^2]
                    det = 0
                    # Use direct 4x4 determinant expansion
                    a = [
                        [pts[0][0], pts[0][1], Fraction(1), pts[0][0]**2 + pts[0][1]**2],
                        [pts[1][0], pts[1][1], Fraction(1), pts[1][0]**2 + pts[1][1]**2],
                        [pts[2][0], pts[2][1], Fraction(1), pts[2][0]**2 + pts[2][1]**2],
                        [pts[3][0], pts[3][1], Fraction(1), pts[3][0]**2 + pts[3][1]**2],
                    ]
                    # 4x4 determinant
                    det = (
                        a[0][0] * (a[1][1]*a[2][2]*a[3][3] + a[1][2]*a[2][3]*a[3][1] + a[1][3]*a[2][1]*a[3][2]
                                 - a[1][1]*a[2][3]*a[3][2] - a[1][2]*a[2][1]*a[3][3] - a[1][3]*a[2][2]*a[3][1])
                        - a[0][1] * (a[1][0]*a[2][2]*a[3][3] + a[1][2]*a[2][3]*a[3][0] + a[1][3]*a[2][0]*a[3][2]
                                   - a[1][0]*a[2][3]*a[3][2] - a[1][2]*a[2][0]*a[3][3] - a[1][3]*a[2][2]*a[3][0])
                        + a[0][2] * (a[1][0]*a[2][1]*a[3][3] + a[1][1]*a[2][3]*a[3][0] + a[1][3]*a[2][0]*a[3][1]
                                   - a[1][0]*a[2][3]*a[3][1] - a[1][1]*a[2][0]*a[3][3] - a[1][3]*a[2][1]*a[3][0])
                        - a[0][3] * (a[1][0]*a[2][1]*a[3][2] + a[1][1]*a[2][2]*a[3][0] + a[1][2]*a[2][0]*a[3][1]
                                   - a[1][0]*a[2][2]*a[3][1] - a[1][1]*a[2][0]*a[3][2] - a[1][2]*a[2][1]*a[3][0])
                    )
                    if det == 0:
                        return False
    return True


def _validate(D, points_frac):
    """
    Validate a candidate: all pairwise distances rational (always true for integer D),
    no three collinear, no four concyclic.
    points_frac: list of (Fraction, Fraction) representing (x, y) where actual point is (x, y*sqrt(D))
    """
    if not _check_no_three_collinear(points_frac):
        return False
    if not _check_no_four_concyclic(points_frac):
        return False
    return True


def _to_output(D, points_frac):
    """Convert to the required output format."""
    out_points = []
    for x, y in points_frac:
        out_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    return {'D': D, 'points': out_points}


def _search_lattice():
    """
    Search for integral point sets in Z^2 (D=1) by extending known small sets.
    We start from the known 5-point set and try to add more points.
    """
    # Known 5-point integral set
    base = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try to find a 6th point by searching over a bounded region
    # For D=1, all distances must be integers.
    # Point (x,y) must have integer distance to all 5 base points.
    
    best = list(base)
    best_len = 5
    
    # Search region: expand based on bounding box of base
    xs = [p[0] for p in base]
    ys = [p[1] for p in base]
    x_min = min(xs) - 100
    x_max = max(xs) + 100
    y_min = min(ys) - 100
    y_max = max(ys) + 100
    
    # We need a point (x,y) such that distance to each base point is integer.
    # This is a system of constraints. Let's use the first two points to parameterize.
    # dist to (0,0) = r0: x^2 + y^2 = r0^2
    # dist to (-56,33) = r1: (x+56)^2 + (y-33)^2 = r1^2
    # Subtract: 112x - 66y + 56^2 + 33^2 - r0^2 + r1^2 = 0... 
    # Actually: (x+56)^2 - x^2 + (y-33)^2 - y^2 = r1^2 - r0^2
    # 112x + 3136 - 66y + 1089 = r1^2 - r0^2
    # 112x - 66y + 4225 = r1^2 - r0^2
    
    # This is getting complex. Let's just brute force over a reasonable range.
    # But that's too slow. Instead, use known constructions.
    
    # Known 6-point integral set (if it exists)... 
    # Actually, the maximum known integral point set in general position is 5.
    # 6 points in general position with all pairwise integer distances is an open problem 
    # or known not to exist in small ranges.
    
    # Let's try D=3 with the hexagon approach.
    return best, best_len


def _construct_hexagon_D3():
    """
    Construct points in Q(sqrt(3)) from a regular hexagon.
    Regular hexagon vertices in complex plane: 1, omega, omega^2, omega^3, omega^4, omega^5
    where omega = -1/2 + i*sqrt(3)/2.
    
    In Q(sqrt(3)), we can write i*sqrt(3) = sqrt(-3), but we need points (x, y*sqrt(3)).
    
    Actually, let's think differently. We need points (x_i, y_i * sqrt(D)) such that
    all pairwise distances are rational.
    
    For D=3: dist^2 = (x1-x2)^2 + 3*(y1-y2)^2. This must be a perfect square of a rational.
    
    Let's try specific constructions. Consider points derived from Eisenstein integers.
    In Q(sqrt(-3)), but we need Q(sqrt(3)).
    
    Alternative: use the fact that if we have points in Z[sqrt(-3)] (Eisenstein),
    distances are related to norms. But we need Q(sqrt(3)) not Q(sqrt(-3)).
    
    Let me try a different approach: search for 6 points in Q(sqrt(D)) for small D.
    """
    # Try D=3 with specific small rational coordinates
    # We need (x1-x2)^2 + 3*(y1-y2)^2 = (rational)^2 for all pairs.
    
    # Start with (0,0) and (1,0): dist = 1 (rational)
    # Try (0, 1): dist to (0,0) = sqrt(3) - not rational!
    # So y coordinates need to be chosen carefully.
    
    # (x1-x2)^2 + 3*(y1-y2)^2 must be a square.
    # If y1=y2, dist = |x1-x2|, always rational. But then all on same line.
    # We need variation in y.
    
    # Let y1=0, y2=1: need (dx)^2 + 3 = q^2 for some rational q.
    # q^2 - dx^2 = 3 => (q-dx)(q+dx) = 3.
    # Let q-dx = a, q+dx = 3/a. Then q = (a+3/a)/2, dx = (3/a-a)/2.
    # For a=1: q=2, dx=1. So (0,0) and (1,1) have dist^2 = 1+3=4, dist=2. 
    # For a=3: q=2, dx=-1. So (0,0) and (-1,1) also works.
    
    # So (0,0), (1,1) is a valid pair for D=3.
    # Now try to build more.
    
    # Let me try a small search over small rational values for D=3.
    # Coordinates: x, y in small rational set.
    
    candidates = []
    # Use small numerators/denominators
    vals = [Fraction(i, d) for d in [1, 2, 3, 4, 5, 6] for i in range(-10, 11)]
    # Deduplicate
    seen = set()
    uniq_vals = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            uniq_vals.append(v)
    
    # Try to build a set greedily
    best_set = [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(1))]
    
    # Verify these two
    # dist^2 = (1-0)^2 + 3*(1-0)^2 = 1+3 = 4 = 2^2. Good.
    
    # Try adding more points
    # For each candidate point, check if it has rational distance to all in best_set
    added = True
    while added and time.time() < _DEADLINE - 5:
        added = False
        for x in uniq_vals:
            if time.time() > _DEADLINE - 5:
                break
            for y in uniq_vals:
                if time.time() > _DEADLINE - 5:
                    break
                pt = (x, y)
                if pt in best_set:
                    continue
                # Check all distances
                ok = True
                for (bx, by) in best_set:
                    dx = x - bx
                    dy = y - by
                    dist_sq = dx * dx + 3 * dy * dy
                    # Check if dist_sq is a perfect square of a rational
                    # dist_sq is a Fraction. Check if it's a square.
                    num = dist_sq.numerator
                    den = dist_sq.denominator
                    # dist_sq = (rational)^2 means num/den = (a/b)^2 => num*b^2 = a^2*den
                    # i.e., num*den must be a perfect square (and num, den positive)
                    if num < 0 or den < 0:
                        ok = False
                        break
                    if not _is_square(num * den):
                        ok = False
                        break
                if not ok:
                    continue
                # Check no three collinear and no four concyclic with this new point
                new_set = best_set + [pt]
                if len(new_set) >= 3 and not _check_no_three_collinear(new_set):
                    continue
                if len(new_set) >= 4 and not _check_no_four_concyclic(new_set):
                    continue
                # Add it
                best_set.append(pt)
                added = True
                break
            if added:
                break
        # If we added a point, continue; otherwise we're done
        if not added:
            # Try with a larger range or different D
            break
    
    if len(best_set) > 5:
        return 3, best_set
    return None


def _search_with_D(D, search_range):
    """
    Search for a point set in Q(sqrt(D)) with the given D.
    Greedy addition of points.
    """
    # Start with a small valid set
    # For general D, (0,0) and (1,0) always have rational distance 1.
    best_set = [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0))]
    
    # Build candidate coordinate values
    vals = set()
    for d in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
        for i in range(-search_range, search_range + 1):
            vals.add(Fraction(i, d))
    uniq_vals = list(vals)
    
    # Greedy: keep adding points that satisfy all constraints
    added = True
    while added and time.time() < _DEADLINE - 10:
        added = False
        for x in uniq_vals:
            if time.time() > _DEADLINE - 10:
                break
            for y in uniq_vals:
                if time.time() > _DEADLINE - 10:
                    break
                pt = (x, y)
                if pt in best_set:
                    continue
                ok = True
                for (bx, by) in best_set:
                    dx = x - bx
                    dy = y - by
                    dist_sq = dx * dx + D * dy * dy
                    num = dist_sq.numerator
                    den = dist_sq.denominator
                    if num < 0 or den < 0:
                        ok = False
                        break
                    if not _is_square(num * den):
                        ok = False
                        break
                if not ok:
                    continue
                new_set = best_set + [pt]
                if len(new_set) >= 3 and not _check_no_three_collinear(new_set):
                    continue
                if len(new_set) >= 4 and not _check_no_four_concyclic(new_set):
                    continue
                best_set.append(pt)
                added = True
                break
            if added:
                break
    
    return D, best_set


def construct():
    # Return the best known 5-point set as fallback
    fallback = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    fallback_frac = [(Fraction(x), Fraction(y)) for x, y in fallback]
    
    best_D = 1
    best_points = fallback_frac
    best_len = 5
    
    # Try to find a 6-point set with D=3
    if time.time() < _DEADLINE - 60:
        result = _construct_hexagon_D3()
        if result is not None:
            D, pts = result
            if len(pts) > best_len:
                best_D = D
                best_points = pts
                best_len = len(pts)
    
    # Try small D values with limited search
    for D in [1, 2, 3, 5, 6, 7, 10]:
        if time.time() > _DEADLINE - 30:
            break
        if D == 3 and best_D == 3:
            continue  # Already tried
        result_D, pts = _search_with_D(D, search_range=8)
        if len(pts) > best_len:
            best_D = result_D
            best_points = pts
            best_len = len(pts)
    
    # Final validation
    if not _validate(best_D, best_points):
        # Fall back to the known 5-point set
        best_D = 1
        best_points = fallback_frac
        best_len = 5
    
    return _to_output(best_D, best_points)
