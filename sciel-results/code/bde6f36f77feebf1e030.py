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
                cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                if cross == 0:
                    return False
    return True

def _check_no_four_concyclic(points):
    """
    Check no four points are concyclic using exact rational arithmetic.
    """
    n = len(points)
    if n < 4:
        return True
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for m in range(k + 1, n):
                    pts = [points[i], points[j], points[k], points[m]]
                    a = [
                        [pts[0][0], pts[0][1], Fraction(1), pts[0][0]**2 + pts[0][1]**2],
                        [pts[1][0], pts[1][1], Fraction(1), pts[1][0]**2 + pts[1][1]**2],
                        [pts[2][0], pts[2][1], Fraction(1), pts[2][0]**2 + pts[2][1]**2],
                        [pts[3][0], pts[3][1], Fraction(1), pts[3][0]**2 + pts[3][1]**2],
                    ]
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
    if not _check_no_three_collinear(points_frac):
        return False
    if not _check_no_four_concyclic(points_frac):
        return False
    return True

def _to_output(D, points_frac):
    out_points = []
    for x, y in points_frac:
        out_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    return {'D': D, 'points': out_points}

def _search_for_points(D, max_coords=10, max_den=4, target_size=6):
    """
    Greedy search for a set of 'target_size' points in Q(sqrt(D)).
    """
    # Start with a small valid set
    # (0,0) and (1,0) is always valid for any D
    best_set = [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0))]
    
    # Generate candidate points
    candidates = []
    seen = set()
    for den in range(1, max_den + 1):
        for num in range(-max_coords, max_coords + 1):
            for den_y in range(1, max_den + 1):
                for num_y in range(-max_coords, max_coords + 1):
                    pt = (Fraction(num, den), Fraction(num_y, den_y))
                    if pt not in seen:
                        seen.add(pt)
                        candidates.append(pt)
    
    # Sort candidates by distance from origin to prefer "dense" sets?
    # Or just iterate.
    
    added = True
    while added and time.time() < _DEADLINE - 10:
        added = False
        for pt in candidates:
            if time.time() > _DEADLINE - 10:
                break
            if pt in best_set:
                continue
            
            ok = True
            for (bx, by) in best_set:
                dx = pt[0] - bx
                dy = pt[1] - by
                dist_sq = dx * dx + D * dy * dy
                num = dist_sq.numerator
                den = dist_sq.denominator
                if num < 0:
                    ok = False
                    break
                # Check if num/den is a perfect square
                # num/den = (a/b)^2 => num*b^2 = a^2*den => num*den = (a*b)^2 ? No.
                # num/den = k^2 => num = k^2 * den.
                # Let k = p/q. num/den = p^2/q^2 => num*q^2 = p^2*den.
                # This implies num*den is a perfect square.
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
            if len(best_set) >= target_size:
                break
    
    return best_set

def construct():
    # Fallback 5-point set
    fallback = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    fallback_frac = [(Fraction(x), Fraction(y)) for x, y in fallback]
    
    best_D = 1
    best_points = fallback_frac
    best_len = 5
    
    # Try to find a 6-point set
    # Prioritize D values that are known to have rich structure
    D_values = [3, 1, 5, 6, 7, 10, 12, 13, 15, 21]
    
    for D in D_values:
        if time.time() > _DEADLINE - 20:
            break
        # Search with increasing coordinate limits
        # Start small for speed
        for limit in [5, 10, 20]:
            if time.time() > _DEADLINE - 10:
                break
            pts = _search_for_points(D, max_coords=limit, max_den=3, target_size=6)
            if len(pts) >= 6:
                if _validate(D, pts):
                    best_D = D
                    best_points = pts
                    best_len = len(pts)
                    break
            # If we found 5 points but not 6, keep searching if time permits
            # But 5 is our baseline, so we only care about 6+
    
    # Final validation
    if not _validate(best_D, best_points):
        best_D = 1
        best_points = fallback_frac
        best_len = 5
    
    return _to_output(best_D, best_points)
