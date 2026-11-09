"""Improved program: circle-intersection based search for integral point sets."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations
from math import isqrt

_DEADLINE = 140.0  # safety margin before 150s kill

def _is_square(n):
    """Check if non-negative integer n is a perfect square."""
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _rational_dist2(p, q):
    """Squared distance between two points as a Fraction."""
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy

def _check_rational_dist(p, q, D=1):
    """Check if distance between p and q is rational."""
    d2 = _rational_dist2(p, q)
    num, den = d2.numerator, d2.denominator
    prod = num * den
    return _is_square(prod)

def _collinear(p, q, r):
    """Check if three points are collinear using cross product."""
    dx1 = q[0] - p[0]
    dy1 = q[1] - p[1]
    dx2 = r[0] - p[0]
    dy2 = r[1] - p[1]
    return dx1 * dy2 - dy1 * dx2 == 0

def _concyclic(p, q, r, s):
    """Check if four points are concyclic using determinant of circle equation."""
    def f(pt):
        return (pt[0]*pt[0] + pt[1]*pt[1], pt[0], pt[1], 1)
    A, B, C, D = f(p), f(q), f(r), f(s)
    # 4x4 determinant
    det = (A[0]*(B[1]*(C[2]*D[3]-C[3]*D[2]) - B[2]*(C[1]*D[3]-C[3]*D[1]) + B[3]*(C[1]*D[2]-C[2]*D[1]))
          - A[1]*(B[0]*(C[2]*D[3]-C[3]*D[2]) - B[2]*(C[0]*D[3]-C[3]*D[0]) + B[3]*(C[0]*D[2]-C[2]*D[0]))
          + A[2]*(B[0]*(C[1]*D[3]-C[3]*D[1]) - B[1]*(C[0]*D[3]-C[3]*D[0]) + B[3]*(C[0]*D[1]-C[1]*D[0]))
          - A[3]*(B[0]*(C[1]*D[2]-C[2]*D[1]) - B[1]*(C[0]*D[2]-C[2]*D[0]) + B[2]*(C[0]*D[1]-C[1]*D[0])))
    return det == 0

def _check_valid(points, D=1):
    """Full validity check: all pairwise distances rational, no 3 collinear, no 4 concyclic."""
    n = len(points)
    if n < 3:
        return False
    for i in range(n):
        for j in range(i+1, n):
            if points[i][0] == points[j][0] and points[i][1] == points[j][1]:
                return False
            if not _check_rational_dist(points[i], points[j], D):
                return False
    for combo in combinations(range(n), 3):
        if _collinear(points[combo[0]], points[combo[1]], points[combo[2]]):
            return False
    for combo in combinations(range(n), 4):
        if _concyclic(points[combo[0]], points[combo[1]], points[combo[2]], points[combo[3]]):
            return False
    return True

def _circle_intersections(p1, p2, r1, r2):
    """Find intersection points of two circles. Returns list of (Fraction, Fraction) or empty."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    d2 = dx*dx + dy*dy
    if d2 == 0:
        return []
    d = d2.sqrt() if hasattr(d2, 'sqrt') else None
    # For integer/rational points, we need d to be rational
    # Check if d2 is a perfect square rational
    num, den = d2.numerator, d2.denominator
    prod = num * den
    if not _is_square(prod):
        return []
    # Compute d as Fraction
    r = isqrt(prod)
    # d = sqrt(d2) = sqrt(num/den) = sqrt(num*den)/den
    d_val = Fraction(r, den)
    
    # Standard circle intersection formula
    a = (r1*r1 - r2*r2 + d2) / (2 * d_val)
    h2 = r1*r1 - a*a
    if h2 < 0:
        return []
    h_num, h_den = h2.numerator, h2.denominator
    h_prod = h_num * h_den
    if not _is_square(h_prod):
        return []
    h_r = isqrt(h_prod)
    h_val = Fraction(h_r, h_den)
    
    # Midpoint offset
    mx = p1[0] + a * dx / d_val
    my = p1[1] + a * dy / d_val
    
    # Perpendicular direction
    px = -dy / d_val * h_val
    py = dx / d_val * h_val
    
    pt1 = (mx + px, my + py)
    pt2 = (mx - px, my - py)
    
    # Check if these are rational points
    results = []
    for pt in [pt1, pt2]:
        x, y = pt
        if isinstance(x, Fraction) and isinstance(y, Fraction):
            results.append(pt)
    return results

def _generate_candidates_from_intersections(current, max_radius=200):
    """Generate candidate points that have integer distance to at least 2 current points."""
    candidates = set()
    n = len(current)
    
    for i in range(n):
        for j in range(i+1, n):
            p1, p2 = current[i], current[j]
            d12 = _rational_dist2(p1, p2)
            d12_num, d12_den = d12.numerator, d12.denominator
            d12_prod = d12_num * d12_den
            if not _is_square(d12_prod):
                continue
            d12_val = Fraction(isqrt(d12_prod), d12_den)
            
            # Try integer distances r1, r2
            for r1 in range(1, max_radius+1):
                for r2 in range(1, max_radius+1):
                    # Quick check: triangle inequality
                    if r1 + r2 < d12_val or abs(r1 - r2) > d12_val:
                        continue
                    pts = _circle_intersections(p1, p2, Fraction(r1), Fraction(r2))
                    for pt in pts:
                        candidates.add(pt)
    
    return list(candidates)

def _search_incremental(starting_points, max_points=8, time_limit=130.0):
    """Incrementally add points using circle intersections."""
    start_time = time.time()
    best = list(starting_points)
    current = list(starting_points)
    
    for target_n in range(len(current)+1, max_points+1):
        if time.time() - start_time > time_limit:
            break
        
        # Generate candidates from circle intersections
        candidates = _generate_candidates_from_intersections(current, max_radius=150)
        
        # Filter candidates
        filtered = []
        for cand in candidates:
            if cand in current:
                continue
            ok = True
            # Check rational distance to all current points
            for bp in current:
                if not _check_rational_dist(cand, bp, 1):
                    ok = False
                    break
            if not ok:
                continue
            # Check no collinearity
            for i in range(len(current)):
                for j in range(i+1, len(current)):
                    if _collinear(cand, current[i], current[j]):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            # Check no concyclicity
            if len(current) >= 3:
                for combo in combinations(range(len(current)), 3):
                    if _concyclic(cand, current[combo[0]], current[combo[1]], current[combo[2]]):
                        ok = False
                        break
            if ok:
                filtered.append(cand)
        
        # Try to find one candidate that extends the set
        found = False
        for cand in filtered:
            if time.time() - start_time > time_limit:
                break
            new_set = current + [cand]
            if _check_valid(new_set, 1):
                current = new_set
                found = True
                if len(current) > len(best):
                    best = list(current)
                break
        
        if not found:
            # Try with larger radius
            candidates2 = _generate_candidates_from_intersections(current, max_radius=250)
            filtered2 = []
            for cand in candidates2:
                if cand in current:
                    continue
                ok = True
                for bp in current:
                    if not _check_rational_dist(cand, bp, 1):
                        ok = False
                        break
                if not ok:
                    continue
                for i in range(len(current)):
                    for j in range(i+1, len(current)):
                        if _collinear(cand, current[i], current[j]):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                if len(current) >= 3:
                    for combo in combinations(range(len(current)), 3):
                        if _concyclic(cand, current[combo[0]], current[combo[1]], current[combo[2]]):
                            ok = False
                            break
                if ok:
                    filtered2.append(cand)
            
            for cand in filtered2:
                if time.time() - start_time > time_limit:
                    break
                new_set = current + [cand]
                if _check_valid(new_set, 1):
                    current = new_set
                    found = True
                    if len(current) > len(best):
                        best = list(current)
                    break
            
            if not found:
                break
    
    return best

def construct():
    start_time = time.time()
    
    # Known valid 5-point set
    base_points = [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)),
                   (Fraction(-16), Fraction(63)), (Fraction(0), Fraction(33)),
                   (Fraction(16), Fraction(63))]
    
    best = base_points
    
    # Try to extend from the base set
    try:
        extended = _search_incremental(base_points, max_points=8, time_limit=130.0)
        if len(extended) > len(best):
            best = extended
    except Exception:
        pass
    
    # If we still only have 5, try a different starting configuration
    if len(best) <= 5 and time.time() - start_time < 100:
        # Try another known small set
        alt_base = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(4)),
                    (Fraction(0), Fraction(4)), (Fraction(3), Fraction(8)),
                    (Fraction(6), Fraction(4))]
        if _check_valid(alt_base, 1):
            try:
                extended_alt = _search_incremental(alt_base, max_points=8, time_limit=120.0)
                if len(extended_alt) > len(best):
                    best = extended_alt
            except Exception:
                pass
    
    # Format output
    points_out = []
    for pt in best:
        x, y = pt
        points_out.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': 1, 'points': points_out}
# EVOLVE-BLOCK-END
