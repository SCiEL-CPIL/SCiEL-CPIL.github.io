"""Improved program: constrained incremental search for integral point sets."""
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
    """Check if distance between p and q is rational (for D=1, dist2 must be a perfect square rational)."""
    d2 = _rational_dist2(p, q)
    # For D=1: need d2 to be a perfect square of a rational
    num, den = d2.numerator, d2.denominator
    if den == 1:
        return _is_square(num)
    # General case: d2 = (a/b)^2 means num*den must be a perfect square and num/den must be a square ratio
    # Simplify: check if num/den is a perfect square in rationals
    # num/den = (sqrt(num)/sqrt(den))^2, so num*den must be perfect square and gcd(num,den) handling
    # Actually: num/den is a perfect square rational iff num*den is a perfect square
    prod = num * den
    return _is_square(prod)

def _collinear(p, q, r):
    """Check if three points are collinear using cross product."""
    # (q-p) x (r-p) = 0
    dx1 = q[0] - p[0]
    dy1 = q[1] - p[1]
    dx2 = r[0] - p[0]
    dy2 = r[1] - p[1]
    return dx1 * dy2 - dy1 * dx2 == 0

def _concyclic(p, q, r, s):
    """Check if four points are concyclic using determinant of circle equation."""
    # Points (x_i, y_i) are concyclic iff the determinant is zero:
    # |x^2+y^2  x  y  1| = 0 for all four points
    def circ_det(a, b, c, d):
        def f(pt):
            return (pt[0]*pt[0] + pt[1]*pt[1], pt[0], pt[1], 1)
        A, B, C, D = f(a), f(b), f(c), f(d)
        # 4x4 determinant
        return (A[0]*(B[1]*(C[2]*D[3]-C[3]*D[2]) - B[2]*(C[1]*D[3]-C[3]*D[1]) + B[3]*(C[1]*D[2]-C[2]*D[1]))
              - A[1]*(B[0]*(C[2]*D[3]-C[3]*D[2]) - B[2]*(C[0]*D[3]-C[3]*D[0]) + B[3]*(C[0]*D[2]-C[2]*D[0]))
              + A[2]*(B[0]*(C[1]*D[3]-C[3]*D[1]) - B[1]*(C[0]*D[3]-C[3]*D[0]) + B[3]*(C[0]*D[1]-C[1]*D[0]))
              - A[3]*(B[0]*(C[1]*D[2]-C[2]*D[1]) - B[1]*(C[0]*D[2]-C[2]*D[0]) + B[2]*(C[0]*D[1]-C[1]*D[0])))
    return circ_det(p, q, r, s) == 0

def _check_valid(points, D=1):
    """Full validity check: all pairwise distances rational, no 3 collinear, no 4 concyclic."""
    n = len(points)
    if n < 3:
        return False
    # Check all pairwise distances
    for i in range(n):
        for j in range(i+1, n):
            if not _check_rational_dist(points[i], points[j], D):
                return False
    # Check no three collinear
    for combo in combinations(range(n), 3):
        if _collinear(points[combo[0]], points[combo[1]], points[combo[2]]):
            return False
    # Check no four concyclic
    for combo in combinations(range(n), 4):
        if _concyclic(points[combo[0]], points[combo[1]], points[combo[2]], points[combo[3]]):
            return False
    return True

def _generate_candidates_near(base_points, max_coord=200, max_candidates=5000):
    """Generate candidate integer points with rational distances to all base points."""
    candidates = []
    # Generate a grid of candidate points
    for x in range(-max_coord, max_coord+1):
        for y in range(-max_coord, max_coord+1):
            pt = (Fraction(x), Fraction(y))
            # Check if pt has rational distance to all base points
            valid = True
            for bp in base_points:
                if not _check_rational_dist(pt, bp, 1):
                    valid = False
                    break
            if valid:
                candidates.append(pt)
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break
    return candidates

def _search_incremental(starting_points, max_points=8, time_limit=130.0):
    """Incrementally add points, returning the largest valid set found."""
    start_time = time.time()
    best = list(starting_points)
    current = list(starting_points)
    
    for target_n in range(len(current)+1, max_points+1):
        if time.time() - start_time > time_limit:
            break
        
        # Generate candidates that have rational distance to all current points
        candidates = _generate_candidates_near(current, max_coord=150, max_candidates=3000)
        
        # Filter candidates: no collinearity with any pair in current
        filtered = []
        for cand in candidates:
            ok = True
            for i in range(len(current)):
                for j in range(i+1, len(current)):
                    if _collinear(cand, current[i], current[j]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                # Also check no concyclicity with any triple in current
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
            # Try with slightly larger coordinate range
            candidates2 = _generate_candidates_near(current, max_coord=250, max_candidates=5000)
            filtered2 = []
            for cand in candidates2:
                if cand in current:
                    continue
                ok = True
                for i in range(len(current)):
                    for j in range(i+1, len(current)):
                        if _collinear(cand, current[i], current[j]):
                            ok = False
                            break
                    if not ok:
                        break
                if ok and len(current) >= 3:
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
                break  # Can't extend further
    
    return best

def _search_from_multiple_starts(max_points=8, time_limit=130.0):
    """Try multiple starting configurations to find the largest valid set."""
    start_time = time.time()
    best = []
    
    # Known valid 5-point sets
    base_sets = [
        [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)),
         (Fraction(-16), Fraction(63)), (Fraction(0), Fraction(33)),
         (Fraction(16), Fraction(63))],
        [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(4)),
         (Fraction(0), Fraction(4)), (Fraction(3), Fraction(8)),
         (Fraction(6), Fraction(4))],
    ]
    
    for base in base_sets:
        if time.time() - start_time > time_limit:
            break
        # Verify base is valid
        if not _check_valid(base, 1):
            continue
        if len(base) > len(best):
            best = list(base)
        
        # Try to extend
        try:
            extended = _search_incremental(base, max_points=max_points, time_limit=time_limit)
            if len(extended) > len(best):
                best = extended
        except Exception:
            pass
    
    return best

def construct():
    start_time = time.time()
    
    # Try multiple starting configurations
    best = _search_from_multiple_starts(max_points=8, time_limit=130.0)
    
    # If we still only have 5 or fewer, try a direct search with larger coordinate range
    if len(best) <= 5 and time.time() - start_time < 100:
        # Try a direct search from a small valid set
        small_base = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(4)),
                      (Fraction(0), Fraction(4))]
        if _check_valid(small_base, 1):
            try:
                extended = _search_incremental(small_base, max_points=8, time_limit=120.0)
                if len(extended) > len(best):
                    best = extended
            except Exception:
                pass
    
    # Format output
    points_out = []
    for pt in best:
        x, y = pt
        points_out.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': 1, 'points': points_out}
# EVOLVE-BLOCK-END
