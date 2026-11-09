"""Improved program: multi-strategy search for 6+ point integral sets."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations
from math import isqrt

_DEADLINE = 140.0  # safety margin before 150s kill

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _is_square_frac(f):
    if f < 0:
        return False
    num, den = f.numerator, f.denominator
    return _is_square(num) and _is_square(den)

def _rational_dist2(p, q, D=1):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + D * dy * dy

def _check_rational_dist(p, q, D=1):
    d2 = _rational_dist2(p, q, D)
    return _is_square_frac(d2)

def _collinear(p, q, r):
    dx1 = q[0] - p[0]
    dy1 = q[1] - p[1]
    dx2 = r[0] - p[0]
    dy2 = r[1] - p[1]
    return dx1 * dy2 - dy1 * dx2 == 0

def _concyclic(p, q, r, s, D=1):
    def row(pt):
        return (pt[0]*pt[0] + D*pt[1]*pt[1], pt[0], pt[1], Fraction(1))
    A, B, C, Dv = row(p), row(q), row(r), row(s)
    det = (A[0]*(B[1]*(C[2]*Dv[3]-C[3]*Dv[2]) - B[2]*(C[1]*Dv[3]-C[3]*Dv[1]) + B[3]*(C[1]*Dv[2]-C[2]*Dv[1]))
         - A[1]*(B[0]*(C[2]*Dv[3]-C[3]*Dv[2]) - B[2]*(C[0]*Dv[3]-C[3]*Dv[0]) + B[3]*(C[0]*Dv[2]-C[2]*Dv[0]))
         + A[2]*(B[0]*(C[1]*Dv[3]-C[3]*Dv[1]) - B[1]*(C[0]*Dv[3]-C[3]*Dv[0]) + B[3]*(C[0]*Dv[1]-C[1]*Dv[0]))
         - A[3]*(B[0]*(C[1]*Dv[2]-C[2]*Dv[1]) - B[1]*(C[0]*Dv[2]-C[2]*Dv[0]) + B[2]*(C[0]*Dv[1]-C[1]*Dv[0])))
    return det == 0

def _check_valid(points, D=1):
    n = len(points)
    if n < 3:
        return False
    for i in range(n):
        for j in range(i+1, n):
            if not _check_rational_dist(points[i], points[j], D):
                return False
    for combo in combinations(range(n), 3):
        if _collinear(points[combo[0]], points[combo[1]], points[combo[2]]):
            return False
    for combo in combinations(range(n), 4):
        if _concyclic(points[combo[0]], points[combo[1]], points[combo[2]], points[combo[3]], D):
            return False
    return True

def _quick_add_check(cand, current, D):
    """Fast check if cand can be added to current set."""
    n = len(current)
    # Check distances
    for i in range(n):
        if not _check_rational_dist(cand, current[i], D):
            return False
    # Check collinearity
    for i in range(n):
        for j in range(i+1, n):
            if _collinear(cand, current[i], current[j]):
                return False
    # Check concyclicity
    if n >= 3:
        for combo in combinations(range(n), 3):
            if _concyclic(cand, current[combo[0]], current[combo[1]], current[combo[2]], D):
                return False
    return True

def _smart_search(base_points, D, max_points=8, time_limit=120.0):
    """Efficient incremental search with multiple strategies."""
    start_time = time.time()
    current = list(base_points)
    best = list(current)
    
    # Pre-compute candidate pool with small coordinates
    candidate_pool = []
    for x in range(-50, 51):
        for y in range(-50, 51):
            pt = (Fraction(x), Fraction(y))
            candidate_pool.append(pt)
    
    for _ in range(max_points - len(current)):
        if time.time() - start_time > time_limit:
            break
        
        # Try to find a valid next point
        found = False
        for cand in candidate_pool:
            if time.time() - start_time > time_limit:
                break
            if cand in current:
                continue
            if _quick_add_check(cand, current, D):
                current.append(cand)
                found = True
                if len(current) > len(best):
                    best = list(current)
                break
        
        if not found:
            # Try larger coordinates
            for x in range(-100, 101):
                if time.time() - start_time > time_limit:
                    break
                for y in range(-100, 101):
                    if time.time() - start_time > time_limit:
                        break
                    cand = (Fraction(x), Fraction(y))
                    if cand in current:
                        continue
                    if _quick_add_check(cand, current, D):
                        current.append(cand)
                        found = True
                        if len(current) > len(best):
                            best = list(current)
                        break
                if found:
                    break
        
        if not found:
            break
    
    return best

def _try_known_bases():
    """Try multiple known base configurations."""
    bases = [
        # Base 1: Small integer points
        [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(4)), (Fraction(0), Fraction(4))],
        # Base 2: Slightly larger
        [(Fraction(0), Fraction(0)), (Fraction(4), Fraction(3)), (Fraction(0), Fraction(3))],
        # Base 3: Symmetric
        [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))],
        # Base 4: 5-point base from literature
        [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)),
         (Fraction(-16), Fraction(63)), (Fraction(0), Fraction(33)),
         (Fraction(16), Fraction(63))],
        # Base 5: Another 5-point
        [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(4)),
         (Fraction(0), Fraction(4)), (Fraction(3), Fraction(8)),
         (Fraction(6), Fraction(4))],
    ]
    
    best_result = None
    start_time = time.time()
    
    for i, base in enumerate(bases):
        if time.time() - start_time > 100:
            break
        try:
            if _check_valid(base, 1):
                result = _smart_search(base, 1, max_points=8, time_limit=min(20, 100 - (time.time() - start_time)))
                if best_result is None or len(result) > len(best_result):
                    best_result = result
        except Exception:
            pass
    
    return best_result

def construct():
    start_time = time.time()
    best_points = None
    
    # Strategy 1: Try known bases with smart search
    try:
        result = _try_known_bases()
        if result is not None and len(result) >= 3:
            best_points = result
    except Exception:
        pass
    
    # Strategy 2: If we have 5+, try to extend further
    if best_points is not None and len(best_points) >= 5 and time.time() - start_time < 100:
        try:
            extended = _smart_search(best_points, 1, max_points=8, time_limit=min(30, 100 - (time.time() - start_time)))
            if len(extended) > len(best_points):
                best_points = extended
        except Exception:
            pass
    
    # Fallback: minimal valid set
    if best_points is None or len(best_points) < 3:
        best_points = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(4)), (Fraction(0), Fraction(4))]
    
    # Verify final result
    if not _check_valid(best_points, 1):
        best_points = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(4)), (Fraction(0), Fraction(4))]
    
    # Format output
    points_out = []
    for pt in best_points:
        x, y = pt
        points_out.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': 1, 'points': points_out}
# EVOLVE-BLOCK-END
