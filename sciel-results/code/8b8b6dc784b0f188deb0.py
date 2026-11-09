"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt
from fractions import Fraction

_START_TIME = time.time()
_TIMEOUT = 110.0  # safety margin before 120s wall-clock limit

def _time_left():
    return _TIMEOUT - (time.time() - _START_TIME)

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _check_distances(points):
    """Check all pairwise distances are rational (integers for D=1)."""
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xi - xj
            dy = yi - yj
            d2 = dx * dx + dy * dy
            if not _is_square(d2):
                return False
    return True

def _check_collinear(points):
    """Check no three points are collinear."""
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            for k in range(j + 1, n):
                xk, yk = points[k]
                if (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi) == 0:
                    return False
    return True

def _check_concyclic(points):
    """Check no four points are concyclic using determinant method."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                # Find circle through i, j, k
                dx2 = x2 - x1
                dy2 = y2 - y1
                rhs2 = x2*x2 + y2*y2 - x1*x1 - y1*y1
                dx3 = x3 - x1
                dy3 = y3 - y1
                rhs3 = x3*x3 + y3*y3 - x1*x1 - y1*y1
                det = dx2 * dy3 - dx3 * dy2
                if det == 0:
                    continue
                A_num = rhs2 * dy3 - rhs3 * dy2
                B_num = dx2 * rhs3 - dx3 * rhs2
                C_num = det * (x1*x1 + y1*y1) - A_num * x1 - B_num * y1
                for l in range(k + 1, n):
                    xl, yl = points[l]
                    lhs = det * (xl*xl + yl*yl)
                    rhs = A_num * xl + B_num * yl + C_num
                    if lhs == rhs:
                        return False
    return True

def _is_valid(points):
    if len(points) < 3:
        return False
    if not _check_distances(points):
        return False
    if not _check_collinear(points):
        return False
    if not _check_concyclic(points):
        return False
    return True

def _format_points(points):
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}

def _get_candidate_points(points, max_range=150):
    """Generate candidate points that could potentially be added."""
    n = len(points)
    existing = set(points)
    candidates = []
    
    # Compute bounding box
    min_x = min(p[0] for p in points) - max_range
    max_x = max(p[0] for p in points) + max_range
    min_y = min(p[1] for p in points) - max_range
    max_y = max(p[1] for p in points) + max_range
    
    # Generate candidates in expanding rings around centroid
    cx = sum(p[0] for p in points) // n
    cy = sum(p[1] for p in points) // n
    
    # Priority: closer to centroid first, then by distance to existing points
    seen = set()
    for r in range(1, max_range + 1):
        if _time_left() < 5.0:
            break
        # Points on circle of radius r around centroid
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx*dx + dy*dy != r*r:
                    continue
                x, y = cx + dx, cy + dy
                if (x, y) in existing or (x, y) in seen:
                    continue
                seen.add((x, y))
                candidates.append((x, y))
    
    return candidates

def _quick_check_candidate(points, cand):
    """Quick necessary checks before full validation."""
    x, y = cand
    # Check distance to all existing points
    for px, py in points:
        dx = x - px
        dy = y - py
        d2 = dx*dx + dy*dy
        if not _is_square(d2):
            return False
    return True

def _local_search(points, max_candidates=2000):
    """Try to add one more point by searching lattice positions."""
    if _time_left() < 3.0:
        return None
    
    candidates = _get_candidate_points(points)
    candidates = candidates[:max_candidates]
    
    for cand in candidates:
        if _time_left() < 2.0:
            break
        # Quick check first
        if not _quick_check_candidate(points, cand):
            continue
        # Full check
        new_points = points + [cand]
        if _is_valid(new_points):
            return new_points
    return None

def _try_multiple_seeds():
    """Try multiple starting configurations."""
    seeds = [
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        [(0, 0), (3, 4), (8, 6), (15, 8), (20, 12)],
        [(0, 0), (5, 12), (12, 5), (15, 8), (20, 21)],
    ]
    
    best = None
    for seed in seeds:
        if _time_left() < 10.0:
            break
        if not _is_valid(seed):
            continue
        
        current = list(seed)
        # Try to extend
        for target in range(6, 13):
            if _time_left() < 5.0:
                break
            result = _local_search(current)
            if result is not None:
                current = result
            else:
                break
        
        if best is None or len(current) > len(best):
            best = current
    
    return best

def construct():
    # Start with known 5-point configuration
    initial_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # First try to extend the initial configuration
    current = list(initial_points)
    for target in range(6, 13):
        if _time_left() < 5.0:
            break
        result = _local_search(current)
        if result is not None:
            current = result
        else:
            break
    
    best = current
    
    # Try multiple seeds if we have time
    if _time_left() > 20.0:
        alt_best = _try_multiple_seeds()
        if alt_best is not None and len(alt_best) > len(best):
            best = alt_best
    
    return _format_points(best)
# EVOLVE-BLOCK-END
