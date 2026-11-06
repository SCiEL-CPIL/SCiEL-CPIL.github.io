"""
Search for integral point sets in general position (no 3 collinear, no 4 concyclic)
in the plane with points (x, y*sqrt(D)) for rational x, y.
"""
import time
import random
from math import gcd
from fractions import Fraction

DEADLINE = 110.0  # safety margin before 120s kill

def _is_square(n):
    if n < 0:
        return False
    if n == 0:
        return True
    r = int(n ** 0.5)
    for c in (r - 1, r, r + 1):
        if c >= 0 and c * c == n:
            return True
    return False

def _sq_dist_int(p, q, D):
    """Squared distance between integer points. Returns int."""
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + D * dy * dy

def _dist_rational_int(p, q, D):
    """Check if squared distance is a perfect square for integer coords."""
    d2 = _sq_dist_int(p, q, D)
    return _is_square(d2)

def _no_three_collinear_int(points):
    """Check no three points are collinear. Points are (x,y) int tuples."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                if cross == 0:
                    return False
    return True

def _no_four_concyclic_int(points, D):
    """Check no four points are concyclic for integer points in (x, y*sqrt(D)) space."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                    x4, y4 = p4
                    x1, y1 = p1
                    x2, y2 = p2
                    x3, y3 = p3
                    a1 = x1*x1 + D*y1*y1 - x4*x4 - D*y4*y4
                    a2 = x2*x2 + D*y2*y2 - x4*x4 - D*y4*y4
                    a3 = x3*x3 + D*y3*y3 - x4*x4 - D*y4*y4
                    b1 = x1 - x4
                    b2 = x2 - x4
                    b3 = x3 - x4
                    c1 = y1 - y4
                    c2 = y2 - y4
                    c3 = y3 - y4
                    det = a1*(b2*c3 - b3*c2) - b1*(a2*c3 - a3*c2) + c1*(a2*b3 - a3*b2)
                    if det == 0:
                        return False
    return True

def _all_dist_rational_int(points, D):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            if not _dist_rational_int(points[i], points[j], D):
                return False
    return True

def _is_valid_int(points, D):
    """Validate integer-coordinate point set."""
    if len(points) < 3:
        return False
    if not _all_dist_rational_int(points, D):
        return False
    if not _no_three_collinear_int(points):
        return False
    if not _no_four_concyclic_int(points, D):
        return False
    return True

def _find_candidates_around(existing, D, max_coord=200):
    """Find integer points at integer distance from all existing points."""
    n = len(existing)
    if n == 0:
        return []
    candidates = None
    for (ex, ey) in existing:
        valid_for_this = set()
        for dx in range(-max_coord, max_coord + 1):
            for dy in range(-max_coord, max_coord + 1):
                if dx == 0 and dy == 0:
                    continue
                d2 = dx * dx + D * dy * dy
                if _is_square(d2):
                    px = ex + dx
                    py = ey + dy
                    valid_for_this.add((px, py))
        if candidates is None:
            candidates = valid_for_this
        else:
            candidates = candidates & valid_for_this
    if candidates is None:
        return []
    existing_set = set(existing)
    candidates = candidates - existing_set
    return list(candidates)

def _try_extend(points, D, max_coord=150, deadline=None):
    """Try to extend a valid point set by adding one more point."""
    if deadline is not None and time.time() > deadline:
        return points
    candidates = _find_candidates_around(points, D, max_coord)
    random.shuffle(candidates)
    for cand in candidates:
        if deadline is not None and time.time() > deadline:
            break
        new_set = points + [cand]
        if _is_valid_int(new_set, D):
            return new_set
    return points

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Known 5-point integral set (D=1)
    known5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    if not _is_valid_int(known5, 1):
        known5 = [(0, 0), (56, 0), (0, 33), (40, 45), (48, 55)]
        if not _is_valid_int(known5, 1):
            known5 = [(0, 0), (3, 4), (0, 5)]
    
    best_points = known5[:]
    best_D = 1
    best_len = len(best_points)
    
    # Phase 1: Try to extend known set
    current = best_points[:]
    for _ in range(3):
        if time.time() > deadline - 20:
            break
        new_set = _try_extend(current, 1, max_coord=200, deadline=deadline - 20)
        if len(new_set) > len(current):
            current = new_set
            best_points = current[:]
            best_len = len(current)
        else:
            break
    
    # Phase 2: Random search
    if time.time() < deadline - 10:
        for attempt in range(50000):
            if time.time() > deadline - 10:
                break
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            cand = best_points + [(x, y)]
            if _is_valid_int(cand, 1):
                best_points = cand[:]
                best_len = len(cand)
                if best_len >= 8:
                    break
        
        if best_len < 6:
            for attempt in range(30000):
                if time.time() > deadline - 5:
                    break
                n = random.randint(4, 6)
                pts = []
                for _ in range(n):
                    x = random.randint(-80, 80)
                    y = random.randint(-80, 80)
                    pts.append((x, y))
                if _is_valid_int(pts, 1) and len(pts) > best_len:
                    best_points = pts[:]
                    best_len = len(pts)
                    if best_len >= 7:
                        break
    
    # Phase 3: Try D=2, D=3
    if time.time() < deadline - 5:
        for D_val in [2, 3]:
            if time.time() > deadline - 3:
                break
            for attempt in range(10000):
                if time.time() > deadline - 3:
                    break
                n = random.randint(3, 5)
                pts = []
                for _ in range(n):
                    x = random.randint(-40, 40)
                    y = random.randint(-40, 40)
                    pts.append((x, y))
                if _is_valid_int(pts, D_val) and len(pts) > best_len:
                    best_points = pts[:]
                    best_D = D_val
                    best_len = len(pts)
                if best_len >= 7:
                    break
    
    # Format result
    result_points = []
    for (x, y) in best_points:
        result_points.append([[x, 1], [y, 1]])
    
    return {'D': best_D, 'points': result_points}
