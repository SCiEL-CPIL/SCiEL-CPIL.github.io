"""
Search for integral point sets in general position (no 3 collinear, no 4 concyclic)
in the plane with points (x, y*sqrt(D)) for rational x, y.
"""
import time
import random
import math
from fractions import Fraction

DEADLINE = 145.0  # safety margin before 150s kill

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _sq_dist(p, q, D):
    """Squared distance between points p=(x1,y1), q=(x2,y2) in (x, y*sqrt(D)) space."""
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + D * dy * dy

def _dist_rational(p, q, D):
    """Check if squared distance is a perfect square (of a rational)."""
    d2 = _sq_dist(p, q, D)
    num = d2.numerator
    den = d2.denominator
    if _is_square(num) and _is_square(den):
        return True
    return False

def _no_three_collinear(points):
    """Check no three points are collinear using cross product = 0."""
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

def _concyclic_det(p1, p2, p3, p4, D):
    """Compute the 4x4 determinant for concyclicity test."""
    a1 = p1[0]*p1[0] + D*p1[1]*p1[1]
    a2 = p2[0]*p2[0] + D*p2[1]*p2[1]
    a3 = p3[0]*p3[0] + D*p3[1]*p3[1]
    a4 = p4[0]*p4[0] + D*p4[1]*p4[1]
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    x3, y3 = p3[0], p3[1]
    x4, y4 = p4[0], p4[1]
    return (a1*(x2*y3 - x2*y4 - x3*y2 + x3*y4 + x4*y2 - x4*y3)
          - x1*(a2*y3 - a2*y4 - a3*y2 + a3*y4 + a4*y2 - a4*y3)
          + y1*(a2*x3 - a2*x4 - a3*x2 + a3*x4 + a4*x2 - a4*x3)
          - (a2*x3*y4 - a2*x4*y3 - a3*x2*y4 + a3*x4*y2 + a4*x2*y3 - a4*x3*y2))

def _no_four_concyclic(points, D):
    """Check no four points are concyclic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _concyclic_det(points[i], points[j], points[k], points[l], D) == 0:
                        return False
    return True

def _all_distances_rational(points, D):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            if not _dist_rational(points[i], points[j], D):
                return False
    return True

def _is_valid(points, D):
    if len(points) < 3:
        return False
    if not _all_distances_rational(points, D):
        return False
    if not _no_three_collinear(points):
        return False
    if not _no_four_concyclic(points, D):
        return False
    return True

def _try_add_point(points, D, x, y):
    """Try to add point (x,y) to existing set. Returns True if valid."""
    new_pt = (Fraction(x), Fraction(y))
    # Check new point against all existing points for distance
    for p in points:
        if not _dist_rational(p, new_pt, D):
            return False
    # Check collinearity with pairs
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            x3, y3 = new_pt
            cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
            if cross == 0:
                return False
    # Check concyclicity with triples
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _concyclic_det(points[i], points[j], points[k], new_pt, D) == 0:
                    return False
    return True

def _systematic_add(points, D, deadline, max_range=500):
    """Systematically try to add points to the set."""
    current = list(points)
    best = current[:]
    best_len = len(current)
    
    for x in range(-max_range, max_range + 1):
        if time.time() > deadline:
            break
        for y in range(-max_range, max_range + 1):
            if time.time() > deadline:
                break
            if len(current) >= 12:
                return best
            if _try_add_point(current, D, x, y):
                current.append((Fraction(x), Fraction(y)))
                best = current[:]
                best_len = len(current)
                if best_len >= 8:
                    return best
    return best

def _randomized_search(points, D, deadline, max_coord=300):
    """Randomized search to find additional points."""
    current = list(points)
    best = current[:]
    best_len = len(current)
    
    # Generate a pool of candidate points
    candidates = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            if (x, y) not in [(p[0], p[1]) for p in points]:
                candidates.append((x, y))
    
    random.shuffle(candidates)
    
    for x, y in candidates:
        if time.time() > deadline:
            break
        if len(current) >= 12:
            break
        if _try_add_point(current, D, x, y):
            current.append((Fraction(x), Fraction(y)))
            best = current[:]
            best_len = len(current)
            if best_len >= 8:
                break
    
    return best

def _try_multiple_configs(deadline):
    """Try multiple initial configurations and search strategies."""
    best_points = []
    best_D = 1
    best_len = 0
    
    # Known 6-point integral set
    known_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63), (56, 33)]
    pts_frac = [(Fraction(x), Fraction(y)) for x, y in known_points]
    
    # Verify and fall back if needed
    if _is_valid(pts_frac, 1):
        if len(pts_frac) > best_len:
            best_points = pts_frac[:]
            best_len = len(best_points)
    else:
        known_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        pts_frac = [(Fraction(x), Fraction(y)) for x, y in known_points]
        if _is_valid(pts_frac, 1):
            best_points = pts_frac[:]
            best_len = len(best_points)
    
    # Try systematic extension
    if time.time() < deadline - 10 and len(best_points) > 0:
        extended = _systematic_add(best_points, 1, deadline - 5, max_range=300)
        if len(extended) > best_len:
            best_points = extended[:]
            best_len = len(best_points)
    
    # Try randomized search with different seeds
    for seed in range(10):
        if time.time() > deadline - 5:
            break
        random.seed(seed * 12345 + 67890)
        
        # Start from the best known set
        if len(best_points) > 0:
            start_points = best_points[:]
        else:
            start_points = [(Fraction(0), Fraction(0))]
        
        extended = _randomized_search(start_points, 1, deadline - 2, max_coord=200)
        if len(extended) > best_len:
            best_points = extended[:]
            best_len = len(best_points)
        
        # Also try starting fresh with random small sets
        if time.time() < deadline - 8:
            fresh = [(Fraction(0), Fraction(0))]
            for _ in range(3):
                if time.time() > deadline - 3:
                    break
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if (x, y) != (0, 0) and _try_add_point(fresh, 1, x, y):
                    fresh.append((Fraction(x), Fraction(y)))
            
            if len(fresh) >= 3:
                extended_fresh = _randomized_search(fresh, 1, deadline - 2, max_coord=200)
                if len(extended_fresh) > best_len:
                    best_points = extended_fresh[:]
                    best_len = len(best_points)
    
    return best_points, best_D, best_len

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_points, best_D, best_len = _try_multiple_configs(deadline)
    
    # Format result
    result_points = []
    for (x, y) in best_points:
        result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': best_D, 'points': result_points}
