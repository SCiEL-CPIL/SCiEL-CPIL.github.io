"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    # First, return the known 5-point set as a fallback
    best_result = _try_known_5_points()
    
    if time.time() > deadline - 5:
        return best_result

    # Try to find 6+ points
    # Strategy 1: Search for 6 integer points (D=1)
    result_6 = _search_6_points_d1()
    if result_6 is not None:
        return result_6

    if time.time() > deadline - 2:
        return best_result

    # Strategy 2: Search for 6+ points with D > 1
    result_d = _search_with_D()
    if result_d is not None and _count_points(result_d) > _count_points(best_result):
        return result_d
    
    # Strategy 3: Try to find 7+ points with D > 1
    result_7 = _search_7_points_d()
    if result_7 is not None and _count_points(result_7) > _count_points(best_result):
        return result_7

    return best_result

def _count_points(result):
    return len(result['points'])

def _try_known_5_points():
    # Five integer points with integer pairwise distances, no three collinear and no four concyclic
    points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _check_pairwise_distances(points, D):
    """Check all pairwise distances are rational (squared distances are perfect squares when scaled)"""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Distance squared = (x1-x2)^2 + D*(y1-y2)^2
            dx = x1 - x2
            dy = y1 - y2
            dist_sq = dx*dx + D*dy*dy
            if not _is_square(dist_sq):
                return False
    return True

def _no_three_collinear(points):
    """Check no three points are collinear"""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                # Cross product (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1) should be != 0
                cross = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
                if cross == 0:
                    return False
    return True

def _no_four_concyclic(points, D):
    """Check no four points are concyclic"""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _four_points_concyclic(points[i], points[j], points[k], points[l], D):
                        return False
    return True

def _four_points_concyclic(p1, p2, p3, p4, D):
    """Check if four points are concyclic using determinant method"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    # For points (x, y*sqrt(D)), use transformed coordinates
    # Circle equation: x^2 + D*y^2 + Ax + By + C = 0
    # Set up linear system for A, B, C from three points, check fourth
    # From point 1: x1^2 + D*y1^2 + A*x1 + B*y1 + C = 0
    # From point 2: x2^2 + D*y2^2 + A*x2 + B*y2 + C = 0
    # From point 3: x3^2 + D*y3^2 + A*x3 + B*y3 + C = 0
    
    # Subtract eq1 from eq2 and eq3 to eliminate C:
    # A*(x2-x1) + B*(y2-y1) = -(x2^2 - x1^2) - D*(y2^2 - y1^2)
    # A*(x3-x1) + B*(y3-y1) = -(x3^2 - x1^2) - D*(y3^2 - y1^2)
    
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = -(x2*x2 - x1*x1) - D*(y2*y2 - y1*y1)
    
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = -(x3*x3 - x1*x1) - D*(y3*y3 - y1*y1)
    
    det = a1*b2 - a2*b1
    if det == 0:
        # Points 1,2,3 collinear, degenerate case
        return False
    
    # Solve for A and B
    # A = (c1*b2 - c2*b1) / det
    # B = (a1*c2 - a2*c1) / det
    
    # Instead of dividing, check if point 4 satisfies the circle
    # A*x4 + B*y4 + C = 0 where C = -(x1^2 + D*y1^2 + A*x1 + B*y1)
    # Substitute: A*x4 + B*y4 - x1^2 - D*y1^2 - A*x1 - B*y1 = 0
    # A*(x4-x1) + B*(y4-y1) = x1^2 + D*y1^2 - x4^2 - D*y4^2
    
    rhs = x1*x1 + D*y1*y1 - x4*x4 - D*y4*y4
    
    # A = (c1*b2 - c2*b1) / det
    # B = (a1*c2 - a2*c1) / det
    # So: (c1*b2 - c2*b1)*(x4-x1) + (a1*c2 - a2*c1)*(y4-y1) = det*rhs
    
    lhs = (c1*b2 - c2*b1)*(x4-x1) + (a1*c2 - a2*c1)*(y4-y1)
    rhs_scaled = det * rhs
    
    return lhs == rhs_scaled

def _validate(points, D):
    """Full validation of a point set"""
    if len(points) < 3:
        return False
    # Check distinctness
    seen = set()
    for p in points:
        if p in seen:
            return False
        seen.add(p)
    # Check pairwise distances
    if not _check_pairwise_distances(points, D):
        return False
    # Check no three collinear
    if not _no_three_collinear(points):
        return False
    # Check no four concyclic
    if not _no_four_concyclic(points, D):
        return False
    return True

def _search_6_points_d1():
    """Search for 6 integer points with integer pairwise distances, no 3 collinear, no 4 concyclic"""
    # Use a greedy/heuristic approach: start with known 5-point set, try adding 6th
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try adding points in a bounded region
    for x in range(-100, 101):
        for y in range(-100, 101):
            new_point = (x, y)
            candidate = base_points + [new_point]
            if _validate(candidate, 1):
                return {'D': 1, 'points': [[[px, 1], [py, 1]] for px, py in candidate]}
    
    # Try different base configurations
    # Search more broadly
    candidates = []
    for x1 in range(-20, 21):
        for y1 in range(-20, 21):
            candidates.append((x1, y1))
    
    # Try random subsets and check
    import random
    random.seed(42)
    
    for _ in range(500):
        subset = random.sample(candidates, 6)
        if _validate(subset, 1):
            return {'D': 1, 'points': [[[px, 1], [py, 1]] for px, py in subset]}
    
    return None

def _search_with_D():
    """Search for point sets with D > 1"""
    # For D > 1, we can have more flexibility
    # Try small squarefree D values
    for D in [2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 46, 47, 51, 53, 55, 57, 58, 59, 61, 62, 65, 66, 67, 69, 70, 71, 73, 74, 77, 78, 79, 82, 83, 85, 86, 87, 89, 93, 94, 95]:
        # Try simple configurations
        # Points at (0,0), (a,0), (b,c), etc. with rational coordinates
        # For D>1, y coordinates can be rational
        
        # Try lattice-like points
        points = [(0, 0)]
        
        # Add points systematically
        for x in range(-10, 11):
            for y in range(-10, 11):
                if x == 0 and y == 0:
                    continue
                new_point = (x, y)
                candidate = points + [new_point]
                if len(candidate) >= 3 and _validate(candidate, D):
                    if len(candidate) > len(points):
                        points = candidate
                        if len(points) >= 6:
                            return {'D': D, 'points': [[[px, 1], [py, 1]] for px, py in points]}
        
        # Try with different starting points
        for start_x in range(-5, 6):
            for start_y in range(-5, 6):
                points = [(start_x, start_y)]
                for x in range(-10, 11):
                    for y in range(-10, 11):
                        if x == start_x and y == start_y:
                            continue
                        new_point = (x, y)
                        candidate = points + [new_point]
                        if len(candidate) >= 3 and _validate(candidate, D):
                            if len(candidate) > len(points):
                                points = candidate
                                if len(points) >= 6:
                                    return {'D': D, 'points': [[[px, 1], [py, 1]] for px, py in points]}
    
    return None

def _search_7_points_d():
    """Search for 7+ points with D > 1"""
    # Try small squarefree D values
    for D in [2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 46, 47, 51, 53, 55, 57, 58, 59, 61, 62, 65, 66, 67, 69, 70, 71, 73, 74, 77, 78, 79, 82, 83, 85, 86, 87, 89, 93, 94, 95]:
        # Try lattice-like points
        points = [(0, 0)]
        
        # Add points systematically
        for x in range(-15, 16):
            for y in range(-15, 16):
                if x == 0 and y == 0:
                    continue
                new_point = (x, y)
                candidate = points + [new_point]
                if len(candidate) >= 3 and _validate(candidate, D):
                    if len(candidate) > len(points):
                        points = candidate
                        if len(points) >= 7:
                            return {'D': D, 'points': [[[px, 1], [py, 1]] for px, py in points]}
        
        # Try with different starting points
        for start_x in range(-8, 9):
            for start_y in range(-8, 9):
                points = [(start_x, start_y)]
                for x in range(-15, 16):
                    for y in range(-15, 16):
                        if x == start_x and y == start_y:
                            continue
                        new_point = (x, y)
                        candidate = points + [new_point]
                        if len(candidate) >= 3 and _validate(candidate, D):
                            if len(candidate) > len(points):
                                points = candidate
                                if len(points) >= 7:
                                    return {'D': D, 'points': [[[px, 1], [py, 1]] for px, py in points]}
    
    return None
# EVOLVE-BLOCK-END
