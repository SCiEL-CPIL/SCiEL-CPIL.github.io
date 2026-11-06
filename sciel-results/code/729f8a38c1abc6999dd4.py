import time
import math
import random

def _is_square(n):
    """Check if n is a perfect square."""
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _check_integer_distances(points):
    """Check if all pairwise distances between integer points are integers."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            d2 = dx * dx + dy * dy
            if not _is_square(d2):
                return False
    return True

def _no_three_collinear(points):
    """Check no three points are collinear using integer cross products."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Cross product of (Pj - Pi) and (Pk - Pi)
                dx1 = points[j][0] - points[i][0]
                dy1 = points[j][1] - points[i][1]
                dx2 = points[k][0] - points[i][0]
                dy2 = points[k][1] - points[i][1]
                cross = dx1 * dy2 - dy1 * dx2
                if cross == 0:
                    return False
    return True

def _no_four_concyclic(points):
    """Check no four points are concyclic using integer determinants."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for m in range(k + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    x4, y4 = points[m]
                    
                    s1 = x1*x1 + y1*y1
                    s2 = x2*x2 + y2*y2
                    s3 = x3*x3 + y3*y3
                    s4 = x4*x4 + y4*y4
                    
                    # Determinant of 4x4 matrix:
                    # | s1, x1, y1, 1 |
                    # | s2, x2, y2, 1 |
                    # | s3, x3, y3, 1 |
                    # | s4, x4, y4, 1 |
                    
                    # Expansion along the last column
                    det = s1 * (x2*(y3 - y4) - y2*(x3 - x4) + (x3*y4 - y3*x4)) \
                        - x1 * (s2*(y3 - y4) - y2*(s3 - s4) + (s3*y4 - y3*s4)) \
                        + y1 * (s2*(x3 - x4) - x2*(s3 - s4) + (s3*x4 - x3*s4)) \
                        + 1 * (s2*(x3*y4 - y3*x4) - x2*(s3*y4 - y3*s4) + y2*(s3*x4 - x3*s4))
                    
                    if det == 0:
                        return False
    return True

def _valid_set(points):
    """Full validation for a set of integer points."""
    if not _check_integer_distances(points):
        return False
    if not _no_three_collinear(points):
        return False
    if not _no_four_concyclic(points):
        return False
    return True

def _try_add_point(base_points, candidate, deadline):
    """Check if candidate can be added to base_points to form a valid set."""
    # Quick checks
    if candidate in base_points:
        return False
    
    # Check integer distances to all base points
    for px, py in base_points:
        dx = candidate[0] - px
        dy = candidate[1] - py
        d2 = dx*dx + dy*dy
        if not _is_square(d2):
            return False
            
    # Full check on the new set
    new_set = base_points + [candidate]
    
    # Check collinearity and concyclicity for the new set
    # Since base_points is valid, we only need to check triples/quads involving the new point
    # But for simplicity and safety, we can run the full check on the new set if it's small
    # However, full check is O(n^4). For n=6, it's 15 quads. Fast enough.
    if not _no_three_collinear(new_set):
        return False
    if not _no_four_concyclic(new_set):
        return False
        
    return True

def _search_for_extension(base_points, deadline):
    """Search for a 6th point to extend the 5-point base set."""
    # Determine bounding box
    xs = [p[0] for p in base_points]
    ys = [p[1] for p in base_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Expand search radius
    # Start with a moderate radius and increase if time permits
    # The known 5-point set has span ~100. A 6th point might be close or far.
    # Let's try radii: 50, 100, 200, 500, 1000
    
    radii = [50, 100, 200, 500, 1000, 2000]
    
    for r in radii:
        if time.time() > deadline:
            break
            
        start_x = min_x - r
        end_x = max_x + r
        start_y = min_y - r
        end_y = max_y + r
        
        # To avoid checking too many points, we can be smart about it, 
        # but a simple loop is okay if we break early.
        # 1000x1000 is 1M points. Checking 5 distances each is 5M isqrt calls. 
        # isqrt is fast, but 5M might take a few seconds in Python.
        # Let's limit the grid size or use a smarter search.
        
        # Actually, let's just iterate. Python is slow, so we need to be careful.
        # 2000x2000 is 4M points. Too slow for Python loops in 150s.
        # We should stick to smaller radii or use a different strategy.
        
        # Strategy: Iterate x, then y.
        for x in range(start_x, end_x + 1):
            if time.time() > deadline:
                return None
            for y in range(start_y, end_y + 1):
                if time.time() > deadline:
                    return None
                cand = (x, y)
                if _try_add_point(base_points, cand, deadline):
                    return cand
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Known 5-point sets with integer distances
    # Set 1: (0,0), (-56,33), (-16,63), (0,33), (16,63)
    set1 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Set 2: Another known set? 
    # Let's try to generate or use a second one if available.
    # For now, let's focus on extending Set 1.
    # If we find a 6th point, we return it.
    
    best_set = set1
    best_count = 5
    
    # Verify Set 1 is valid (should be)
    if _valid_set(set1):
        # Try to extend
        new_point = _search_for_extension(set1, deadline)
        if new_point:
            best_set = set1 + [new_point]
            best_count = 6
    
    # If we didn't find a 6th point, try other strategies?
    # Random search for 6 points from scratch?
    # This is less likely to succeed than extending a known 5-point set.
    
    # Fallback: Return the best set found
    # Ensure points are in the required format: [[x_num, x_den], [y_num, y_den]]
    points_out = []
    for x, y in best_set:
        points_out.append([[x, 1], [y, 1]])
        
    return {'D': 1, 'points': points_out}
