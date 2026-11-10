"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
from math import gcd, isqrt

def construct():
    """Search for integral point sets with no three collinear and no four concyclic."""
    t_start = time.time()
    DEADLINE = t_start + 140  # 10s safety margin
    
    def time_ok():
        return time.time() < DEADLINE
    
    def is_square(n):
        if n < 0:
            return False
        r = isqrt(n)
        return r * r == n
    
    def check_collinear(p1, p2, p3):
        """Check if three points are collinear using exact arithmetic."""
        (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
        # Cross product: (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0
    
    def check_concyclic(p1, p2, p3, p4):
        """Check if four points are concyclic using exact arithmetic."""
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = p1, p2, p3, p4
        # Use the determinant of the circle equation
        # x^2 + y^2 + Dx + Ey + F = 0
        # For point (xi, yi): xi^2 + yi^2 + D*xi + E*yi + F = 0
        # We can use the determinant form:
        # |x1 y1 x1^2+y1^2 1|
        # |x2 y2 x2^2+y2^2 1|
        # |x3 y3 x3^2+y3^2 1| = 0
        # |x4 y4 x4^2+y4^2 1|
        a1, b1, c1 = x1, y1, x1*x1 + y1*y1
        a2, b2, c2 = x2, y2, x2*x2 + y2*y2
        a3, b3, c3 = x3, y3, x3*x3 + y3*y3
        a4, b4, c4 = x4, y4, x4*x4 + y4*y4
        
        # Compute determinant using expansion
        # det = a1*(b2*(c3*1 - 1*c4) - c2*(b3*1 - 1*b4) + 1*(b3*c4 - c3*b4))
        #     - b1*(a2*(c3*1 - 1*c4) - c2*(a3*1 - 1*a4) + 1*(a3*c4 - c3*a4))
        #     + c1*(a2*(b3*1 - 1*b4) - b2*(a3*1 - 1*a4) + 1*(a3*b4 - b3*a4))
        #     - 1*(a2*(b3*c4 - c3*b4) - b2*(a3*c4 - c3*a4) + c2*(a3*b4 - b3*a4))
        
        det = (a1 * (b2 * (c3 - c4) - c2 * (b3 - b4) + (b3 * c4 - c3 * b4))
             - b1 * (a2 * (c3 - c4) - c2 * (a3 - a4) + (a3 * c4 - c3 * a4))
             + c1 * (a2 * (b3 - b4) - b2 * (a3 - a4) + (a3 * b4 - b3 * a4))
             - (a2 * (b3 * c4 - c3 * b4) - b2 * (a3 * c4 - c3 * a4) + c2 * (a3 * b4 - b3 * a4)))
        
        return det == 0
    
    def validate(points):
        """Validate a set of points: all distances rational, no three collinear, no four concyclic."""
        n = len(points)
        if n < 3:
            return False
        # Check all pairwise distances are integers
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                d2 = dx*dx + dy*dy
                if not is_square(d2):
                    return False
        # Check no three collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if check_collinear(points[i], points[j], points[k]):
                        return False
        # Check no four concyclic
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if check_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True
    
    def try_add_point(points, candidate):
        """Try to add a candidate point to the set."""
        new_points = points + [candidate]
        return validate(new_points)
    
    # Start with known good 5-point set
    best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Search for more points
    search_range = 100
    found_improvement = False
    
    # Try to add points one at a time
    while time_ok():
        current_size = len(best_points)
        if current_size >= 8:
            break
        
        # Generate candidate points in a region around existing points
        candidates = []
        for x in range(-search_range, search_range + 1):
            for y in range(-search_range, search_range + 1):
                candidates.append((x, y))
        
        random.shuffle(candidates)
        
        improved = False
        for cand in candidates:
            if not time_ok():
                break
            # Check if candidate is distinct from all existing points
            if any(cand == p for p in best_points):
                continue
            # Try to add
            if try_add_point(best_points, cand):
                best_points.append(cand)
                improved = True
                found_improvement = True
                search_range = min(search_range + 20, 300)
                break  # Restart with larger range
        
        if not improved:
            # Try different search strategies
            # Strategy 1: Points near the convex hull
            if len(best_points) >= 3:
                # Try points just outside the convex hull
                for p in best_points:
                    for dx in range(-10, 11):
                        for dy in range(-10, 11):
                            cand = (p[0] + dx, p[1] + dy)
                            if any(cand == q for q in best_points):
                                continue
                            if try_add_point(best_points, cand):
                                best_points.append(cand)
                                improved = True
                                found_improvement = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                # Try symmetric extensions
                for p in list(best_points):
                    # Reflect across axes
                    for cand in [(p[0], -p[1]), (-p[0], p[1]), (-p[0], -p[1])]:
                        if any(cand == q for q in best_points):
                            continue
                        if try_add_point(best_points, cand):
                            best_points.append(cand)
                            improved = True
                            found_improvement = True
                            break
                    if improved:
                        break
                if not improved:
                    # Give up on this iteration
                    break
    
    # Format the result
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
# EVOLVE-BLOCK-END
