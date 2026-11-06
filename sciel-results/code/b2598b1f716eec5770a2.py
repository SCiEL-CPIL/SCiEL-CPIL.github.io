"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from fractions import Fraction

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _dist_sq(p, q):
    """Squared distance between two points (x, y) as integer."""
    return (p[0] - q[0])**2 + (p[1] - q[1])**2

def _check_no_three_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                if ax*(by - cy) + bx*(cy - ay) + cx*(ay - by) == 0:
                    return False
    return True

def _check_no_four_concyclic(points):
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    ax, ay = points[i]
                    bx, by = points[j]
                    cx, cy = points[k]
                    dx, dy = points[l]
                    # Four points are concyclic iff the determinant is zero
                    # Using the circle equation: x^2 + y^2 + Dx + Ey + F = 0
                    # We can check if the determinant of the 4x4 matrix is zero
                    # | ax^2+ay^2  ax  ay  1 |
                    # | bx^2+by^2  bx  by  1 |
                    # | cx^2+cy^2  cx  cy  1 |
                    # | dx^2+dy^2  dx  dy  1 |
                    a2 = ax*ax + ay*ay
                    b2 = bx*bx + by*by
                    c2 = cx*cx + cy*cy
                    d2 = dx*dx + dy*dy
                    # Compute determinant using expansion
                    # For efficiency, we can use a simpler check
                    # The points are concyclic iff the power of each point w.r.t. the circle through any 3 is the same
                    # But let's just compute the determinant
                    det = (a2*(bx*(cy*1 - 1*dy) - by*(cx*1 - 1*dx) + 1*(cx*dy - cy*dx)) -
                           ax*(b2*(cy*1 - 1*dy) - by*(c2*1 - 1*d2) + 1*(c2*dy - d2*dx)) +
                           ay*(b2*(cx*1 - 1*dx) - bx*(c2*1 - 1*d2) + 1*(c2*dx - d2*cx)) -
                           1*(b2*(cx*dy - cy*dx) - bx*(c2*dy - d2*dx) + bx*(c2*dy - cy*d2)))
                    # This is getting complex, let me use a different approach
                    # Actually, let me just use the standard formula
                    # Four points (x1,y1), (x2,y2), (x3,y3), (x4,y4) are concyclic iff:
                    # | x1^2+y1^2  x1  y1  1 |
                    # | x2^2+y2^2  x2  y2  1 | = 0
                    # | x3^2+y3^2  x3  y3  1 |
                    # | x4^2+y4^2  x4  y4  1 |
                    # Let me compute this determinant properly
                    # Using row operations to simplify
                    # Actually, for small coordinates, direct computation is fine
                    # Let me use a helper function
                    det_val = _det4(a2, ax, ay, 1,
                                    b2, bx, by, 1,
                                    c2, cx, cy, 1,
                                    d2, dx, dy, 1)
                    if det_val == 0:
                        return False
    return True

def _det4(a1,b1,c1,d1, a2,b2,c2,d2, a3,b3,c3,d3, a4,b4,c4,d4):
    return (a1*(b2*(c3*d4 - d3*c4) - c2*(b3*d4 - d3*b4) + d2*(b3*c4 - c3*b4)) -
            b1*(a2*(c3*d4 - d3*c4) - c2*(a3*d4 - d3*a4) + d2*(a3*c4 - c3*a4)) +
            c1*(a2*(b3*d4 - d3*b4) - b2*(a3*d4 - d3*a4) + d2*(a3*b4 - b3*a4)) -
            d1*(a2*(b3*c4 - c3*b4) - b2*(a3*c4 - c3*a4) + c2*(a3*b4 - b3*a4)))

def _all_distances_integer(points):
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            ds = _dist_sq(points[i], points[j])
            if not _is_square(ds):
                return False
    return True

def _validate(points):
    if len(points) < 3:
        return False
    # Check distinctness
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if points[i] == points[j]:
                return False
    if not _all_distances_integer(points):
        return False
    if not _check_no_three_collinear(points):
        return False
    if not _check_no_four_concyclic(points):
        return False
    return True

def _try_extend(current_points, max_points, deadline):
    """Try to extend the current set with more points using local search."""
    if len(current_points) >= max_points:
        return current_points
    if time.time() > deadline:
        return current_points
    
    # Generate candidate points in a reasonable range
    # Use the existing points to guide the search
    best_set = list(current_points)
    
    # Try adding points near the existing configuration
    # First, find the bounding box
    min_x = min(p[0] for p in current_points)
    max_x = max(p[0] for p in current_points)
    min_y = min(p[1] for p in current_points)
    max_y = max(p[1] for p in current_points)
    
    # Expand the search range
    range_expand = max(100, (max_x - min_x) + (max_y - min_y))
    
    # Try to add points one at a time
    for _ in range(100):  # Limit iterations
        if time.time() > deadline:
            break
        if len(best_set) >= max_points:
            break
            
        # Try random candidates in the expanded range
        found = False
        for _ in range(1000):
            if time.time() > deadline:
                break
            # Generate a candidate point
            cx = min_x - range_expand + (max_x - min_x + 2*range_expand) * (hash((id(best_set), _)) % 10000) // 10000
            cy = min_y - range_expand + (max_y - min_y + 2*range_expand) * (hash((id(best_set), _+1)) % 10000) // 10000
            candidate = (cx, cy)
            
            # Check if this point can be added
            new_set = best_set + [candidate]
            if _validate(new_set):
                best_set = new_set
                found = True
                break
        
        if not found:
            break
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with the known 5-point configuration
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try to extend to 6, 7, 8 points
    current = list(base_points)
    
    for target in range(6, 9):
        if time.time() > deadline:
            break
        extended = _try_extend(current, target, deadline)
        if len(extended) > len(current):
            current = extended
    
    # Return the best set found
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in current]}
# EVOLVE-BLOCK-END
