"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140  # safety margin before 150s kill


def _line_key(p1, p2):
    """Canonical key for line through p1=(x1,y1) and p2=(x2,y2)."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (0, 0, x1)
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0:
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)


def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for c in line_counts.values() if c == 1)


def _is_valid(points):
    """Check that points are 12 distinct integer points with |x|,|y| <= 1e9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        if len(p) != 2:
            return False
        x, y = p
        if not (isinstance(x, int) and isinstance(y, int)):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    # Check not all collinear
    if len(seen) < 3:
        return False
    p0 = points[0]
    p1 = points[1]
    p2 = points[2]
    # Cross product
    if (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]) != 0:
        return True
    # Check all points collinear with p0, p1
    for i in range(3, 12):
        pi = points[i]
        if (p1[0] - p0[0]) * (pi[1] - p0[1]) - (p1[1] - p0[1]) * (pi[0] - p0[0]) != 0:
            return True
    return False


def _boroczky_config_12():
    """
    Construct a Böröczky-like configuration for 12 points.
    
    Theoretical minimum for ordinary lines for n=12 is n/2 = 6.
    The Böröczky construction for n=2k consists of:
    - k points on a conic (e.g., circle)
    - k points at infinity (directions of tangents to the conic at the first k points)
    
    In integer coordinates, we approximate 'points at infinity' by placing them
    far away in the direction of the tangent. Specifically, for a point P on the
    circle, the tangent direction is perpendicular to the radius.
    If P = (cos theta, sin theta), the tangent direction is (-sin theta, cos theta).
    We place the 'infinity' point at M * (-sin theta, cos theta) for large M.
    
    To ensure integer coordinates and avoid collisions, we use a large radius R for
    the circle points and a large offset M for the infinity points.
    """
    R = 1000  # Radius of the circle
    M = 10**8 # Scale for 'infinity' points
    
    points = []
    
    # 6 points on the circle x^2 + y^2 = R^2
    # We choose angles that yield integer or near-integer coordinates, 
    # or just use rational approximations scaled to integers.
    # Actually, any distinct points on the circle work. 
    # Let's use 6 distinct integer points on x^2+y^2=R^2 if possible, 
    # or just use trigonometric values rounded to integers.
    # Rounding might break the 'conic' property slightly, but the structure
    # of having 6 'close' points and 6 'far' points in tangent directions
    # should still reduce ordinary lines significantly compared to random.
    
    # Better: Use exact integer points on a circle if possible?
    # x^2 + y^2 = R^2.
    # For R=1000, there are many solutions.
    # Let's just pick 6 distinct angles.
    angles = [2 * math.pi * i / 6 for i in range(6)]
    
    circle_pts = []
    for i, ang in enumerate(angles):
        x = int(round(R * math.cos(ang)))
        y = int(round(R * math.sin(ang)))
        # Ensure distinctness
        if (x, y) not in [(p[0], p[1]) for p in circle_pts]:
            circle_pts.append([x, y])
            
    # If we didn't get 6 distinct (unlikely for R=1000), pad
    while len(circle_pts) < 6:
        # Add random integer points on circle approx
        ang = random.uniform(0, 2*math.pi)
        x = int(round(R * math.cos(ang)))
        y = int(round(R * math.sin(ang)))
        if (x, y) not in [(p[0], p[1]) for p in circle_pts]:
            circle_pts.append([x, y])
            
    points.extend(circle_pts)
    
    # 6 points 'at infinity' corresponding to tangents at the 6 circle points
    inf_pts = []
    for i in range(6):
        x, y = circle_pts[i]
        # Tangent direction at (x,y) on circle x^2+y^2=R^2 is (-y, x) or (y, -x)
        # The line is x*x_i + y*y_i = R^2.
        # The direction vector is (-y_i, x_i).
        # We place the point far away in this direction.
        # To avoid collinearity with other infinity points, we vary the distance slightly
        # or ensure the directions are distinct.
        # The directions (-y_i, x_i) for i=0..5 are distinct for distinct points on circle.
        
        dx = -y
        dy = x
        
        # Normalize to avoid huge numbers? No, we want them far.
        # But we must stay within 10^9.
        # M = 10^8 is safe.
        # However, if dx, dy are small, the point is close to origin?
        # No, we want the point to be at distance ~M.
        # Let's scale (dx, dy) to have length M.
        length = math.sqrt(dx*dx + dy*dy)
        if length == 0:
            continue
        sx = int(round(M * dx / length))
        sy = int(round(M * dy / length))
        
        # Add a small offset to ensure distinctness if directions are similar?
        # The directions are 60 deg apart, so they are distinct.
        # But rounding might cause collisions.
        if (sx, sy) not in [(p[0], p[1]) for p in inf_pts]:
            inf_pts.append([sx, sy])
        else:
            # Collision, perturb slightly
            inf_pts.append([sx + 1, sy])
            
    points.extend(inf_pts)
    
    # Ensure all points are distinct and valid
    seen = set()
    unique_points = []
    for p in points:
        if (p[0], p[1]) not in seen:
            seen.add((p[0], p[1]))
            unique_points.append(p)
            
    # If we lost points due to collision, replace with random valid points
    # This is a fallback, shouldn't happen with M=10^8
    while len(unique_points) < 12:
        x = random.randint(-10**8, 10**8)
        y = random.randint(-10**8, 10**8)
        if (x, y) not in seen:
            seen.add((x, y))
            unique_points.append([x, y])
            
    return unique_points[:12]


def _local_search_improve(points, time_limit):
    """
    Perform local search to reduce ordinary lines further.
    Focus on moving points to create more collinearities.
    """
    current_pts = [p[:] for p in points]
    current_count = count_ordinary_lines(current_pts)
    best_pts = [p[:] for p in current_pts]
    best_count = current_count
    
    # Identify lines with >2 points
    # We want to increase counts of lines from 2 to 3, or 3 to 4, etc.
    # Strategy: Pick a point on an ordinary line, and move it to lie on a line
    # with 2 other points that are not on the original ordinary line?
    # Or simply: Pick a point, move it to a position that creates a new collinearity.
    
    # A simpler heuristic: 
    # 1. Find all lines with exactly 2 points (ordinary lines).
    # 2. For each ordinary line (A, B), try to move A or B to a position C 
    #    such that C, A, B are collinear with another point D? No, that breaks the line.
    #    We want to eliminate the ordinary line.
    #    To eliminate line AB, we need a third point on AB.
    #    So, pick a third point C (not A or B) and move it to lie on AB.
    
    for _ in range(5000):
        if time.time() > time_limit:
            break
            
        # Find ordinary lines
        line_counts = {}
        for i in range(12):
            for j in range(i+1, 12):
                key = _line_key(current_pts[i], current_pts[j])
                if key not in line_counts:
                    line_counts[key] = []
                line_counts[key].append((i, j))
        
        ordinary_lines = [v for v in line_counts.values() if len(v) == 1]
        
        if not ordinary_lines:
            break # Perfect? Or no ordinary lines found?
            
        # Pick a random ordinary line
        idx_pair = random.choice(ordinary_lines)[0]
        i, j = idx_pair
        A = current_pts[i]
        B = current_pts[j]
        
        # Try to move a point k (not i, j) to lie on line AB
        candidates = [k for k in range(12) if k != i and k != j]
        k = random.choice(candidates)
        C = current_pts[k]
        
        # Project C onto line AB?
        # Or just move C to a new integer point on line AB that is distinct from A and B.
        # Line AB: y = mx + c (if not vertical)
        # We need an integer point (x', y') on the line such that (x', y') != A and (x', y') != B.
        
        x1, y1 = A
        x2, y2 = B
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0:
            # Vertical line x = x1
            # Pick a new y
            # Avoid y1, y2, and y-coords of other points on this line (none, it's ordinary)
            # Also avoid y-coords of other points to not create new distinct points? 
            # We just need it to be a valid point.
            new_y = random.randint(-10**8, 10**8)
            if new_y == y1 or new_y == y2:
                new_y += 1
            new_point = [x1, new_y]
        else:
            # Parametric: x = x1 + t*dx, y = y1 + t*dy
            # We want t such that (x, y) is integer.
            # Since dx, dy are integers, any integer t works.
            # We need t such that new point is distinct from A (t!=0) and B (t!=1).
            # Also distinct from other points.
            t = random.randint(-10, 10)
            while t in [0, 1]:
                t = random.randint(-10, 10)
            new_x = x1 + t * dx
            new_y = y1 + t * dy
            
            # Check bounds
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue
                
            new_point = [new_x, new_y]
            
            # Check distinctness from all other points
            valid = True
            for m in range(12):
                if m == k:
                    continue
                if current_pts[m][0] == new_x and current_pts[m][1] == new_y:
                    valid = False
                    break
            if not valid:
                continue
        
        # Try the move
        old_point = current_pts[k][:]
        current_pts[k] = new_point
        
        new_count = count_ordinary_lines(current_pts)
        
        if new_count <= current_count:
            current_count = new_count
            if new_count < best_count:
                best_count = new_count
                best_pts = [p[:] for p in current_pts]
        else:
            # Revert
            current_pts[k] = old_point
            
    return best_pts, best_count


def construct():
    """Find 12 integer points minimizing ordinary lines."""
    global _START_TIME, _DEADLINE
    
    time_limit = _DEADLINE - 5
    
    # Start with Böröczky configuration
    best_pts = _boroczky_config_12()
    best_count = count_ordinary_lines(best_pts)
    
    if not _is_valid(best_pts):
        # Fallback to grid if construction failed
        best_pts = []
        for x in range(4):
            for y in range(3):
                best_pts.append([x, y])
        best_count = count_ordinary_lines(best_pts)
        
    # Improve with local search
    best_pts, best_count = _local_search_improve(best_pts, time_limit)
    
    # Ensure valid
    if not _is_valid(best_pts):
        best_pts = []
        for x in range(4):
            for y in range(3):
                best_pts.append([x, y])
    
    return best_pts
# EVOLVE-BLOCK-END
