import time
import math
import random
from itertools import combinations

def _count_ordinary(points):
    """Count the number of lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            a = dy
            b = -dx
            c = -a * x1 - b * y1
            g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
            if g > 0:
                a //= g
                b //= g
                c //= g
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
            key = (a, b, c)
            if key not in lines:
                lines[key] = []
            lines[key].append(i)
    
    ordinary = 0
    for key, pts in lines.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _valid(points):
    """Check if points are valid: 12 distinct integer points, |x|,|y| <= 10^9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        if not isinstance(p, list) or len(p) != 2:
            return False
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    
    # Check not all collinear
    if len(points) < 3:
        return True
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    if cross == 0:
        for i in range(3, 12):
            xi, yi = points[i]
            cross_i = (x2 - x1) * (yi - y1) - (y2 - y1) * (xi - x1)
            if cross_i != 0:
                return True
        return False
    return True

def _try_move(points, idx, new_x, new_y):
    """Try moving point idx to (new_x, new_y). Returns (new_points, improvement) if better, else None."""
    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
        return None
    
    old_x, old_y = points[idx]
    if old_x == new_x and old_y == new_y:
        return None
    
    # Check if new position collides with another point
    for j in range(12):
        if j == idx:
            continue
        if points[j][0] == new_x and points[j][1] == new_y:
            return None
    
    new_points = [p[:] for p in points]
    new_points[idx] = [new_x, new_y]
    
    if not _valid(new_points):
        return None
    
    count = _count_ordinary(new_points)
    return new_points, count

def construct():
    """
    Construct 12 integer points with a minimized number of ordinary lines.
    """
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Initialize with a 3x4 grid
    points_grid = [[x, y] for x in range(4) for y in range(3)]
    
    # Another seed: 4x3 grid (same)
    # Another seed: "Star" or other
    # Let's try a seed based on 3 lines of 4
    # (0,0), (1,0), (2,0), (3,0)
    # (0,1), (1,1), (2,1), (3,1)
    # (0,2), (1,2), (2,2), (3,2)
    # This is the grid.
    
    # Try a different seed:
    # 3 points on x-axis, 3 on y-axis, 3 on x=y, 3 on x=-y?
    # (1,0), (2,0), (3,0)
    # (0,1), (0,2), (0,3)
    # (1,1), (2,2), (3,3)
    # (1,-1), (2,-2), (3,-3)
    points_star = [
        [1, 0], [2, 0], [3, 0],
        [0, 1], [0, 2], [0, 3],
        [1, 1], [2, 2], [3, 3],
        [1, -1], [2, -2], [3, -3]
    ]
    
    candidates = [points_grid, points_star]
    best_points = None
    best_count = float('inf')
    
    for pts in candidates:
        if not _valid(pts):
            continue
        c = _count_ordinary(pts)
        if c < best_count:
            best_count = c
            best_points = [p[:] for p in pts]
    
    current_points = best_points
    current_count = best_count
    
    random.seed(42)
    
    # Local Search
    # We will iterate until deadline
    # To avoid getting stuck, we use a simple hill climbing with random restarts if stuck
    stuck_count = 0
    
    while time.time() < deadline:
        improved = False
        
        # Generate smart candidates for each point
        for i in range(12):
            if time.time() > deadline:
                break
                
            x, y = current_points[i]
            
            candidates = set()
            
            # 1. Small perturbations
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    candidates.add((x + dx, y + dy))
            
            # 2. Smart moves: Align with lines defined by other points
            # For each pair of other points (j, k), consider the line through them.
            # We want to place i on this line.
            # The line equation is Ax + By + C = 0.
            # We look for integer solutions (nx, ny) near (x, y) or within bounds.
            # To keep it fast, we only check a small range of "steps" along the line?
            # Or just check if there's an integer point on the line within a small box around (x,y).
            
            for j in range(12):
                if j == i:
                    continue
                for k in range(j + 1, 12):
                    if k == i:
                        continue
                    
                    x1, y1 = current_points[j]
                    x2, y2 = current_points[k]
                    
                    # Line through (x1,y1) and (x2,y2)
                    dx_l = x2 - x1
                    dy_l = y2 - y1
                    
                    # If points are same, skip (shouldn't happen)
                    if dx_l == 0 and dy_l == 0:
                        continue
                    
                    # Find integer points on this line near (x, y)
                    # Parametric: (x1 + t*dx_l, y1 + t*dy_l)
                    # We need t such that x1 + t*dx_l is close to x.
                    # If dx_l != 0, t = (x - x1) / dx_l
                    # We check integer t values in a small range.
                    
                    # To be efficient, we just check a few candidate t values
                    # that would put the point within a small distance from (x,y).
                    
                    # Approximate t
                    if dx_l != 0:
                        t_approx = (x - x1) / dx_l
                    elif dy_l != 0:
                        t_approx = (y - y1) / dy_l
                    else:
                        continue
                        
                    # Check t values around t_approx
                    # t must be such that the point is integer.
                    # Since x1, y1, dx_l, dy_l are integers,
                    # x1 + t*dx_l is integer iff t*dx_l is integer.
                    # If we restrict t to integers, it's always integer.
                    # But the line might have other integer points if gcd(dx_l, dy_l) > 1.
                    # Let g = gcd(dx_l, dy_l).
                    # The primitive step is (dx_l/g, dy_l/g).
                    # So integer points are (x1 + m*(dx_l/g), y1 + m*(dy_l/g)) for integer m.
                    
                    g = math.gcd(abs(dx_l), abs(dy_l))
                    if g == 0:
                        continue
                    step_x = dx_l // g
                    step_y = dy_l // g
                    
                    # We want m such that (x1 + m*step_x, y1 + m*step_y) is close to (x, y)
                    # m_approx = (x - x1) / step_x
                    if step_x != 0:
                        m_approx = (x - x1) / step_x
                    elif step_y != 0:
                        m_approx = (y - y1) / step_y
                    else:
                        continue
                        
                    # Check m values around m_approx
                    m_int = int(round(m_approx))
                    for dm in range(-3, 4):
                        m = m_int + dm
                        nx = x1 + m * step_x
                        ny = y1 + m * step_y
                        if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                            candidates.add((nx, ny))
            
            # Evaluate candidates
            best_move = None
            best_move_count = current_count
            
            for nx, ny in candidates:
                if time.time() > deadline:
                    break
                res = _try_move(current_points, i, nx, ny)
                if res is None:
                    continue
                new_pts, cnt = res
                if cnt < best_move_count:
                    best_move_count = cnt
                    best_move = (new_pts, cnt)
            
            if best_move:
                current_points = best_move[0]
                current_count = best_move[1]
                improved = True
                stuck_count = 0
                if current_count <= 6:
                    return current_points
        
        if not improved:
            stuck_count += 1
            # If stuck for a while, try a random restart or larger perturbation
            if stuck_count > 5:
                # Random restart: perturb all points slightly
                for i in range(12):
                    x, y = current_points[i]
                    current_points[i] = [x + random.randint(-10, 10), y + random.randint(-10, 10)]
                if not _valid(current_points):
                    # Revert if invalid
                    current_points = best_points
                    current_count = best_count
                else:
                    current_count = _count_ordinary(current_points)
                stuck_count = 0
            
            # Also try a completely new random seed occasionally
            if stuck_count > 10:
                new_pts = [[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(12)]
                if _valid(new_pts):
                    cnt = _count_ordinary(new_pts)
                    if cnt < current_count:
                        current_points = new_pts
                        current_count = cnt
                stuck_count = 0

    return current_points

# EVOLVE-BLOCK-END
