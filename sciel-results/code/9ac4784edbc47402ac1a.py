"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations
from math import gcd

# EVOLVE-BLOCK-START

def _line_key(p1, p2):
    """Get normalized line key for two points."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    g = gcd(abs(dx), abs(dy))
    if g > 0:
        dx //= g
        dy //= g
    # Normalize direction
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    return (dx, dy)

def _count_4point_lines(points):
    """Count lines containing exactly 4 of the given points."""
    n = len(points)
    count = 0
    # Use a dict to track lines by their normalized direction and intercept
    # Key: (dx, dy, x_intercept, y_intercept) where line is y = (dy/dx)x + c
    # Better: use the form ax + by = c with gcd(a,b)=1
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            # Line: dy*x - dx*y = c
            c = dy * x1 - dx * y1
            # Normalize sign
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy, c = -dx, -dy, -c
            key = (dx, dy, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    
    for key, pts in line_map.items():
        if len(pts) == 4:
            count += 1
    return count

def _validate(points):
    """Check if points are valid: 16 distinct, |x|,|y|<=1000, no 5 collinear."""
    if len(points) != 16:
        return False
    seen = set()
    for p in points:
        x, y = p
        if abs(x) > 1000 or abs(y) > 1000:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    
    # Check no 5 collinear
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            c = dy * x1 - dx * y1
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy, c = -dx, -dy, -c
            key = (dx, dy, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    
    for key, pts in line_map.items():
        if len(pts) >= 5:
            return False
    return True

def _initial_grid():
    """Return the 4x4 grid."""
    return [[x, y] for x in range(4) for y in range(4)]

def _local_search(points, deadline, rng):
    """Perform local search to improve the 4-point line count."""
    current = [list(p) for p in points]
    current_score = _count_4point_lines(current)
    
    # Precompute all pair indices for speed
    n = len(current)
    
    while time.time() < deadline:
        improved = False
        # Try moving each point to a random nearby position
        for attempt in range(50):
            if time.time() >= deadline:
                break
            idx = rng.randint(0, n - 1)
            old_x, old_y = current[idx]
            # Try a random displacement
            dx = rng.randint(-5, 5)
            dy = rng.randint(-5, 5)
            if dx == 0 and dy == 0:
                continue
            new_x = old_x + dx
            new_y = old_y + dy
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
            # Check if new position is already occupied
            if (new_x, new_y) in [(p[0], p[1]) for k, p in enumerate(current) if k != idx]:
                continue
            
            current[idx] = [new_x, new_y]
            new_score = _count_4point_lines(current)
            if new_score > current_score:
                current_score = new_score
                improved = True
            else:
                current[idx] = [old_x, old_y]
        
        if not improved:
            # Try larger moves
            for attempt in range(20):
                if time.time() >= deadline:
                    break
                idx = rng.randint(0, n - 1)
                old_x, old_y = current[idx]
                dx = rng.randint(-15, 15)
                dy = rng.randint(-15, 15)
                if dx == 0 and dy == 0:
                    continue
                new_x = old_x + dx
                new_y = old_y + dy
                if abs(new_x) > 1000 or abs(new_y) > 1000:
                    continue
                if (new_x, new_y) in [(p[0], p[1]) for k, p in enumerate(current) if k != idx]:
                    continue
                
                current[idx] = [new_x, new_y]
                new_score = _count_4point_lines(current)
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                else:
                    current[idx] = [old_x, old_y]
            
            if not improved:
                break
    
    return current, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_points = None
    best_score = -1
    rng = random.Random(42)
    
    # Start from the 4x4 grid
    initial = _initial_grid()
    if _validate(initial):
        best_points = [list(p) for p in initial]
        best_score = _count_4point_lines(best_points)
    
    # Local search from initial grid
    if time.time() < deadline:
        points, score = _local_search(initial, deadline, rng)
        if _validate(points) and score > best_score:
            best_points = points
            best_score = score
    
    # Try some random restarts
    while time.time() < deadline - 5:
        # Generate a random starting configuration near the grid
        start_pts = []
        for i in range(16):
            x = rng.randint(-3, 6)
            y = rng.randint(-3, 6)
            start_pts.append([x, y])
        
        if len(set((p[0], p[1]) for p in start_pts)) != 16:
            continue
        if not _validate(start_pts):
            continue
        
        points, score = _local_search(start_pts, deadline, rng)
        if _validate(points) and score > best_score:
            best_points = points
            best_score = score
    
    if best_points is None:
        best_points = _initial_grid()
    
    return best_points

# EVOLVE-BLOCK-END

