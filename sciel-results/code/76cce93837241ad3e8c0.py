"""Rewritten program targeting fewer ordinary lines for 12 points."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations, permutations

# Deadline guard: evaluator kills after 150s wall-clock. Use 130s as safety margin.
_DEADLINE = 130.0
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    # Use a dict to map line (normalized) -> count of points on it
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = _gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            c = dy * x1 - dx * y1
            key = (dx, dy, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    ordinary = 0
    for count in line_counts.values():
        if count == 2:
            ordinary += 1
    return ordinary

def _score(points):
    """Lower is better: number of ordinary lines."""
    return _count_ordinary_lines(points)

def _is_valid(points):
    """Check that points are 12 distinct integer points with |x|,|y| <= 10^9, not all collinear."""
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
    if len(points) >= 3:
        x1, y1 = points[0]
        x2, y2 = points[1]
        for i in range(2, 12):
            x3, y3 = points[i]
            cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
            if cross != 0:
                return True
        return False
    return True

def _line_through_two(p1, p2):
    """Return normalized line representation (dx, dy, c) for the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = _gcd(abs(dx), abs(dy))
    if g > 0:
        dx //= g
        dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _points_on_line(line_key, points):
    """Return indices of points that lie on the given line."""
    dx, dy, c = line_key
    result = []
    for i, (x, y) in enumerate(points):
        if dy * x - dx * y == c:
            result.append(i)
    return result

def _try_complete_line(points, target_size=3):
    """
    Try to find a pair of points and a third point such that moving one point
    to a new position creates a 3-point line.
    Returns (new_points, new_score) if improvement found, else (None, None).
    """
    n = len(points)
    best_new_points = None
    best_new_score = _score(points)
    
    # For each pair, compute the line
    for i in range(n):
        for j in range(i + 1, n):
            line_key = _line_through_two(points[i], points[j])
            pts_on_line = _points_on_line(line_key, points)
            
            if len(pts_on_line) == 2:
                # This is an ordinary line. Try to extend it to 3 points.
                # The line is determined by points[i] and points[j].
                # We can pick a third point k and move it to a position on this line.
                # The new position should be an integer point on the line.
                # Line: dy*x - dx*y = c
                # We can parametrize: if gcd(dx, dy) == 1, then (x, y) = (x0 + dx*t, y0 + dy*t)
                # But we need integer coordinates.
                
                dx, dy, c = line_key
                x0, y0 = points[i]
                
                # Find a nearby integer point on this line that is not already occupied
                # and is distinct from all other points.
                occupied = set((p[0], p[1]) for p in points)
                
                # Try t = -5 to 5 (excluding t=0 and t=1 which give points i and j)
                for t in range(-10, 11):
                    if t == 0:
                        continue
                    new_x = x0 + dx * t
                    new_y = y0 + dy * t
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    if (new_x, new_y) in occupied:
                        continue
                    
                    # Try moving each point k (not i or j) to (new_x, new_y)
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        old_x, old_y = points[k]
                        if (old_x, old_y) == (new_x, new_y):
                            continue
                        
                        new_points = [p[:] for p in points]
                        new_points[k] = [new_x, new_y]
                        
                        # Check validity
                        seen = set()
                        valid = True
                        for p in new_points:
                            if (p[0], p[1]) in seen:
                                valid = False
                                break
                            seen.add((p[0], p[1]))
                        if not valid:
                            continue
                        
                        # Check not all collinear
                        x1, y1 = new_points[0]
                        x2, y2 = new_points[1]
                        all_collinear = True
                        for idx in range(2, 12):
                            x3, y3 = new_points[idx]
                            cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                            if cross != 0:
                                all_collinear = False
                                break
                        if all_collinear:
                            continue
                        
                        new_score = _score(new_points)
                        if new_score < best_new_score:
                            best_new_score = new_score
                            best_new_points = new_points
    
    return best_new_points, best_new_score

def _local_search_line_completion(points, max_iters=5000):
    """Local search that focuses on completing ordinary lines to 3-point lines."""
    points = [p[:] for p in points]
    current_score = _score(points)
    best_points = [p[:] for p in points]
    best_score = current_score
    temp = 5.0
    cooling = 0.9995
    
    for iteration in range(max_iters):
        if _time_left() < 5.0:
            break
        if iteration % 200 == 0:
            temp *= cooling
        
        # Try line completion
        new_points, new_score = _try_complete_line(points)
        if new_points is not None and new_score < current_score:
            points = new_points
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
            continue
        
        # If no improvement from line completion, do a random move
        idx = random.randint(0, 11)
        old_x, old_y = points[idx]
        
        # Try a random new position
        new_x = random.randint(-30, 30)
        new_y = random.randint(-30, 30)
        
        new_point = (new_x, new_y)
        if any(new_point == (points[i][0], points[i][1]) for i in range(12) if i != idx):
            continue
        
        points[idx] = [new_x, new_y]
        new_score = _score(points)
        
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / max(temp, 0.1)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
        else:
            points[idx] = [old_x, old_y]
    
    return best_points, best_score

def _construct():
    """Main construction function with time management."""
    random.seed(42)
    
    # Initial configurations
    initial_configs = []
    
    # 3x4 grid
    initial_configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # 4x3 grid
    initial_configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # 2x6 grid
    initial_configs.append([[x, y] for x in range(6) for y in range(2)])
    
    # 6x2 grid
    initial_configs.append([[x, y] for x in range(2) for y in range(6)])
    
    # Try a configuration based on 4 lines of 3 points
    # Line 1: (0,0), (1,0), (2,0)
    # Line 2: (0,1), (1,1), (2,1)
    # Line 3: (0,2), (1,2), (2,2)
    # Line 4: (0,0), (0,1), (0,2)
    # But we need 12 distinct points, so this gives only 6.
    # Let's try a different arrangement.
    
    # Configuration: 3 points on each of 4 lines, with overlaps
    # This is hard to construct by hand. Let's rely on search.
    
    best_points = None
    best_score = float('inf')
    
    # Evaluate initial configurations
    for config in initial_configs:
        if not _is_valid(config):
            continue
        s = _score(config)
        if s < best_score:
            best_score = s
            best_points = [p[:] for p in config]
    
    # Local search from each initial configuration
    for config in initial_configs:
        if _time_left() < 10.0:
            break
        if not _is_valid(config):
            continue
        improved_points, improved_score = _local_search_line_completion(config, max_iters=8000)
        if improved_score < best_score:
            best_score = improved_score
            best_points = improved_points
    
    # Random restarts
    while _time_left() > 20.0:
        random_points = []
        for _ in range(12):
            x = random.randint(-20, 20)
            y = random.randint(-20, 20)
            if (x, y) not in [(p[0], p[1]) for p in random_points]:
                random_points.append([x, y])
        
        if len(random_points) == 12 and _is_valid(random_points):
            improved_points, improved_score = _local_search_line_completion(random_points, max_iters=5000)
            if improved_score < best_score:
                best_score = improved_score
                best_points = improved_points
    
    # Final validation
    if not _is_valid(best_points):
        best_points = [[x, y] for x in range(4) for y in range(3)]
    
    return best_points

def construct():
    """Entry point: returns exactly 12 distinct integer points."""
    global _START_TIME
    _START_TIME = time.time()
    return _construct()
# EVOLVE-BLOCK-END
