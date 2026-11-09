"""
Optimized 12-point configuration with few ordinary lines.
Uses a 3x4 grid base with targeted local search.
"""
import time
import random
from math import gcd

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    lines = {}
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
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            c = y1 * dx - x1 * dy
            key = (dx, dy, c)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    ordinary = sum(1 for pts in lines.values() if len(pts) == 2)
    return ordinary

def _init_3x4_grid():
    """Return a 3x4 grid of points."""
    points = []
    for j in range(3):
        for i in range(4):
            points.append([i, j])
    return points

def _init_alternative():
    """Return an alternative configuration: 4x3 grid with some modifications."""
    points = []
    # 4x3 grid
    for j in range(3):
        for i in range(4):
            points.append([i, j])
    return points

def _init_two_lines():
    """Return points on two parallel lines."""
    points = []
    # 6 points on y=0
    for i in range(6):
        points.append([i, 0])
    # 6 points on y=1
    for i in range(6):
        points.append([i, 1])
    return points

def _init_star():
    """Return a star-like configuration."""
    points = [
        [0, 0],
        [1, 0], [-1, 0],
        [0, 1], [0, -1],
        [1, 1], [-1, -1],
        [1, -1], [-1, 1],
        [2, 0], [0, 2]
    ]
    return points

def _valid(points):
    """Check if points are valid: 12 distinct integer points, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for x, y in points:
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    # Check not all collinear
    if len(points) >= 3:
        x0, y0 = points[0]
        x1, y1 = points[1]
        x2, y2 = points[2]
        cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        if cross == 0:
            # Check if all collinear
            all_collinear = True
            for i in range(3, 12):
                xi, yi = points[i]
                cross_i = (x1 - x0) * (yi - y0) - (y1 - y0) * (xi - x0)
                if cross_i != 0:
                    all_collinear = False
                    break
            if all_collinear:
                return False
    return True

def _local_search(points, time_limit=5.0):
    """Local search: move points to reduce ordinary lines."""
    start_time = time.time()
    current = [p[:] for p in points]
    current_count = _count_ordinary_lines(current)
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(len(current)):
            x, y = current[i]
            # Try small moves
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = x + dx
                    new_y = y + dy
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    # Check distinctness
                    valid = True
                    for j in range(len(current)):
                        if j != i and current[j][0] == new_x and current[j][1] == new_y:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    old = current[i][:]
                    current[i] = [new_x, new_y]
                    new_count = _count_ordinary_lines(current)
                    if new_count < current_count:
                        current_count = new_count
                        improved = True
                    else:
                        current[i] = old
        if not improved:
            break
    
    return current, current_count

def _local_search_wider(points, time_limit=5.0):
    """Local search with wider moves."""
    start_time = time.time()
    current = [p[:] for p in points]
    current_count = _count_ordinary_lines(current)
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(len(current)):
            x, y = current[i]
            # Try wider moves
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = x + dx
                    new_y = y + dy
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    # Check distinctness
                    valid = True
                    for j in range(len(current)):
                        if j != i and current[j][0] == new_x and current[j][1] == new_y:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    old = current[i][:]
                    current[i] = [new_x, new_y]
                    new_count = _count_ordinary_lines(current)
                    if new_count < current_count:
                        current_count = new_count
                        improved = True
                    else:
                        current[i] = old
        if not improved:
            break
    
    return current, current_count

def _try_configurations(deadline):
    """Try multiple configurations and local search."""
    best_points = None
    best_count = float('inf')
    
    configs = [
        _init_3x4_grid,
        _init_alternative,
        _init_two_lines,
        _init_star
    ]
    
    config_results = []
    for config_func in configs:
        try:
            points = config_func()
            if not _valid(points):
                continue
            count = _count_ordinary_lines(points)
            config_results.append((count, points, config_func))
        except Exception:
            continue
    
    config_results.sort(key=lambda x: x[0])
    
    # Try local search on the best configurations
    for count, points, config_func in config_results[:10]:
        if time.time() > deadline - 10:
            break
        improved_points, improved_count = _local_search(points, time_limit=3.0)
        if improved_count < best_count:
            best_count = improved_count
            best_points = improved_points
        
        # Wider local search on the best result so far
        if time.time() > deadline - 5:
            break
        improved_points2, improved_count2 = _local_search_wider(best_points, time_limit=3.0)
        if improved_count2 < best_count:
            best_count = improved_count2
            best_points = improved_points2
    
    # Final: if we have time, do one more round of local search
    if time.time() < deadline - 2 and best_points is not None:
        improved_points, improved_count = _local_search(best_points, time_limit=5.0)
        if improved_count < best_count:
            best_count = improved_count
            best_points = improved_points
    
    # Ensure we have a valid result
    if best_points is None or not _valid(best_points):
        # Fallback: 3x4 grid
        best_points = _init_3x4_grid()
        best_count = _count_ordinary_lines(best_points)
    
    # Final validation
    if not _valid(best_points):
        # Last resort
        best_points = [[i, 0] for i in range(11)] + [[0, 1]]
    
    return best_points[:12]

def construct():
    """
    Construct 12 integer points with a minimized number of ordinary lines.
    Uses combinatorial design with 3x4 grid and local search.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_points = _try_configurations(deadline)
    
    # Ensure exactly 12 distinct points
    seen = set()
    unique_points = []
    for p in best_points:
        key = (p[0], p[1])
        if key not in seen:
            seen.add(key)
            unique_points.append(p)
    
    # If we don't have 12, add more
    while len(unique_points) < 12:
        for i in range(1, 1000):
            for j in range(1, 1000):
                p = [i, j]
                if (p[0], p[1]) not in seen:
                    seen.add((p[0], p[1]))
                    unique_points.append(p)
                    break
            if len(unique_points) >= 12:
                break
    
    return unique_points[:12]
