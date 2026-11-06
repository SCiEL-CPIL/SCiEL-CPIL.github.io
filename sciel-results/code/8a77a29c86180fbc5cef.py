"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

DEADLINE = 140.0  # Safety margin before 150s limit

def _count_ordinary(points):
    """Count the number of lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    # For each pair of points, count how many other points lie on the line through them
    ordinary = 0
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Count points on the line through (x1,y1) and (x2,y2)
            count = 2  # Start with the two points themselves
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                # Check collinearity using cross product
                cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                if cross == 0:
                    count += 1
            if count == 2:
                ordinary += 1
    return ordinary

def _is_valid(points):
    """Check if the configuration is valid: 12 distinct points, not all collinear, |x|,|y| <= 10^9."""
    if len(points) != 12:
        return False
    seen = set()
    for x, y in points:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    # Check not all collinear
    if len(points) >= 3:
        x1, y1 = points[0]
        x2, y2 = points[1]
        x3, y3 = points[2]
        cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        if cross == 0:
            # Check if all points are collinear
            all_collinear = True
            for k in range(3, 12):
                xk, yk = points[k]
                cross_k = (x2 - x1) * (yk - y1) - (y2 - y1) * (xk - x1)
                if cross_k != 0:
                    all_collinear = False
                    break
            if all_collinear:
                return False
    return True

def _random_point():
    return [random.randint(-10**9, 10**9), random.randint(-10**9, 10**9)]

def construct():
    start_time = time.time()
    
    # Start with a 3x4 grid as initial configuration
    best_points = [[x, y] for x in range(4) for y in range(3)]
    best_ordinary = _count_ordinary(best_points)
    
    # Local search with random restarts
    max_time_per_phase = 20  # seconds per phase
    
    # Phase 1: Local search from the initial grid
    points = [p[:] for p in best_points]
    phase_start = time.time()
    while time.time() - phase_start < max_time_per_phase:
        if time.time() > DEADLINE:
            break
        # Pick a random point and try to move it
        idx = random.randint(0, 11)
        old_point = points[idx][:]
        
        # Try several random moves
        improved = False
        for _ in range(50):
            if time.time() > DEADLINE:
                break
            new_x = random.randint(-10**9, 10**9)
            new_y = random.randint(-10**9, 10**9)
            points[idx] = [new_x, new_y]
            
            if not _is_valid(points):
                points[idx] = old_point
                continue
            
            ord_count = _count_ordinary(points)
            if ord_count < best_ordinary:
                best_ordinary = ord_count
                best_points = [p[:] for p in points]
                improved = True
                break
            else:
                points[idx] = old_point
        
        if not improved and random.random() < 0.1:
            # Occasional random jump
            for i in range(12):
                points[i] = _random_point()
    
    # Phase 2: More aggressive search with larger moves
    phase_start = time.time()
    points = [p[:] for p in best_points]
    while time.time() - phase_start < max_time_per_phase:
        if time.time() > DEADLINE:
            break
        idx = random.randint(0, 11)
        old_point = points[idx][:]
        
        # Try moves with smaller perturbations first, then larger
        for scale in [1, 10, 100, 1000, 1000000]:
            if time.time() > DEADLINE:
                break
            for _ in range(10):
                if time.time() > DEADLINE:
                    break
                new_x = old_point[0] + random.randint(-scale, scale)
                new_y = old_point[1] + random.randint(-scale, scale)
                new_x = max(-10**9, min(10**9, new_x))
                new_y = max(-10**9, min(10**9, new_y))
                points[idx] = [new_x, new_y]
                
                if not _is_valid(points):
                    points[idx] = old_point
                    continue
                
                ord_count = _count_ordinary(points)
                if ord_count < best_ordinary:
                    best_ordinary = ord_count
                    best_points = [p[:] for p in points]
                    break
                else:
                    points[idx] = old_point
            if ord_count < best_ordinary:
                break
    
    # Phase 3: Try known good configurations
    # 3x4 grid
    grid_3x4 = [[x, y] for x in range(4) for y in range(3)]
    ord_3x4 = _count_ordinary(grid_3x4)
    if ord_3x4 < best_ordinary:
        best_ordinary = ord_3x4
        best_points = grid_3x4
    
    # 2x6 grid
    grid_2x6 = [[x, y] for x in range(6) for y in range(2)]
    ord_2x6 = _count_ordinary(grid_2x6)
    if ord_2x6 < best_ordinary:
        best_ordinary = ord_2x6
        best_points = grid_2x6
    
    # Hexagonal-like: 2 rows of 3 with offset
    hex_config = [[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,2],[2,2],[3,0],[3,1],[3,2]]
    ord_hex = _count_ordinary(hex_config)
    if ord_hex < best_ordinary:
        best_ordinary = ord_hex
        best_points = hex_config
    
    # Return the best found
    return [list(p) for p in best_points]
# EVOLVE-BLOCK-END
