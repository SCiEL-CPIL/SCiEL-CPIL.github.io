"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

DEADLINE = 140.0  # Safety margin before 150s limit

def _count_ordinary_fast(points):
    """Count ordinary lines using normalized line representation (faster than O(n^3))."""
    n = len(points)
    if n < 2:
        return 0
    
    line_counts = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Line through two points: a*x + b*y + c = 0
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            
            # Normalize the line representation
            g = gcd(gcd(abs(a), abs(b)), abs(c)) if (a != 0 or b != 0 or c != 0) else 1
            if g == 0:
                g = 1
            a, b, c = a // g, b // g, c // g
            
            # Ensure consistent sign: make the first non-zero of (a,b,c) positive
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a, b, c = -a, -b, -c
            
            key = (a, b, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    # Count lines with exactly 2 points (count == 1 pair on that line means 2 points)
    ordinary = sum(1 for count in line_counts.values() if count == 1)
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

def construct():
    start_time = time.time()
    
    # Start with a 3x4 grid as initial configuration
    best_points = [[x, y] for x in range(4) for y in range(3)]
    best_ordinary = _count_ordinary_fast(best_points)
    
    # Phase 1: Local search with fast counting
    max_time_per_phase = 30  # seconds per phase
    
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
        for _ in range(30):
            if time.time() > DEADLINE:
                break
            new_x = random.randint(-10**9, 10**9)
            new_y = random.randint(-10**9, 10**9)
            points[idx] = [new_x, new_y]
            
            if not _is_valid(points):
                points[idx] = old_point
                continue
            
            ord_count = _count_ordinary_fast(points)
            if ord_count < best_ordinary:
                best_ordinary = ord_count
                best_points = [p[:] for p in points]
                improved = True
                break
            else:
                points[idx] = old_point
        
        if not improved and random.random() < 0.05:
            # Occasional random jump
            for i in range(12):
                points[i] = [random.randint(-10**9, 10**9), random.randint(-10**9, 10**9)]
    
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
                
                ord_count = _count_ordinary_fast(points)
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
    ord_3x4 = _count_ordinary_fast(grid_3x4)
    if ord_3x4 < best_ordinary:
        best_ordinary = ord_3x4
        best_points = grid_3x4
    
    # 2x6 grid
    grid_2x6 = [[x, y] for x in range(6) for y in range(2)]
    ord_2x6 = _count_ordinary_fast(grid_2x6)
    if ord_2x6 < best_ordinary:
        best_ordinary = ord_2x6
        best_points = grid_2x6
    
    # Hexagonal-like: 2 rows of 3 with offset
    hex_config = [[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,2],[2,2],[3,0],[3,1],[3,2]]
    ord_hex = _count_ordinary_fast(hex_config)
    if ord_hex < best_ordinary:
        best_ordinary = ord_hex
        best_points = hex_config
    
    # Two lines configuration: 6 points on one line, 6 on another, sharing an intersection
    two_lines = [[i, 0] for i in range(6)] + [[0, i] for i in range(1, 6)]
    two_lines.append([5, 5])
    if len(two_lines) == 12 and len(set((p[0], p[1]) for p in two_lines)) == 12:
        ord_two = _count_ordinary_fast(two_lines)
        if ord_two < best_ordinary:
            best_ordinary = ord_two
            best_points = two_lines
    
    # Return the best found
    return [list(p) for p in best_points]
# EVOLVE-BLOCK-END
