"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    # Use a set of normalized line representations
    line_points = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Normalize the direction
            if dx == 0 and dy == 0:
                continue
            # Normalize: make gcd positive, and handle signs
            g = _gcd(abs(dx), abs(dy))
            dx_n = dx // g
            dy_n = dy // g
            # Canonical sign: first non-zero component positive
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            # The line is determined by direction (dx_n, dy_n) and a point on it
            # Use the "cross product" invariant: for line with direction (a,b) through (x0,y0):
            # b*x - a*y = b*x0 - a*y0
            c = dy_n * x1 - dx_n * y1
            line_key = (dx_n, dy_n, c)
            if line_key not in line_points:
                line_points[line_key] = set()
            line_points[line_key].add(i)
            line_points[line_key].add(j)
    
    ordinary = 0
    for key, pts in line_points.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # Start with a 3x4 grid as baseline
    best_points = [[x, y] for x in range(4) for y in range(3)]
    best_score = count_ordinary_lines(best_points)
    
    # Try several initial configurations
    initial_configs = [
        # 3x4 grid
        [[x, y] for x in range(4) for y in range(3)],
        # 4x3 grid
        [[x, y] for x in range(3) for y in range(4)],
        # Try some structured configs with more collinearities
        # Two rows of 6 with shared points
        [[i, 0] for i in range(6)] + [[i, 1] for i in range(3, 9)],
        # 3 rows of 4
        [[i, j] for i in range(4) for j in range(3)],
        # A configuration based on the Hesse curve idea (simplified)
        # Points on a cubic-like arrangement
        [[i, i*i % 13] for i in range(12)],
    ]
    
    for config in initial_configs:
        if time.time() > deadline:
            break
        score = count_ordinary_lines(config)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in config]
    
    # Local search: try to improve by moving individual points
    random.seed(42)
    points = [p[:] for p in best_points]
    
    max_iterations = 50000
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Try moving a random point to a random nearby location
        idx = random.randint(0, 11)
        old_x, old_y = points[idx]
        
        # Try several candidate moves
        candidates = []
        # Small perturbations
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                candidates.append([old_x + dx, old_y + dy])
        # Larger jumps
        for _ in range(5):
            candidates.append([old_x + random.randint(-10, 10), old_y + random.randint(-10, 10)])
        
        for new_pt in candidates:
            if abs(new_pt[0]) > 10**9 or abs(new_pt[1]) > 10**9:
                continue
            # Check distinctness
            if any(new_pt[0] == points[j][0] and new_pt[1] == points[j][1] for j in range(12) if j != idx):
                continue
            points[idx] = new_pt
            score = count_ordinary_lines(points)
            if score < best_score:
                best_score = score
                best_points = [p[:] for p in points]
            else:
                points[idx] = [old_x, old_y]
                # If no improvement, keep original
    
    # Try a few more structured approaches
    # Configuration with points on a circle-like pattern (integer approximations)
    circle_configs = [
        # 12 points: try to get many 3-point lines
        # Use a 2x6 grid with some modifications
        [[i, 0] for i in range(6)] + [[i+1, 1] for i in range(6)],
        # 3x4 but with one point moved to create extra collinearities
        [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,2]],
        # Try a configuration with 4 points on a line, 4 on another, etc.
        [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,3],[3,3]],
    ]
    
    for config in circle_configs:
        if time.time() > deadline:
            break
        score = count_ordinary_lines(config)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in config]
    
    # Final local search from best
    points = [p[:] for p in best_points]
    for iteration in range(30000):
        if time.time() > deadline:
            break
        idx = random.randint(0, 11)
        old_x, old_y = points[idx]
        
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                if dx == 0 and dy == 0:
                    continue
                new_pt = [old_x + dx, old_y + dy]
                if abs(new_pt[0]) > 10**9 or abs(new_pt[1]) > 10**9:
                    continue
                if any(new_pt[0] == points[j][0] and new_pt[1] == points[j][1] for j in range(12) if j != idx):
                    continue
                points[idx] = new_pt
                score = count_ordinary_lines(points)
                if score < best_score:
                    best_score = score
                    best_points = [p[:] for p in points]
                else:
                    points[idx] = [old_x, old_y]
    
    return best_points
# EVOLVE-BLOCK-END
