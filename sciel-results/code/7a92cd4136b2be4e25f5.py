"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
from itertools import combinations

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            g = _gcd3(A, B, C)
            if g > 1:
                A //= g
                B //= g
                C //= g
            key = (A, B, C)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    
    ordinary = 0
    for key, pts in line_points.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _gcd3(a, b, c):
    from math import gcd
    g = gcd(abs(a), abs(b))
    g = gcd(g, abs(c))
    return g

def _all_collinear(points):
    if len(points) < 3:
        return False
    (x0, y0), (x1, y1), (x2, y2) = points[0], points[1], points[2]
    cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    if cross != 0:
        return False
    for i in range(3, len(points)):
        xi, yi = points[i]
        if (x1 - x0) * (yi - y0) - (y1 - y0) * (xi - x0) != 0:
            return False
    return True

def _generate_grid(rows, cols):
    pts = []
    for i in range(rows):
        for j in range(cols):
            pts.append([j, i])
    return pts

def _generate_11_on_line():
    pts = []
    for i in range(11):
        pts.append([i, 0])
    pts.append([0, 100])
    return pts

def _generate_cross():
    # 6 points on x-axis, 6 on y-axis, shifted to avoid origin
    pts = []
    for i in range(1, 7):
        pts.append([i, 0])
    for i in range(1, 7):
        pts.append([0, i])
    return pts

def _simulated_annealing(points, time_limit):
    current = [p[:] for p in points]
    current_score = _count_ordinary_lines(current)
    best = [p[:] for p in current]
    best_score = current_score
    
    n = len(current)
    T = 10.0
    T_end = 0.01
    start_time = time.time()
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit:
        if T < T_end:
            break
            
        # Move a random point
        i = random.randint(0, n - 1)
        orig = current[i][:]
        
        # Random move in a large box
        # Allow large jumps to escape local minima
        new_x = orig[0] + random.randint(-100, 100)
        new_y = orig[1] + random.randint(-100, 100)
        
        # Check bounds
        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
            continue
            
        # Check distinctness
        pos = (new_x, new_y)
        if pos == (orig[0], orig[1]):
            continue
        dup = False
        for j in range(n):
            if j != i and current[j][0] == new_x and current[j][1] == new_y:
                dup = True
                break
        if dup:
            continue
            
        current[i] = [new_x, new_y]
        new_score = _count_ordinary_lines(current)
        
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best = [p[:] for p in current]
        else:
            current[i] = orig
        
        T *= cooling_rate
        
    return best, best_score

def construct():
    deadline = time.time() + 145  # Leave 5s margin for 150s limit
    best_points = None
    best_score = float('inf')
    
    # Initial configurations
    configs = []
    configs.append(_generate_grid(3, 4))
    configs.append(_generate_grid(4, 3))
    configs.append(_generate_11_on_line())
    configs.append(_generate_cross())
    
    # Add some random perturbations of the grid
    random.seed(42)
    for _ in range(5):
        base = _generate_grid(3, 4)
        for p in base:
            p[0] += random.randint(-10, 10)
            p[1] += random.randint(-10, 10)
        configs.append(base)
        
    # Shuffle configs to diversify starting points
    random.shuffle(configs)

    for config in configs:
        if time.time() > deadline:
            break
        
        # Validate
        seen = set()
        valid = True
        for p in config:
            tp = tuple(p)
            if tp in seen or abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
                valid = False
                break
            seen.add(tp)
        if not valid or len(seen) != 12:
            continue
        if _all_collinear(config):
            continue
            
        # Run SA
        time_left = deadline - time.time()
        if time_left < 5:
            continue
            
        pts, score = _simulated_annealing(config, time_limit=time_left * 0.9)
        if score < best_score:
            best_score = score
            best_points = pts
            
        if best_score <= 6:
            break
            
    if best_points is None:
        # Fallback: return the 3x4 grid
        return _generate_grid(3, 4)
        
    return best_points
# EVOLVE-BLOCK-END
