"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    lines = {}
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
            else:
                continue
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            c = dy * x1 - dx * y1
            key = (dx, dy, c)
            if key not in lines:
                lines[key] = 2
            else:
                lines[key] += 1
    return sum(1 for v in lines.values() if v == 2)

def _local_search(points, deadline):
    """Local search to minimize ordinary lines."""
    points = [list(p) for p in points]
    best = [list(p) for p in points]
    best_count = _count_ordinary(best)
    
    while time.time() < deadline:
        idx = random.randint(0, 11)
        old_x, old_y = points[idx]
        
        # Try a random perturbation
        new_x = old_x + random.randint(-3, 3)
        new_y = old_y + random.randint(-3, 3)
        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
            continue
        dup = any(j != idx and points[j][0] == new_x and points[j][1] == new_y for j in range(12))
        if dup:
            continue
        
        points[idx] = [new_x, new_y]
        cnt = _count_ordinary(points)
        if cnt < best_count:
            best = [list(p) for p in points]
            best_count = cnt
            if best_count <= 6:
                break
        else:
            points[idx] = [old_x, old_y]
    
    return best, best_count

def construct():
    start = time.time()
    deadline = start + 135  # 15s safety margin
    
    # Multiple starting configurations
    configs = [
        # 3x4 grid
        [[x, y] for x in range(4) for y in range(3)],
        # 4x3 grid  
        [[x, y] for x in range(3) for y in range(4)],
        # 2x6 grid
        [[x, y] for x in range(6) for y in range(2)],
        # 6x2 grid
        [[x, y] for x in range(2) for y in range(6)],
        # 3-3-3-3 pattern
        [[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,2],[2,2],[0,3],[1,3],[2,3]],
        # 4+4+4 pattern
        [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,2]],
        # 11+1 baseline
        [[i, 0] for i in range(11)] + [[0, 1]],
    ]
    
    best_points = None
    best_count = float('inf')
    
    # Give each config roughly equal time
    time_per_config = (deadline - start) / len(configs)
    
    for cfg in configs:
        if time.time() >= deadline:
            break
        cfg_deadline = min(deadline, time.time() + time_per_config)
        pts, cnt = _local_search(cfg, cfg_deadline)
        if cnt < best_count:
            best_count = cnt
            best_points = [list(p) for p in pts]
        if best_count <= 6:
            break
    
    if best_points is None:
        best_points = [[i, 0] for i in range(11)] + [[0, 1]]
    
    return best_points
# EVOLVE-BLOCK-END
