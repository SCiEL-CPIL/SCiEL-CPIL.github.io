import time
import random
from collections import defaultdict

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    # Map each line to the set of point indices on it
    line_map = defaultdict(set)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            if dx == 0 and dy == 0:
                continue
            g = _gcd(abs(dx), abs(dy))
            dx_n = dx // g
            dy_n = dy // g
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            c = dy_n * xi - dx_n * yi
            line_key = (dx_n, dy_n, c)
            line_map[line_key].add(i)
            line_map[line_key].add(j)
    
    ordinary = 0
    for key, pts in line_map.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _local_search(points, deadline):
    """Local search: move one point at a time to reduce ordinary lines."""
    n = len(points)
    best = [p[:] for p in points]
    best_cnt = _count_ordinary_lines(best)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(n):
            if time.time() > deadline:
                break
            xi, yi = best[idx]
            # Try nearby moves
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    new_pt = [xi + dx, yi + dy]
                    if abs(new_pt[0]) > 10**9 or abs(new_pt[1]) > 10**9:
                        continue
                    # Check distinctness
                    valid = True
                    for j in range(n):
                        if j != idx and best[j] == new_pt:
                            valid = False
                            break
                    if not valid:
                        continue
                    # Check not all collinear
                    # (Optional quick check, usually not an issue in search)
                    
                    best[idx] = new_pt
                    cnt = _count_ordinary_lines(best)
                    if cnt < best_cnt:
                        best_cnt = cnt
                        improved = True
                    else:
                        best[idx] = [xi, yi]
            # Try random distant moves
            for _ in range(10):
                if time.time() > deadline:
                    break
                rx = random.randint(-100, 100)
                ry = random.randint(-100, 100)
                new_pt = [rx, ry]
                valid = True
                for j in range(n):
                    if j != idx and best[j] == new_pt:
                        valid = False
                        break
                if not valid:
                    continue
                best[idx] = new_pt
                cnt = _count_ordinary_lines(best)
                if cnt < best_cnt:
                    best_cnt = cnt
                    improved = True
                else:
                    best[idx] = [xi, yi]
    return best, best_cnt

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Generate diverse seeds
    seeds = []
    
    # 1. 3x3 Grid + 3 "Infinity" points (large coords)
    # Grid: 0..2, 0..2
    grid = [[x, y] for x in range(3) for y in range(3)]
    # Add points at (100, 0), (0, 100), (100, 100) ?
    # Better: align with grid directions
    seeds.append(grid + [[10, 0], [0, 10], [10, 10]])
    
    # 2. 4x3 Grid
    seeds.append([[x, y] for x in range(4) for y in range(3)])
    
    # 3. 3x4 Grid
    seeds.append([[x, y] for x in range(3) for y in range(4)])
    
    # 4. Pappus-like
    # Two triangles, vertices collinear with opposite vertices?
    # Just random for now
    
    # 5. 12 points on a circle? (Integer approx)
    
    best_points = None
    best_score = float('inf')
    
    for seed in seeds:
        # Ensure distinct and within bounds
        if len(set(tuple(p) for p in seed)) != 12:
            continue
        if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in seed):
            continue
        pts, score = _local_search(seed, deadline)
        if score < best_score:
            best_score = score
            best_points = pts
            
    # If no seeds worked, fallback to random
    if best_points is None:
        random.seed(42)
        pts = [[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(12)]
        # Ensure distinct
        while len(set(tuple(p) for p in pts)) < 12:
            i = random.randint(0, 11)
            pts[i] = [random.randint(-10, 10), random.randint(-10, 10)]
        best_points, best_score = _local_search(pts, deadline)
        
    return [list(p) for p in best_points]
