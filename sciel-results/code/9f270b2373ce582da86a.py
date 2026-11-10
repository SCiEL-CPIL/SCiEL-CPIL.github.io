import random
import time
import math
from collections import defaultdict

def _line_key(pts, i, j):
    """Compute canonical line key for two points."""
    x1, y1 = pts[i]
    x2, y2 = pts[j]
    dx = x2 - x1
    dy = y2 - y1
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        return None
    dx //= g
    dy //= g
    # Normal vector
    A = dy
    B = -dx
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
    k = A * x1 + B * y1
    return (A, B, k)

def _evaluate(pts):
    """Fast evaluation: count lines with exactly 4 points, return -1 if any line has >=5."""
    line_map = defaultdict(list)
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(pts, i, j)
            if key is not None:
                line_map[key].append((i, j))
    
    # Build point-to-lines mapping
    point_lines = defaultdict(set)
    for key, pairs in line_map.items():
        for i, j in pairs:
            point_lines[i].add(key)
            point_lines[j].add(key)
    
    # Count points per line
    line_count = defaultdict(int)
    for key in line_map:
        pts_on_line = set()
        for i, j in line_map[key]:
            pts_on_line.add(i)
            pts_on_line.add(j)
        line_count[key] = len(pts_on_line)
    
    count_4 = 0
    for key, cnt in line_count.items():
        if cnt >= 5:
            return -1, 0
        if cnt == 4:
            count_4 += 1
    
    return 1, count_4

def _generate_initial_config(seed):
    """Generate diverse initial configurations."""
    rng = random.Random(seed)
    configs = []
    
    # 4x4 grid
    configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # Perturbed 4x4 grid
    base = [[x, y] for x in range(4) for y in range(4)]
    perturbed = [p[:] for p in base]
    for _ in range(5):
        idx = rng.randint(0, 15)
        dx = rng.randint(-2, 2)
        dy = rng.randint(-2, 2)
        perturbed[idx] = [perturbed[idx][0] + dx, perturbed[idx][1] + dy]
    configs.append(perturbed)
    
    # Lattice structure: points on a 4x4 grid with one row/col shifted
    lattice = [[x, y] for x in range(4) for y in range(4)]
    lattice[0] = [lattice[0][0], lattice[0][1] + 1]
    configs.append(lattice)
    
    # Random points in a bounded region
    random_pts = []
    for _ in range(16):
        x = rng.randint(-10, 10)
        y = rng.randint(-10, 10)
        if [x, y] not in random_pts:
            random_pts.append([x, y])
    while len(random_pts) < 16:
        x = rng.randint(-10, 10)
        y = rng.randint(-10, 10)
        if [x, y] not in random_pts:
            random_pts.append([x, y])
    configs.append(random_pts)
    
    return configs

def _local_polish(points, time_limit=5.0):
    """Final exhaustive local polish: try all single-point moves and accept improvements."""
    start = time.time()
    current = [p[:] for p in points]
    _, current_score = _evaluate(current)
    best = [p[:] for p in current]
    best_score = current_score
    
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        for idx in range(16):
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    new_pts = [p[:] for p in current]
                    nx = new_pts[idx][0] + dx
                    ny = new_pts[idx][1] + dy
                    if not (-1000 <= nx <= 1000 and -1000 <= ny <= 1000):
                        continue
                    new_pts[idx] = [nx, ny]
                    # Check distinctness
                    distinct = True
                    for i, p in enumerate(new_pts):
                        if i != idx and p == new_pts[idx]:
                            distinct = False
                            break
                    if not distinct:
                        continue
                    valid, score = _evaluate(new_pts)
                    if valid and score > best_score:
                        best_score = score
                        best = [p[:] for p in new_pts]
                        current = new_pts
                        improved = True
    return best

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    # Generate diverse initial configurations
    initial_configs = _generate_initial_config(42)
    
    best_points = None
    best_score = -1
    
    for config_idx, initial in enumerate(initial_configs):
        # Check if we have time
        if time.time() - start_time > time_limit - 20:
            break
        
        # Validate initial config
        valid, score = _evaluate(initial)
        if not valid:
            continue
        
        current_points = [p[:] for p in initial]
        current_score = score
        
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in current_points]
        
        # Simulated annealing
        T = 15.0
        T_min = 0.1
        cooling_rate = 0.9998
        
        step = 0
        sa_start = time.time()
        sa_time_limit = min(25.0, time_limit - (time.time() - start_time) - 10)
        
        while time.time() - sa_start < sa_time_limit:
            new_points = [p[:] for p in current_points]
            idx = random.randint(0, 15)
            
            # Choose move size
            r = random.random()
            if r < 0.7:
                max_delta = 3
            elif r < 0.9:
                max_delta = 8
            else:
                max_delta = 25
                
            dx = random.randint(-max_delta, max_delta)
            dy = random.randint(-max_delta, max_delta)
            
            if dx == 0 and dy == 0:
                continue
                
            nx = new_points[idx][0] + dx
            ny = new_points[idx][1] + dy
            
            if not (-1000 <= nx <= 1000 and -1000 <= ny <= 1000):
                continue
                
            new_point = [nx, ny]
            is_distinct = True
            for i, p in enumerate(new_points):
                if i != idx and p == new_point:
                    is_distinct = False
                    break
            if not is_distinct:
                continue
                
            new_points[idx] = new_point
            
            valid, score = _evaluate(new_points)
            
            if valid:
                delta = score - current_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_points = new_points
                    current_score = score
                    if current_score > best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current_points]
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
            step += 1
            # Restart to best every 500 steps
            if step % 500 == 0:
                current_points = [p[:] for p in best_points]
                current_score = best_score
                T = 10.0
    
    # Final polish phase
    if best_points is not None and time.time() - start_time < time_limit - 5:
        polish_time = min(10.0, time_limit - (time.time() - start_time) - 2)
        best_points = _local_polish(best_points, polish_time)
    
    return best_points
