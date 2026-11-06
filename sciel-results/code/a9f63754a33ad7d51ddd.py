"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from collections import defaultdict
from math import gcd

def _compute_line_counts(points):
    """Compute line counts for all lines through pairs of points.
    Returns (count_exact_4, has_five_collinear, line_point_counts_dict)"""
    n = len(points)
    if n < 4:
        return 0, False, {}
    
    # Use a dict mapping line_key -> set of point indices
    line_set = defaultdict(set)
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            c = dy * x1 - dx * y1
            line_key = (dx, dy, c)
            line_set[line_key].add(i)
            line_set[line_key].add(j)
    
    count_exact_4 = 0
    has_five = False
    for line_key, idx_set in line_set.items():
        cnt = len(idx_set)
        if cnt == 4:
            count_exact_4 += 1
        elif cnt >= 5:
            has_five = True
    
    return count_exact_4, has_five, dict(line_set)

def _score(points):
    """Return (score, valid) for a configuration."""
    if len(points) != 16:
        return 0, False
    if len(set(tuple(p) for p in points)) != 16:
        return 0, False
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return 0, False
    score, has_five, _ = _compute_line_counts(points)
    if has_five:
        return 0, False
    return score, True

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    random.seed(42)
    
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score, _ = _score(best_points)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 4x4 grid
    initial_configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # Affine plane of order 4 embedded in integers
    # AG(2,4) has 16 points, 35 lines of 4 points
    # We can embed it using modular arithmetic mapped to integers
    # Points: (i, j) for i,j in Z_4, mapped to integer coords
    # Use a field embedding: represent GF(4) elements as pairs
    # Simpler: use a known integer configuration
    
    # Try configurations based on different grid spacings
    for sx in [1, 2, 3]:
        for sy in [1, 2, 3]:
            cfg = [[x*sx, y*sy] for x in range(4) for y in range(4)]
            initial_configs.append(cfg)
    
    # Perturbed grids
    for trial in range(30):
        pts = [[x, y] for x in range(4) for y in range(4)]
        for _ in range(random.randint(1, 8)):
            idx = random.randint(0, 15)
            pts[idx] = [random.randint(-15, 15), random.randint(-15, 15)]
        initial_configs.append(pts)
    
    # Random configurations
    for trial in range(50):
        pts = []
        for _ in range(16):
            pts.append([random.randint(-20, 20), random.randint(-20, 20)])
        initial_configs.append(pts)
    
    # Evaluate initial configs
    for cfg in initial_configs:
        score, valid = _score(cfg)
        if valid and score > best_score:
            best_score = score
            best_points = [p[:] for p in cfg]
    
    # Local search with multiple phases
    # Phase 1: Fine-grained local moves
    improved = True
    max_iter = 0
    while improved and time.time() < deadline:
        improved = False
        max_iter += 1
        if max_iter > 200:
            break
        for i in range(16):
            if time.time() > deadline:
                break
            old_x, old_y = best_points[i]
            # Try small moves
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if abs(new_x) > 1000 or abs(new_y) > 1000:
                        continue
                    candidate = [p[:] for p in best_points]
                    candidate[i] = [new_x, new_y]
                    score, valid = _score(candidate)
                    if valid and score > best_score:
                        best_score = score
                        best_points = candidate
                        improved = True
    
    # Phase 2: Medium moves
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(16):
            if time.time() > deadline:
                break
            old_x, old_y = best_points[i]
            for _ in range(50):
                new_x = random.randint(max(-1000, old_x - 10), min(1000, old_x + 10))
                new_y = random.randint(max(-1000, old_y - 10), min(1000, old_y + 10))
                candidate = [p[:] for p in best_points]
                candidate[i] = [new_x, new_y]
                score, valid = _score(candidate)
                if valid and score > best_score:
                    best_score = score
                    best_points = candidate
                    improved = True
    
    # Phase 3: Simulated annealing style - accept worse moves sometimes
    if time.time() < deadline - 10:
        current = [p[:] for p in best_points]
        cur_score = best_score
        T = 2.0
        for step in range(5000):
            if time.time() > deadline - 5:
                break
            T = max(0.01, T * 0.995)
            i = random.randint(0, 15)
            old_x, old_y = current[i]
            new_x = old_x + random.randint(-8, 8)
            new_y = old_y + random.randint(-8, 8)
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
            candidate = [p[:] for p in current]
            candidate[i] = [new_x, new_y]
            score, valid = _score(candidate)
            if valid:
                delta = score - cur_score
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    current = candidate
                    cur_score = score
                    if cur_score > best_score:
                        best_score = cur_score
                        best_points = [p[:] for p in current]
    
    return best_points
# EVOLVE-BLOCK-END
