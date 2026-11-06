"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def _compute_lines(points):
    """Compute all lines and their point sets. Returns dict of line_key -> set of indices."""
    n = len(points)
    line_points = defaultdict(set)
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
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
            line_points[line_key].add(i)
            line_points[line_key].add(j)
    
    return line_points

def _count_lines_exactly_4(line_points):
    """Count lines containing exactly 4 points from precomputed line data."""
    count = 0
    for indices in line_points.values():
        if len(indices) == 4:
            count += 1
    return count

def _no_five_collinear(line_points):
    """Check that no line contains 5 or more points from precomputed line data."""
    for indices in line_points.values():
        if len(indices) >= 5:
            return False
    return True

def _valid(points):
    """Check all constraints: 16 distinct points, |x|,|y| <= 1000, no five collinear."""
    if len(points) != 16:
        return False
    if len(set(tuple(p) for p in points)) != 16:
        return False
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    line_points = _compute_lines(points)
    if not _no_five_collinear(line_points):
        return False
    return True

def _evaluate(points):
    """Compute score and validity in one pass. Returns (is_valid, score)."""
    if len(points) != 16:
        return False, 0
    if len(set(tuple(p) for p in points)) != 16:
        return False, 0
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False, 0
    line_points = _compute_lines(points)
    if not _no_five_collinear(line_points):
        return False, 0
    score = _count_lines_exactly_4(line_points)
    return True, score

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    random.seed(42)
    
    # Start with 4x4 grid
    best_points = [[x, y] for x in range(4) for y in range(4)]
    valid, best_score = _evaluate(best_points)
    if not valid:
        best_score = 0
    
    # Try various initial configurations
    initial_configs = []
    
    # 4x4 grid
    initial_configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # Shifted grid
    initial_configs.append([[x, y] for x in range(4) for y in range(4, 8)])
    
    # Various perturbations of grid
    for trial in range(30):
        pts = [[x, y] for x in range(4) for y in range(4)]
        for _ in range(5):
            idx = random.randint(0, 15)
            pts[idx] = [random.randint(-20, 20), random.randint(-20, 20)]
        initial_configs.append(pts)
    
    # Random configurations
    for trial in range(80):
        pts = []
        for _ in range(16):
            pts.append([random.randint(-30, 30), random.randint(-30, 30)])
        initial_configs.append(pts)
    
    # Evaluate initial configs
    for cfg in initial_configs:
        valid, score = _evaluate(cfg)
        if valid and score > best_score:
            best_score = score
            best_points = [p[:] for p in cfg]
    
    # Local search: move one point at a time with wider neighborhood
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(16):
            if time.time() > deadline:
                break
            old_x, old_y = best_points[i]
            # Try moving point i to nearby positions
            for dx in range(-8, 9):
                for dy in range(-8, 9):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if abs(new_x) > 1000 or abs(new_y) > 1000:
                        continue
                    candidate = [p[:] for p in best_points]
                    candidate[i] = [new_x, new_y]
                    valid, score = _evaluate(candidate)
                    if valid and score > best_score:
                        best_score = score
                        best_points = candidate
                        improved = True
    
    # Second phase: try larger random moves
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(16):
            if time.time() > deadline:
                break
            old_x, old_y = best_points[i]
            for _ in range(200):
                new_x = random.randint(max(-1000, old_x - 30), min(1000, old_x + 30))
                new_y = random.randint(max(-1000, old_y - 30), min(1000, old_y + 30))
                candidate = [p[:] for p in best_points]
                candidate[i] = [new_x, new_y]
                valid, score = _evaluate(candidate)
                if valid and score > best_score:
                    best_score = score
                    best_points = candidate
                    improved = True
    
    # Third phase: swap two points
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(16):
            if time.time() > deadline:
                break
            for j in range(i+1, 16):
                if time.time() > deadline:
                    break
                for _ in range(50):
                    new_xi = random.randint(max(-1000, best_points[i][0] - 20), min(1000, best_points[i][0] + 20))
                    new_yi = random.randint(max(-1000, best_points[i][1] - 20), min(1000, best_points[i][1] + 20))
                    new_xj = random.randint(max(-1000, best_points[j][0] - 20), min(1000, best_points[j][0] + 20))
                    new_yj = random.randint(max(-1000, best_points[j][1] - 20), min(1000, best_points[j][1] + 20))
                    candidate = [p[:] for p in best_points]
                    candidate[i] = [new_xi, new_yi]
                    candidate[j] = [new_xj, new_yj]
                    valid, score = _evaluate(candidate)
                    if valid and score > best_score:
                        best_score = score
                        best_points = candidate
                        improved = True
    
    return best_points
# EVOLVE-BLOCK-END
