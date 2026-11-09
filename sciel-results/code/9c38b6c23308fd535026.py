import time
import sys
import random
from math import gcd

def _count_ordinary_lines(points):
    """Count the number of lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    
    from collections import defaultdict
    line_map = defaultdict(set)
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            a = dy
            b = -dx
            c = a * x1 + b * y1
            g2 = gcd(abs(a), abs(b))
            if g2 > 0:
                a //= g2
                b //= g2
                c //= g2
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            line_map[(a, b, c)].add(i)
            line_map[(a, b, c)].add(j)
    
    ordinary = 0
    for line, pts in line_map.items():
        if len(pts) == 2:
            ordinary += 1
    
    return ordinary

def _try_config(points):
    """Validate and count ordinary lines for a configuration."""
    if len(points) != 12:
        return None
    if len(set(tuple(p) for p in points)) != 12:
        return None
    for p in points:
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return None
    collinear = True
    for i in range(1, 12):
        dx = points[i][0] - points[0][0]
        dy = points[i][1] - points[0][1]
        for j in range(i + 1, 12):
            dx2 = points[j][0] - points[0][0]
            dy2 = points[j][1] - points[0][1]
            if dx * dy2 != dy * dx2:
                collinear = False
                break
        if not collinear:
            break
    if collinear:
        return None
    return _count_ordinary_lines(points)

def _make_seed3_lines():
    """3 concurrent lines through origin, 4 points each (excluding origin)."""
    pts = []
    # Line y=0
    for i in range(1, 5):
        pts.append([i, 0])
    # Line y=x
    for i in range(1, 5):
        pts.append([i, i])
    # Line y=2x
    for i in range(1, 5):
        pts.append([i, 2*i])
    return pts

def _make_seed_grid():
    """4x3 grid."""
    return [[x, y] for x in range(4) for y in range(3)]

def _make_seed_parallel():
    """Two parallel lines, 6 points each."""
    return [[i, 0] for i in range(6)] + [[i, 1] for i in range(6)]

def _make_seed_cross():
    """Cross: 6 on x-axis, 6 on y-axis (origin shared)."""
    pts = [[i, 0] for i in range(1, 7)] + [[0, i] for i in range(1, 7)]
    return pts

def _simulated_annealing(start_pts, time_limit):
    """Run simulated annealing to minimize ordinary lines."""
    deadline = time.time() + time_limit
    current = [list(p) for p in start_pts]
    current_score = _try_config(current)
    if current_score is None:
        return None, float('inf')
    
    best = [list(p) for p in current]
    best_score = current_score
    
    temp = 10.0
    cooling = 0.995
    rng = random.Random(42)
    
    while time.time() < deadline and temp > 0.1:
        # Pick a random point to move
        idx = rng.randint(0, 11)
        ox, oy = current[idx]
        # Random perturbation
        dx = rng.randint(-5, 5)
        dy = rng.randint(-5, 5)
        if dx == 0 and dy == 0:
            continue
        nx, ny = ox + dx, oy + dy
        if abs(nx) > 10**9 or abs(ny) > 10**9:
            continue
        
        # Check distinctness
        new_pt = [nx, ny]
        if any(tuple(new_pt) == tuple(current[j]) for j in range(12) if j != idx):
            continue
        
        trial = [list(p) for p in current]
        trial[idx] = new_pt
        trial_score = _try_config(trial)
        
        if trial_score is not None:
            delta = trial_score - current_score
            if delta < 0 or rng.random() < pow(2.718281828, -delta / temp):
                current = trial
                current_score = trial_score
                if current_score < best_score:
                    best_score = current_score
                    best = [list(p) for p in current]
        
        temp *= cooling
    
    return best, best_score

def construct():
    """
    Constructs 12 distinct integer points minimizing ordinary lines.
    Uses simulated annealing from multiple diverse seeds.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Generate diverse seeds
    seeds = [
        _make_seed3_lines(),
        _make_seed_grid(),
        _make_seed_parallel(),
        _make_seed_cross(),
        # 3 concurrent lines with different slopes
        [[i, 0] for i in range(1, 5)] + [[i, i] for i in range(1, 5)] + [[i, 3*i] for i in range(1, 5)],
        # 4x3 grid shifted
        [[x+1, y+1] for x in range(4) for y in range(3)],
        # 3 lines through (0,0) with different spacing
        [[i, 0] for i in range(1, 5)] + [[2*i, i] for i in range(1, 5)] + [[i, 2*i] for i in range(1, 5)],
    ]
    
    best_points = None
    best_count = float('inf')
    
    # Evaluate all seeds first
    for seed in seeds:
        if time.time() > deadline:
            break
        count = _try_config(seed)
        if count is not None and count < best_count:
            best_count = count
            best_points = [list(p) for p in seed]
    
    if best_points is None:
        best_points = _make_seed3_lines()
        best_count = _try_config(best_points)
    
    # Run simulated annealing from the best seed and a few others
    seeds_to_anneal = [best_points]
    # Add a couple more diverse seeds for SA
    for s in seeds[:3]:
        if [list(p) for p in s] != best_points:
            seeds_to_anneal.append([list(p) for p in s])
    
    for seed_pts in seeds_to_anneal:
        if time.time() > deadline - 10:
            break
        remaining = deadline - time.time()
        if remaining < 5:
            break
        sa_pts, sa_count = _simulated_annealing(seed_pts, remaining - 5)
        if sa_pts is not None and sa_count < best_count:
            best_count = sa_count
            best_points = sa_pts
    
    return best_points
