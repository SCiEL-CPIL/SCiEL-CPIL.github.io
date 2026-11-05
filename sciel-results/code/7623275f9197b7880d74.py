import time
import random
from math import gcd

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before 150s kill

def _count_lines(points):
    """Count lines by number of points. Returns (ordinary_count, triple_count, all_counts)."""
    n = len(points)
    if n < 2:
        return 0, 0, {}
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
            if g:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
    ordinary = sum(1 for c in line_counts.values() if c == 1)
    triples = sum(1 for c in line_counts.values() if c == 3)
    return ordinary, triples, line_counts

def _objective(points):
    """Objective: Minimize ordinary lines, secondary: Maximize 3-point lines."""
    ord_lines, tri_lines, _ = _count_lines(points)
    return ord_lines * 100 - tri_lines

def _valid(points):
    """12 distinct integer points, |coord|<=1e9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for x, y in points:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        t = (x, y)
        if t in seen:
            return False
        seen.add(t)
    # Check not all collinear
    if len(points) >= 3:
        x0, y0 = points[0]
        x1, y1 = points[1]
        x2, y2 = points[2]
        if (x1 - x0) * (y2 - y0) == (y1 - y0) * (x2 - x0):
            # Check if all others are on this line
            all_collinear = True
            for p in points[3:]:
                if (p[0] - x0) * (y1 - y0) != (p[1] - y0) * (x1 - x0):
                    all_collinear = False
                    break
            if all_collinear:
                return False
    return True

def _random_point():
    return [random.randint(-50, 50), random.randint(-50, 50)]

def _make_seed_grid():
    """3x4 grid seed."""
    points = []
    for y in range(4):
        for x in range(3):
            points.append([x, y])
    return points

def _simulate_annealing(seed_points, max_time=10.0, rng=None):
    if rng is None:
        rng = random.Random(int(time.time() * 1000) % 2**32)
    cur = [p[:] for p in seed_points]
    if not _valid(cur):
        return None, float('inf')
    
    cur_obj = _objective(cur)
    best = [p[:] for p in cur]
    best_obj = cur_obj
    
    T = 1000.0
    T_min = 0.1
    cooling_rate = 0.999
    
    start_time = time.time()
    
    while T > T_min and time.time() - start_time < max_time:
        new_pts = [p[:] for p in cur]
        move_type = rng.random()
        
        if move_type < 0.4:
            # Move one point
            idx = rng.randrange(12)
            new_pts[idx][0] += rng.randint(-5, 5)
            new_pts[idx][1] += rng.randint(-5, 5)
        elif move_type < 0.7:
            # Move two points (to maintain/create collinearity)
            i = rng.randrange(12)
            j = rng.randrange(12)
            if i != j:
                new_pts[i][0] += rng.randint(-5, 5)
                new_pts[i][1] += rng.randint(-5, 5)
                new_pts[j][0] += rng.randint(-5, 5)
                new_pts[j][1] += rng.randint(-5, 5)
        else:
            # Large jump
            idx = rng.randrange(12)
            new_pts[idx] = _random_point()
            
        if not _valid(new_pts):
            T *= cooling_rate
            continue
            
        new_obj = _objective(new_pts)
        delta = new_obj - cur_obj
        
        if delta < 0 or rng.random() < pow(2.71828, -delta / T):
            cur = new_pts
            cur_obj = new_obj
            if cur_obj < best_obj:
                best_obj = cur_obj
                best = [p[:] for p in cur]
                
        T *= cooling_rate
        
    return best, best_obj

def construct():
    best_pts = _make_seed_grid()
    best_score = _objective(best_pts)
    
    seeds = [_make_seed_grid()]
    for _ in range(5):
        pts = [_random_point() for _ in range(12)]
        if _valid(pts):
            seeds.append(pts)
            
    time_per_seed = 15.0
    
    for seed in seeds:
        if time.time() > _DEADLINE - 5:
            break
        pts, score = _simulate_annealing(seed, max_time=time_per_seed)
        if pts and score < best_score:
            best_score = score
            best_pts = pts
            
    if best_pts is None:
        best_pts = _make_seed_grid()
        
    return best_pts
