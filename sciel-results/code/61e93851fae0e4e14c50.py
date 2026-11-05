"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
from math import gcd

_start_time = time.time()
_DEADLINE = 140  # seconds, safety margin for 150s kill

def _count_four_lines(points):
    """Count lines containing exactly 4 of the given points using canonical line signatures."""
    n = len(points)
    # Build a dictionary: line_signature -> set of point indices
    line_map = {}
    for i in range(n):
        for j in range(i+1, n):
            p1, p2 = points[i], points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            # Normalize direction
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            # Canonical line representation: (a, b, c) where ax + by = c
            # a = dy, b = -dx, c = a*x1 + b*y1
            a, b = dy, -dx
            c = a * p1[0] + b * p1[1]
            # Normalize sign: make (a,b) canonical
            if a < 0 or (a == 0 and b < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    
    count = 0
    for key, pts in line_map.items():
        if len(pts) == 4:
            count += 1
    return count

def _is_valid(points):
    """Check that all points are distinct, |x|,|y| <= 1000, and no five collinear."""
    if len(points) != 16:
        return False
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    if len(set(tuple(p) for p in points)) != 16:
        return False
    # Check no five collinear using line map
    line_map = {}
    n = 16
    for i in range(n):
        for j in range(i+1, n):
            p1, p2 = points[i], points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            a, b = dy, -dx
            c = a * p1[0] + b * p1[1]
            if a < 0 or (a == 0 and b < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    for key, pts in line_map.items():
        if len(pts) >= 5:
            return False
    return True

def _affine_transform(points, a, b, c, d, e, f):
    """Apply affine transform: (x,y) -> (ax+by+e, cx+dy+f)"""
    result = []
    for x, y in points:
        nx = a*x + b*y + e
        ny = c*x + d*y + f
        result.append([nx, ny])
    return result

def _initial_grid():
    return [[x, y] for x in range(4) for y in range(4)]

def _try_move_fast(points, idx, dx, dy):
    """Try moving point idx by (dx, dy). Only check lines through the moved point for 5-collinear."""
    new_pts = [list(p) for p in points]
    new_pts[idx] = [new_pts[idx][0] + dx, new_pts[idx][1] + dy]
    x, y = new_pts[idx]
    if abs(x) > 1000 or abs(y) > 1000:
        return None
    # Check distinct
    for j in range(16):
        if j != idx and new_pts[j] == new_pts[idx]:
            return None
    # Check no five collinear: only need to check lines through point idx
    # For each other point j, check the line through idx and j
    n = 16
    checked_lines = set()
    for j in range(n):
        if j == idx:
            continue
        p1, p2 = new_pts[idx], new_pts[j]
        ddx = p2[0] - p1[0]
        ddy = p2[1] - p1[1]
        g = gcd(abs(ddx), abs(ddy))
        if g > 0:
            ddx //= g
            ddy //= g
        a, b = ddy, -ddx
        c = a * p1[0] + b * p1[1]
        if a < 0 or (a == 0 and b < 0):
            a, b, c = -a, -b, -c
        key = (a, b, c)
        if key in checked_lines:
            continue
        checked_lines.add(key)
        # Count points on this line
        cnt = 0
        for k in range(n):
            px, py = new_pts[k]
            if a * px + b * py == c:
                cnt += 1
                if cnt >= 5:
                    return None
    return new_pts

def _local_search(points, time_budget=10.0):
    """Run local search / simulated annealing on a starting configuration."""
    start = time.time()
    current = [list(p) for p in points]
    current_score = _count_four_lines(current)
    best = [list(p) for p in current]
    best_score = current_score
    
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    while time.time() - start < time_budget and T > T_min:
        idx = random.randint(0, 15)
        # Mix of small and medium moves
        if random.random() < 0.7:
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
        else:
            dx = random.randint(-10, 10)
            dy = random.randint(-10, 10)
        if dx == 0 and dy == 0:
            T *= cooling_rate
            continue
            
        new_pts = _try_move_fast(current, idx, dx, dy)
        if new_pts is None:
            T *= cooling_rate
            continue
            
        new_score = _count_four_lines(new_pts)
        
        delta = new_score - current_score
        if delta > 0:
            current = new_pts
            current_score = new_score
            if current_score > best_score:
                best = [list(p) for p in current]
                best_score = current_score
        else:
            p = math.exp(delta / T)
            if random.random() < p:
                current = new_pts
                current_score = new_score
        
        T *= cooling_rate
        
        # Occasional larger perturbation
        if random.random() < 0.02:
            for _ in range(3):
                idx = random.randint(0, 15)
                dx = random.randint(-20, 20)
                dy = random.randint(-20, 20)
                new_pts = _try_move_fast(current, idx, dx, dy)
                if new_pts is not None:
                    ns = _count_four_lines(new_pts)
                    if ns > current_score:
                        current = new_pts
                        current_score = ns
                        if current_score > best_score:
                            best = [list(p) for p in current]
                            best_score = current_score
                        break
                        
    return best, best_score

def construct():
    global _start_time
    random.seed(42)
    
    base_grid = _initial_grid()
    
    # Generate several affine transforms of the base grid
    # These preserve collinearity structure but change coordinates
    transforms = [
        (1, 0, 0, 1, 0, 0),      # Identity
        (2, 0, 0, 2, 0, 0),      # Scale by 2
        (1, 1, 0, 1, 0, 0),      # Shear
        (2, 1, 1, 2, 0, 0),      # Mixed
        (1, 0, 0, 2, 0, 0),      # Y-scale
        (2, 0, 0, 1, 0, 0),      # X-scale
        (1, -1, 0, 1, 0, 0),     # Another shear
        (3, 0, 0, 3, 0, 0),      # Scale by 3
        (1, 2, 2, 1, 0, 0),      # Another mixed
        (2, 1, -1, 2, 0, 0),     # More complex
    ]
    
    best_overall = None
    best_score_overall = 0
    
    # Time per local search
    elapsed = time.time() - _start_time
    remaining = _DEADLINE - elapsed
    n_starts = min(len(transforms), max(1, int(remaining / 12)))
    
    for t_idx in range(n_starts):
        a, b, c, d, e, f = transforms[t_idx]
        transformed = _affine_transform(base_grid, a, b, c, d, e, f)
        
        # Check validity
        if not _is_valid(transformed):
            continue
        
        # Run local search
        time_budget = min(12.0, _DEADLINE - (time.time() - _start_time))
        if time_budget < 2.0:
            break
            
        local_best, local_score = _local_search(transformed, time_budget)
        
        if local_score > best_score_overall:
            best_score_overall = local_score
            best_overall = local_best
    
    # If no valid transform found, fall back to original grid
    if best_overall is None:
        best_overall = base_grid
        best_score_overall = _count_four_lines(base_grid)
    
    # Final polish: try to improve the best solution with more iterations
    elapsed = time.time() - _start_time
    remaining = _DEADLINE - elapsed
    if remaining > 5.0:
        polished, polished_score = _local_search(best_overall, remaining - 2.0)
        if polished_score > best_score_overall:
            best_overall = polished
            best_score_overall = polished_score
    
    return [[int(x), int(y)] for x, y in best_overall]

# EVOLVE-BLOCK-END
