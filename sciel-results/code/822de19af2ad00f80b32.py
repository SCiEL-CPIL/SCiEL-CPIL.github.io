"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def _line_key(p1, p2):
    """Return canonical (a,b,c) for line through p1,p2 where a*x+b*y+c=0, gcd(a,b,c)=1, a>0 or (a==0,b>0)."""
    x1, y1 = p1
    x2, y2 = p2
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    g = _gcd(_gcd(abs(a), abs(b)), abs(c))
    if g > 0:
        a //= g
        b //= g
        c //= g
    # Canonical sign
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_counts[key] = line_counts.get(key, 0) + 1
    ordinary = 0
    for count in line_counts.values():
        if count == 2:
            ordinary += 1
    return ordinary

def _is_valid(points):
    """Check that points are valid: 12 distinct integer points, |x|,|y| <= 10^9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        if len(p) != 2:
            return False
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    # Check not all collinear
    if len(seen) < 3:
        return False
    p0, p1, p2 = list(seen)[:3]
    cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
    if cross == 0:
        # Check all others
        for p in seen:
            if p == p0 or p == p1 or p == p2:
                continue
            cross2 = (p[0] - p0[0]) * (p2[1] - p0[1]) - (p[1] - p0[1]) * (p2[0] - p0[0])
            if cross2 != 0:
                return True
        return False
    return True

def _local_search(points, max_iter=500):
    """Local search: try moving one point to a new position to reduce ordinary lines."""
    current = [p[:] for p in points]
    current_cost = _count_ordinary_lines(current)
    
    for _ in range(max_iter):
        best_move = None
        best_cost = current_cost
        
        # Try moving each point to nearby positions
        for i in range(12):
            x, y = current[i]
            # Try small perturbations
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    new_points = [p[:] for p in current]
                    new_points[i] = [nx, ny]
                    if not _is_valid(new_points):
                        continue
                    cost = _count_ordinary_lines(new_points)
                    if cost < best_cost:
                        best_cost = cost
                        best_move = (i, nx, ny)
        
        if best_move is None:
            break
        
        i, nx, ny = best_move
        current[i] = [nx, ny]
        current_cost = best_cost
    
    return current, current_cost

def _random_restart(start_points, n_restarts=5):
    """Random restarts with local search."""
    best_points = [p[:] for p in start_points]
    best_cost = _count_ordinary_lines(best_points)
    
    for _ in range(n_restarts):
        # Create a random perturbation of the starting points
        perturbed = [p[:] for p in start_points]
        for i in range(12):
            if random.random() < 0.3:
                perturbed[i][0] += random.randint(-5, 5)
                perturbed[i][1] += random.randint(-5, 5)
        
        if not _is_valid(perturbed):
            continue
        
        result, cost = _local_search(perturbed, max_iter=200)
        if cost < best_cost:
            best_cost = cost
            best_points = result
    
    return best_points, best_cost

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Known good configurations to try
    configs = []
    
    # 1. Current Hesse-like configuration
    configs.append([
        [0, 0], [0, 2], [0, 4],
        [2, 0], [2, 2], [2, 4],
        [4, 0], [4, 2], [4, 4],
        [2, 1], [1, 2], [3, 3]
    ])
    
    # 2. 4x3 grid
    configs.append([
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [0, 2], [1, 2], [2, 2], [3, 2]
    ])
    
    # 3. 3x4 grid
    configs.append([
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2], [2, 2],
        [0, 3], [1, 3], [2, 3]
    ])
    
    # 4. Points on a parabola y=x^2 (mod some scaling)
    parabola = []
    for x in range(-3, 4):
        parabola.append([x, x*x])
    configs.append(parabola)
    
    # 5. Mixed: some on lines, some off
    configs.append([
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2],
        [5, 3], [6, 5]
    ])
    
    # 6. Another configuration with more collinearities
    configs.append([
        [0, 0], [1, 1], [2, 2], [3, 3],
        [0, 1], [1, 0], [2, 3], [3, 2],
        [0, 2], [2, 0],
        [4, 4], [5, 5]
    ])
    
    # Evaluate all known configs
    best_points = None
    best_cost = float('inf')
    
    for cfg in configs:
        if not _is_valid(cfg):
            continue
        cost = _count_ordinary_lines(cfg)
        if cost < best_cost:
            best_cost = cost
            best_points = [p[:] for p in cfg]
    
    if best_points is None:
        # Fallback to first config
        best_points = [p[:] for p in configs[0]]
        best_cost = _count_ordinary_lines(best_points)
    
    # Try local search from best config
    if time.time() < deadline - 5.0:
        improved, cost = _local_search(best_points, max_iter=300)
        if cost < best_cost:
            best_cost = cost
            best_points = improved
    
    # Try random restarts if time permits
    if time.time() < deadline - 10.0:
        for cfg in configs[:3]:
            if time.time() >= deadline - 5.0:
                break
            result, cost = _random_restart(cfg, n_restarts=3)
            if cost < best_cost:
                best_cost = cost
                best_points = result
    
    # Final check
    if not _is_valid(best_points):
        # Return a known valid configuration
        best_points = [
            [0, 0], [0, 2], [0, 4],
            [2, 0], [2, 2], [2, 4],
            [4, 0], [4, 2], [4, 4],
            [2, 1], [1, 2], [3, 3]
        ]
    
    return best_points
# EVOLVE-BLOCK-END
