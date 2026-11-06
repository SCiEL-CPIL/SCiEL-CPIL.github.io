"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def _line_key(p1, p2):
    """Return a canonical key for the line through two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    
    # Normalize direction vector
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    
    g = _gcd(dx, dy)
    if g > 0:
        dx //= g
        dy //= g
        
    # Canonical offset: dy*x1 - dx*y1
    offset = dy * x1 - dx * y1
    return (dx, dy, offset)

def _count_ordinary(points):
    """Count lines containing exactly two of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    line_points = defaultdict(set)
    
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_points[key].add(i)
                line_points[key].add(j)
    
    ordinary = 0
    for key, pts in line_points.items():
        if len(pts) == 2:
            ordinary += 1
            
    return ordinary

def _eval_config(points):
    """Evaluate a configuration: return (ordinary_count, is_valid)."""
    n = len(points)
    if n != 12:
        return (float('inf'), False)
    
    # Check distinctness
    if len(set(map(tuple, points))) != 12:
        return (float('inf'), False)
    
    # Check bounds
    for x, y in points:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return (float('inf'), False)
    
    # Check not all collinear
    (x1, y1), (x2, y2), (x3, y3) = points[0], points[1], points[2]
    cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    if cross == 0:
        all_collinear = True
        for i in range(3, n):
            xi, yi = points[i]
            c = (x2 - x1) * (yi - y1) - (y2 - y1) * (xi - x1)
            if c != 0:
                all_collinear = False
                break
        if all_collinear:
            return (float('inf'), False)
            
    ordinary = _count_ordinary(points)
    return (ordinary, True)

def _find_integer_point_on_line(p1, p2, lo, hi, exclude_set):
    """
    Find an integer point on the line through p1 and p2 within [lo, hi] bounds,
    not in exclude_set.
    Returns None if no such point exists or if the line is not rational/integer-friendly in a simple way.
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        return None
        
    g = _gcd(abs(dx), abs(dy))
    if g == 0:
        return None
        
    dx //= g
    dy //= g
    
    # Points on the line are (x1 + t*dx, y1 + t*dy) for integer t
    # We need x in [lo, hi] and y in [lo, hi]
    
    # Calculate valid range for t
    t_min = float('inf')
    t_max = float('-inf')
    
    if dx > 0:
        t_min = max(t_min, math.ceil((lo - x1) / dx))
        t_max = min(t_max, math.floor((hi - x1) / dx))
    elif dx < 0:
        t_min = max(t_min, math.ceil((hi - x1) / dx))
        t_max = min(t_max, math.floor((lo - x1) / dx))
    else:
        # dx == 0, vertical line
        if x1 < lo or x1 > hi:
            return None
        # t is determined by y
        if dy > 0:
            t_min = max(t_min, math.ceil((lo - y1) / dy))
            t_max = min(t_max, math.floor((hi - y1) / dy))
        elif dy < 0:
            t_min = max(t_min, math.ceil((hi - y1) / dy))
            t_max = min(t_max, math.floor((lo - y1) / dy))
        else:
            return None
            
    if dy != 0:
        if dy > 0:
            t_min = max(t_min, math.ceil((lo - y1) / dy))
            t_max = min(t_max, math.floor((hi - y1) / dy))
        elif dy < 0:
            t_min = max(t_min, math.ceil((hi - y1) / dy))
            t_max = min(t_max, math.floor((lo - y1) / dy))
            
    if t_min > t_max:
        return None
        
    # Iterate through possible t values
    # Limit the number of candidates to avoid long loops
    # If the range is huge, pick a few random t's
    if t_max - t_min > 100:
        candidates = random.sample(range(int(t_min), int(t_max) + 1), 100)
    else:
        candidates = range(int(t_min), int(t_max) + 1)
        
    for t in candidates:
        if t == 0 or t == 1: # p1 and p2
            continue
        px = x1 + t * dx
        py = y1 + t * dy
        p = (px, py)
        if p not in exclude_set:
            return p
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    # Initial configuration: 3x4 grid
    best_points = [[x, y] for x in range(4) for y in range(3)]
    best_score, _ = _eval_config(best_points)
    
    # Try a few other structured starts
    starts = [
        [[x, y] for x in range(4) for y in range(3)],
        [[x, y] for x in range(3) for y in range(4)],
        [[x, y] for x in range(6) for y in range(2)],
        [[x, x*x] for x in range(12)],
    ]
    
    for s in starts:
        score, valid = _eval_config(s)
        if valid and score < best_score:
            best_score = score
            best_points = [p[:] for p in s]
            
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    search_lo = -20
    search_hi = 20
    
    n = 12
    
    while T > T_min:
        if time.time() > deadline:
            break
            
        # Generate a move
        idx = random.randint(0, n - 1)
        old_point = current_points[idx][:]
        
        move_type = random.random()
        
        if move_type < 0.5:
            # Collinearity-seeking move
            # Pick two other points
            others = [i for i in range(n) if i != idx]
            j, k = random.sample(others, 2)
            p_j = current_points[j]
            p_k = current_points[k]
            
            # Find a new point on the line through j and k
            exclude = set(map(tuple, current_points))
            new_point = _find_integer_point_on_line(p_j, p_k, search_lo, search_hi, exclude)
            
            if new_point is None:
                # Fallback to random
                new_x = random.randint(search_lo, search_hi)
                new_y = random.randint(search_lo, search_hi)
                new_point = (new_x, new_y)
            else:
                new_point = list(new_point)
                
        elif move_type < 0.8:
            # Random move within bounds
            new_x = random.randint(search_lo, search_hi)
            new_y = random.randint(search_lo, search_hi)
            new_point = [new_x, new_y]
        else:
            # Small perturbation
            new_x = old_point[0] + random.randint(-2, 2)
            new_y = old_point[1] + random.randint(-2, 2)
            new_point = [new_x, new_y]
            
        # Apply move
        current_points[idx] = new_point
        
        # Evaluate
        new_score, valid = _eval_config(current_points)
        
        if valid:
            delta = new_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
            else:
                # Revert
                current_points[idx] = old_point
        else:
            # Revert
            current_points[idx] = old_point
            
        T *= cooling
        
    return best_points
# EVOLVE-BLOCK-END
