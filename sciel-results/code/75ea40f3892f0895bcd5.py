import random
import time
from math import gcd

def count_exact_4_lines_fast(points):
    """
    Fast counting of exact 4-point lines using bitmasks.
    Returns 0 if invalid (duplicates or 5+ collinear).
    """
    n = len(points)
    if n != 16:
        return 0
    
    # Check duplicates
    if len(set(map(tuple, points))) != 16:
        return 0

    lines = {}
    
    # Pre-extract coordinates for speed
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    for i in range(n):
        x1, y1 = xs[i], ys[i]
        for j in range(i + 1, n):
            x2, y2 = xs[j], ys[j]
            
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            
            if a == 0 and b == 0:
                continue # Should not happen with distinct points
            
            g = gcd(gcd(abs(a), abs(b)), abs(c))
            if g == 0:
                continue
            a //= g
            b //= g
            c //= g
            
            # Canonicalize
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
            
            key = (a, b, c)
            mask = (1 << i) | (1 << j)
            
            if key in lines:
                lines[key] |= mask
            else:
                lines[key] = mask
                
    count_4 = 0
    for mask in lines.values():
        # Count set bits
        # Python 3.10+ has int.bit_count()
        # Fallback: bin(mask).count('1')
        k = mask.bit_count() if hasattr(mask, 'bit_count') else bin(mask).count('1')
        
        if k > 4:
            return 0
        if k == 4:
            count_4 += 1
            
    return count_4

def construct():
    # Set random seed for reproducibility
    random.seed(42)
    
    start_time = time.time()
    time_limit = 115.0 # Leave 5s buffer
    
    best_points = None
    best_score = -1
    
    # 1. Greedy Initialization
    # Try a few small grids to find a good seed
    for g_size in [5, 8, 12]:
        candidates = [(x, y) for x in range(-g_size, g_size + 1) for y in range(-g_size, g_size + 1)]
        random.shuffle(candidates)
        
        points = []
        for _ in range(16):
            best_p = None
            best_s = -1
            # Only check a subset of candidates for speed in greedy phase
            # Or all if grid is small
            for p in candidates:
                if p in points:
                    continue
                test_pts = points + [p]
                s = count_exact_4_lines_fast(test_pts)
                if s > best_s:
                    best_s = s
                    best_p = p
            if best_p is not None:
                points.append(best_p)
            else:
                break
                
        if len(points) == 16:
            score = count_exact_4_lines_fast(points)
            if score > best_score:
                best_score = score
                best_points = [p[:] for p in points]
                
    if best_points is None:
        # Fallback to 4x4 grid
        best_points = [[x, y] for x in range(4) for y in range(4)]
        best_score = count_exact_4_lines_fast(best_points)
        
    # 2. Hill Climbing / Simulated Annealing
    # Search space: [-50, 50]
    min_coord = -50
    max_coord = 50
    
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    # Adaptive step size
    step_size = 10
    
    while time.time() - start_time < time_limit:
        # Move a random point
        idx = random.randint(0, 15)
        old_p = current_points[idx]
        
        # New position: random within step_size of old, or random in whole grid
        if random.random() < 0.8:
            new_x = old_p[0] + random.randint(-step_size, step_size)
            new_y = old_p[1] + random.randint(-step_size, step_size)
        else:
            new_x = random.randint(min_coord, max_coord)
            new_y = random.randint(min_coord, max_coord)
            
        new_x = max(min_coord, min(max_coord, new_x))
        new_y = max(min_coord, min(max_coord, new_y))
        
        if new_x == old_p[0] and new_y == old_p[1]:
            continue
            
        current_points[idx] = [new_x, new_y]
        new_score = count_exact_4_lines_fast(current_points)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
            # Reduce step size on success?
            # step_size = max(1, int(step_size * 0.9))
        else:
            current_points[idx] = old_p
            # Increase step size if stuck?
            # Keep simple: fixed step or slight randomization
            
    return best_points
