# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from math import gcd
from itertools import combinations

def count_4_lines(points):
    """Count lines with exactly 4 points.
    Returns (count, has_five_collinear)."""
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    
    line_keys = set()
    has_five = False
    
    for i, j in combinations(range(n), 2):
        x1, y1 = pts[i]
        x2, y2 = pts[j]
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            continue
        
        g = gcd(abs(dx), abs(dy))
        dx_n = dx // g
        dy_n = dy // g
        
        # Canonical direction
        if dx_n < 0 or (dx_n == 0 and dy_n < 0):
            dx_n, dy_n = -dx_n, -dy_n
        
        # Count points on this line
        diffs_x = pts[:, 0] - x1
        diffs_y = pts[:, 1] - y1
        on_line = (diffs_x * dy_n == diffs_y * dx_n)
        cnt = int(np.sum(on_line))
        
        if cnt >= 5:
            return 0, True
        
        if cnt >= 4:
            c = dy_n * x1 - dx_n * y1
            line_keys.add((dx_n, dy_n, c))
    
    return len(line_keys), False

def local_search_selection(grid_size, deadline, initial_subset=None):
    """
    Local search on a subset of 16 points from a grid.
    """
    # Generate grid points
    grid_points = []
    for x in range(-grid_size, grid_size + 1):
        for y in range(-grid_size, grid_size + 1):
            grid_points.append((x, y))
            
    total_points = len(grid_points)
    
    # Initial subset
    if initial_subset is None:
        # Pick 16 random points
        subset_indices = set(random.sample(range(total_points), 16))
    else:
        # Map points to indices
        point_to_idx = {p: i for i, p in enumerate(grid_points)}
        subset_indices = set(point_to_idx[p] for p in initial_subset)
        
    in_subset = [grid_points[i] for i in subset_indices]
    out_subset_indices = [i for i in range(total_points) if i not in subset_indices]
    out_subset = [grid_points[i] for i in out_subset_indices]
    
    current_score, has_five = count_4_lines(in_subset)
    if has_five:
        current_score = -100
    
    best_score = current_score
    best_points = in_subset[:]
    
    temp = 1.0
    cooling = 0.999
    min_temp = 0.01
    
    while temp > min_temp:
        if time.time() > deadline - 1.0:
            break
        
        # Move: Swap a point in subset with a point not in subset
        idx_in = random.randint(0, 15)
        idx_out = random.randint(0, len(out_subset) - 1)
        
        p_in = in_subset[idx_in]
        p_out = out_subset[idx_out]
        
        # Apply move
        in_subset[idx_in] = p_out
        out_subset[idx_out] = p_in
        
        new_score, has_five = count_4_lines(in_subset)
        if has_five:
            new_score = -100
        
        delta = new_score - current_score
        if delta >= 0 or random.random() < math.exp(-delta / temp):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = in_subset[:]
        else:
            # Revert
            in_subset[idx_in] = p_in
            out_subset[idx_out] = p_out
            
        temp *= cooling
        
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_score = -1
    best_points = None
    
    # Try different grid sizes
    for grid_size in [5, 8, 12]:
        if time.time() > deadline - 10.0:
            break
            
        # Run local search
        points, score = local_search_selection(grid_size, deadline)
        
        if score > best_score:
            best_score = score
            best_points = points
            
        # Try a few random restarts
        for _ in range(10):
            if time.time() > deadline - 5.0:
                break
            points, score = local_search_selection(grid_size, deadline)
            if score > best_score:
                best_score = score
                best_points = points
                
    if best_points is None:
        # Fallback to grid
        best_points = [[x, y] for x in range(4) for y in range(4)]
        best_score, _ = count_4_lines(best_points)
        
    return best_points
# EVOLVE-BLOCK-END
