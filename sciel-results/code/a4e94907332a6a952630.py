"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np
from itertools import combinations

def _compute_best_area_multiplicity(points):
    """Compute the largest multiplicity of a single positive doubled area."""
    from collections import Counter
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                det = abs(int(pts[i][0])*(pts[j][1]-pts[k][1]) + 
                         int(pts[j][0])*(pts[k][1]-pts[i][1]) + 
                         int(pts[k][0])*(pts[i][1]-pts[j][1]))
                if det > 0:
                    areas.append(det)
    if not areas:
        return 0
    counter = Counter(areas)
    return counter.most_common(1)[0][1]

def _make_grid(w, h, offset_x=0, offset_y=0):
    """Create a w x h grid of points."""
    points = []
    for x in range(w):
        for y in range(h):
            points.append((x + offset_x, y + offset_y))
    return points

def _make_hexagonal(rows, cols):
    """Create a hexagonal-like grid."""
    points = []
    for r in range(rows):
        for c in range(cols):
            x = c + (r % 2) * 0  # simple grid for now
            y = r
            points.append((x, y))
    return points

def _compute_areas_for_point(points, idx):
    """Compute areas of all triangles involving points[idx]."""
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    areas = []
    pi = pts[idx]
    for j in range(n):
        if j == idx: continue
        pj = pts[j]
        for k in range(j+1, n):
            if k == idx: continue
            pk = pts[k]
            det = abs(int(pi[0])*(int(pj[1]) - int(pk[1])) + 
                      int(pj[0])*(int(pk[1]) - int(pi[1])) + 
                      int(pk[0])*(int(pi[1]) - int(pj[1])))
            if det > 0:
                areas.append(det)
    return areas

def _local_search_optimized(initial_points, steps=5000):
    """Optimized local search maintaining area counts."""
    current_points = [p[:] for p in initial_points]
    n = len(current_points)
    
    # Initialize Counter with all areas
    all_areas = []
    pts_arr = np.array(current_points, dtype=np.int64)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                det = abs(int(pts_arr[i][0])*(pts_arr[j][1]-pts_arr[k][1]) + 
                         int(pts_arr[j][0])*(pts_arr[k][1]-pts_arr[i][1]) + 
                         int(pts_arr[k][0])*(pts_arr[i][1]-pts_arr[j][1]))
                if det > 0:
                    all_areas.append(det)
    
    from collections import Counter
    area_counts = Counter(all_areas)
    current_score = area_counts.most_common(1)[0][1] if area_counts else 0
    
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    for step in range(steps):
        i = random.randint(0, n - 1)
        x, y = current_points[i]
        
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
        if dx == 0 and dy == 0:
            continue
            
        nx, ny = x + dx, y + dy
        
        if not (-30 <= nx <= 30 and -30 <= ny <= 30):
            continue
            
        new_point = (nx, ny)
        is_distinct = True
        for j, p in enumerate(current_points):
            if j != i and p == new_point:
                is_distinct = False
                break
                
        if not is_distinct:
            continue
            
        # Remove old areas involving i
        old_areas = _compute_areas_for_point(current_points, i)
        for a in old_areas:
            area_counts[a] -= 1
            if area_counts[a] == 0:
                del area_counts[a]
                
        # Move point
        current_points[i] = new_point
        
        # Add new areas involving i
        new_areas = _compute_areas_for_point(current_points, i)
        for a in new_areas:
            area_counts[a] += 1
            
        new_score = area_counts.most_common(1)[0][1] if area_counts else 0
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            # Revert
            current_points[i] = (x, y)
            # Revert counts
            for a in new_areas:
                area_counts[a] -= 1
                if area_counts[a] == 0:
                    del area_counts[a]
            for a in old_areas:
                area_counts[a] += 1
                
    return best_points, best_score

def _try_configurations():
    """Try various configurations and return the best one."""
    initial_configs = []
    
    # 6x5 Grid centered
    grid_6_5 = _make_grid(6, 5, -2, -2)
    initial_configs.append(grid_6_5)
    
    # 5x6 Grid centered
    grid_5_6 = _make_grid(5, 6, -2, -2)
    initial_configs.append(grid_5_6)
    
    # Random initial config
    random.seed(42)
    random_points = set()
    while len(random_points) < 30:
        x = random.randint(-10, 10)
        y = random.randint(-10, 10)
        random_points.add((x, y))
    initial_configs.append(list(random_points))
    
    best_pts = None
    best_mult = -1
    
    for config in initial_configs:
        pts, score = _local_search_optimized(config, steps=5000)
        if score > best_mult:
            best_mult = score
            best_pts = pts
            
    if best_pts is None:
        best_pts = _make_grid(6, 5)
        
    return best_pts

def construct():
    return _try_configurations()
