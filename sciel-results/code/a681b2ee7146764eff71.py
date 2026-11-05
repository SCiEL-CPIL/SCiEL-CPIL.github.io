"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

N = 64
TIME_LIMIT = 320  # safety margin before 330s kill

# EVOLVE-BLOCK-START
def _fast_greedy_optimized(start_seed, deadline):
    """
    Optimized greedy using pre-allocated arrays and vectorized checks.
    """
    random.seed(start_seed)
    all_points = [(x, y) for x in range(N) for y in range(N)]
    random.shuffle(all_points)
    
    max_points = N * N
    max_dist_sq = 2 * (N - 1) ** 2
    dist_used = np.zeros(max_dist_sq + 1, dtype=bool)
    
    pts_x = np.empty(max_points, dtype=np.int32)
    pts_y = np.empty(max_points, dtype=np.int32)
    count = 0
    
    for p in all_points:
        if time.time() > deadline:
            break
        
        px, py = p
        
        if count == 0:
            pts_x[0] = px
            pts_y[0] = py
            count = 1
            continue
        
        # Vectorized distance computation
        dx = px - pts_x[:count]
        dy = py - pts_y[:count]
        d2_arr = dx * dx + dy * dy
        
        # Check if any distance is already used globally
        if np.any(dist_used[d2_arr]):
            continue
        
        # Check for local duplicates
        if len(np.unique(d2_arr)) != len(d2_arr):
            continue
        
        # Add point
        pts_x[count] = px
        pts_y[count] = py
        count += 1
        
        # Mark distances as used
        dist_used[d2_arr] = True
        
    return [(int(pts_x[i]), int(pts_y[i])) for i in range(count)]

def _local_search_improve(points, deadline):
    """
    Improve the set by removing a random point and re-filling.
    """
    current_points = list(points)
    best_set = current_points
    best_size = len(current_points)
    
    max_iters = 5
    for _ in range(max_iters):
        if time.time() > deadline - 10:
            break
        if not current_points:
            break
            
        # Pick a random point to remove
        idx = random.randint(0, len(current_points) - 1)
        removed_pt = current_points[idx]
        new_points = [p for i, p in enumerate(current_points) if i != idx]
        
        # Re-greedy on remaining grid points
        remaining_points = [p for p in [(x,y) for x in range(N) for y in range(N)] if p not in set(current_points)]
        random.shuffle(remaining_points)
        
        # Setup for vectorized check
        if new_points:
            new_points_arr_x = np.array([p[0] for p in new_points], dtype=np.int32)
            new_points_arr_y = np.array([p[1] for p in new_points], dtype=np.int32)
        else:
            new_points_arr_x = np.array([], dtype=np.int32)
            new_points_arr_y = np.array([], dtype=np.int32)
            
        max_dist_sq = 2 * (N - 1) ** 2
        dist_used_local = np.zeros(max_dist_sq + 1, dtype=bool)
        
        # Initialize with existing distances
        for i in range(len(new_points)):
            for j in range(i+1, len(new_points)):
                dx = new_points[i][0] - new_points[j][0]
                dy = new_points[i][1] - new_points[j][1]
                d2 = dx*dx + dy*dy
                dist_used_local[d2] = True
        
        for p in remaining_points:
            if time.time() > deadline - 10:
                break
            px, py = p
            if len(new_points) == 0:
                new_points.append(p)
                new_points_arr_x = np.array([px], dtype=np.int32)
                new_points_arr_y = np.array([py], dtype=np.int32)
                continue
                
            dx = px - new_points_arr_x
            dy = py - new_points_arr_y
            d2_arr = dx * dx + dy * dy
            
            if np.any(dist_used_local[d2_arr]):
                continue
                
            if len(np.unique(d2_arr)) != len(d2_arr):
                continue
                
            # Add
            new_points.append(p)
            new_points_arr_x = np.append(new_points_arr_x, px)
            new_points_arr_y = np.append(new_points_arr_y, py)
            dist_used_local[d2_arr] = True
            
        if len(new_points) > best_size:
            best_set = new_points
            best_size = len(new_points)
            current_points = new_points
            
    return best_set

def construct():
    """Construct a set of points with no three forming an isosceles triangle."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_points = []
    
    # Run optimized greedy with many restarts
    for seed in range(500):
        if time.time() > deadline - 30:
            break
        
        points = _fast_greedy_optimized(seed, deadline)
        if len(points) > len(best_points):
            best_points = points
            
    # Local search improvement
    if best_points and time.time() < deadline - 10:
        improved_points = _local_search_improve(best_points, deadline)
        if len(improved_points) > len(best_points):
            best_points = improved_points
            
    return best_points
# EVOLVE-BLOCK-END
