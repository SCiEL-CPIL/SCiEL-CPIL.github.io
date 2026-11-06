"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _evaluate_points(points):
    """
    Evaluate the multiplicity of the most frequent non-zero doubled area
    among all triples of points.
    """
    n = len(points)
    if n < 3:
        return 0
    
    # Precompute all pair differences for speed
    # Use numpy for vectorized determinant computation
    pts = np.array(points, dtype=np.int32)
    
    # Generate all triple indices
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    
    if not i_idx:
        return 0
        
    i_arr = np.array(i_idx, dtype=np.int32)
    j_arr = np.array(j_idx, dtype=np.int32)
    k_arr = np.array(k_idx, dtype=np.int32)
    
    # Compute determinants: (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i)
    x_i = pts[i_arr, 0]
    y_i = pts[i_arr, 1]
    x_j = pts[j_arr, 0]
    y_j = pts[j_arr, 1]
    x_k = pts[k_arr, 0]
    y_k = pts[k_arr, 1]
    
    dets = np.abs((x_j - x_i) * (y_k - y_i) - (y_j - y_i) * (x_k - x_i))
    
    # Exclude zero-area triples
    dets = dets[dets > 0]
    
    if len(dets) == 0:
        return 0
        
    # Count multiplicities
    # Using np.unique with return_counts
    unique_areas, counts = np.unique(dets, return_counts=True)
    
    return int(counts.max())

def _local_search(points, max_iters=500, time_limit=30.0):
    """
    Perform local search by swapping points to improve the score.
    """
    start_time = time.time()
    points = list(points)
    current_score = _evaluate_points(points)
    best_points = list(points)
    best_score = current_score
    
    # Define a set of candidate points to swap in/out
    # We'll use a subset of the bounding box for efficiency
    # Let's consider points in [-15, 15]^2 for the pool, but ensure we don't pick existing ones
    # Actually, simpler: try moving one point to a nearby random integer coordinate
    
    import random
    random.seed(42)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random point to move
        idx = random.randint(0, len(points) - 1)
        old_p = points[idx]
        
        # Generate a new random point in a local neighborhood or global
        # Let's try a global random point in [-30, 30]^2
        # But we must ensure it's distinct from all other points
        new_p = (random.randint(-30, 30), random.randint(-30, 30))
        
        # Check if new_p is already in points (excluding the one we're replacing)
        if new_p in points and points.index(new_p) != idx:
            continue
            
        # Temporarily replace
        points[idx] = new_p
        
        # Evaluate
        new_score = _evaluate_points(points)
        
        if new_score > best_score:
            best_score = new_score
            best_points = list(points)
        else:
            # Revert
            points[idx] = old_p
            
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin
    
    # Candidate 1: 6x5 Grid (known to be good)
    grid_6x5 = [(x, y) for x in range(6) for y in range(5)]
    
    # Candidate 2: 5x6 Grid (same as 6x5 effectively, but let's check orientation doesn't matter for det)
    # Candidate 3: 10x3 Grid
    grid_10x3 = [(x, y) for x in range(10) for y in range(3)]
    
    # Candidate 4: 15x2 Grid (from current program, but score was lower)
    grid_15x2 = [(x, y) for x in range(15) for y in range(2)]
    
    # Candidate 5: 8x4 Grid minus 2 points
    full_8x4 = [(x, y) for x in range(8) for y in range(4)]
    to_remove_8x4 = [(0, 0), (7, 3)]
    grid_8x4_mod = [p for p in full_8x4 if p not in to_remove_8x4]
    
    # Candidate 6: 7x5 Grid minus 5 points
    full_7x5 = [(x, y) for x in range(7) for y in range(5)]
    to_remove_7x5 = [(0, 0), (6, 0), (0, 4), (6, 4), (0, 2)]
    grid_7x5_mod = [p for p in full_7x5 if p not in to_remove_7x5]
    
    candidates = [grid_6x5, grid_10x3, grid_15x2, grid_8x4_mod, grid_7x5_mod]
    
    best_points = grid_6x5
    best_score = 0
    
    for pts in candidates:
        if time.time() > deadline:
            break
        if len(pts) != 30:
            continue
        valid = all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts)
        if not valid:
            continue
        score = _evaluate_points(pts)
        if score > best_score:
            best_score = score
            best_points = list(pts)
            
    # Local search refinement
    if time.time() < deadline - 5.0:
        refined_points, refined_score = _local_search(best_points, max_iters=200, time_limit=deadline - time.time() - 2.0)
        if refined_score > best_score:
            best_points = refined_points
            best_score = refined_score
            
    return best_points
# EVOLVE-BLOCK-END
