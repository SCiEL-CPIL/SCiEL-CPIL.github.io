import numpy as np
import time
import random
from collections import Counter

def _vectorized_compute_areas(points):
    """
    Compute absolute doubled areas for all triples of 30 points.
    points: numpy array of shape (30, 2)
    Returns: numpy array of shape (4500,) with absolute determinants.
    """
    n = len(points)
    if n < 3:
        return np.array([])
    
    # Indices for i < j < k
    # We can generate indices efficiently
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    # Filter for i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    # Extract coordinates
    x1 = points[i_flat, 0]
    y1 = points[i_flat, 1]
    x2 = points[j_flat, 0]
    y2 = points[j_flat, 1]
    x3 = points[k_flat, 0]
    y3 = points[k_flat, 1]
    
    # Determinant formula: x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)
    dets = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    areas = np.abs(dets)
    
    return areas

def _get_score(areas):
    """
    Compute the largest multiplicity of a single positive area.
    """
    if len(areas) == 0:
        return 0
    
    # Filter out zero areas
    pos_areas = areas[areas > 0]
    if len(pos_areas) == 0:
        return 0
    
    # Use np.unique for fast counting
    _, counts = np.unique(pos_areas, return_counts=True)
    return int(np.max(counts))

def _local_search(initial_points, time_limit=50.0):
    """
    Perform Simulated Annealing to maximize the score.
    """
    start_time = time.time()
    best_points = np.array(initial_points, dtype=np.int64)
    best_score = _get_score(_vectorized_compute_areas(best_points))
    
    current_points = best_points.copy()
    current_score = best_score
    
    # Precompute all possible coordinates in [-30, 30]^2
    # Total 61*61 = 3721 points. We need to pick 30 distinct ones.
    # We maintain a list of occupied points and a set for O(1) lookup.
    all_coords = np.array([(x, y) for x in range(-30, 31) for y in range(-30, 31)])
    occupied_set = set(map(tuple, current_points))
    
    # Temperature schedule
    T = 1.0
    cooling_rate = 0.9995
    min_T = 0.01
    
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() - start_time < time_limit:
        if not time_ok_global(5):
            break
            
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_p = tuple(current_points[idx])
        
        # Pick a random new position
        # To avoid collision, we can pick randomly until we find an empty spot
        # Or we can swap with another point. Moving is simpler.
        # Let's try swapping with a random other point first, or moving to random empty.
        # Moving to random empty is more exploratory.
        
        # Efficient way to pick random empty point:
        # Since 30/3721 is small, we can just sample a few times.
        new_p = None
        for _ in range(10):
            cand = tuple(all_coords[random.randint(0, len(all_coords)-1)])
            if cand not in occupied_set:
                new_p = cand
                break
        
        if new_p is None:
            # Fallback: swap with a random other point
            swap_idx = random.randint(0, 29)
            if swap_idx == idx:
                continue
            # Swap
            current_points[idx], current_points[swap_idx] = current_points[swap_idx], current_points[idx]
            new_score = _get_score(_vectorized_compute_areas(current_points))
            
            if new_score > current_score:
                current_score = new_score
                no_improve_count = 0
                if new_score > best_score:
                    best_score = new_score
                    best_points = current_points.copy()
                    no_improve_count = 0
            else:
                # Accept with probability exp(-(delta)/T)
                delta = current_score - new_score
                if delta > 0 and random.random() < np.exp(-delta / max(T, 1e-9)):
                    current_score = new_score
                    no_improve_count += 1
                else:
                    # Revert swap
                    current_points[idx], current_points[swap_idx] = current_points[swap_idx], current_points[idx]
                    no_improve_count += 1
            continue
            
        # Move logic
        old_p_tuple = old_p
        new_p_tuple = new_p
        
        current_points[idx] = new_p
        occupied_set.remove(old_p_tuple)
        occupied_set.add(new_p_tuple)
        
        new_score = _get_score(_vectorized_compute_areas(current_points))
        
        if new_score > current_score:
            current_score = new_score
            no_improve_count = 0
            if new_score > best_score:
                best_score = new_score
                best_points = current_points.copy()
        else:
            delta = current_score - new_score
            if delta > 0 and random.random() < np.exp(-delta / max(T, 1e-9)):
                current_score = new_score
                no_improve_count += 1
            else:
                # Revert
                current_points[idx] = old_p
                occupied_set.remove(new_p_tuple)
                occupied_set.add(old_p_tuple)
                no_improve_count += 1
        
        # Cool down
        T *= cooling_rate
        if T < min_T:
            T = min_T
            
        if no_improve_count > max_no_improve:
            # Restart from best
            current_points = best_points.copy()
            current_score = best_score
            occupied_set = set(map(tuple, current_points))
            no_improve_count = 0
            T = 1.0

    return best_points, best_score

def time_ok_global(margin=0):
    return time.time() < global_deadline - margin

global_deadline = 0

def construct():
    global global_deadline
    start_time = time.time()
    global_deadline = start_time + 85  # 85s deadline with 5s safety margin

    # Generate diverse initial candidates
    best_points = None
    best_score = 0
    
    def try_candidate(points):
        nonlocal best_points, best_score
        if len(points) != 30:
            return
        if not all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in points):
            return
        if len(set(points)) != 30:
            return
            
        score = _get_score(_vectorized_compute_areas(np.array(points)))
        if score > best_score:
            best_score = score
            best_points = points
            # If we found a very good score, we might want to prioritize it
            # But we will run local search on all promising ones later.

    # 1. Grids
    for w in range(1, 31):
        if w * 30 % w != 0: continue
        h = 30 // w
        if h < 1 or h > 30: continue
        for ox in range(-30, 31 - w + 1):
            for oy in range(-30, 31 - h + 1):
                if not time_ok_global(10):
                    break
                points = [(ox + i, oy + j) for i in range(w) for j in range(h)]
                try_candidate(points)
            if not time_ok_global(10):
                break
        if not time_ok_global(10):
            break

    # 2. Sheared grids
    for w in [2, 3, 5, 6, 10, 15, 30]:
        h = 30 // w
        if h < 1: continue
        for shear in range(-5, 6):
            if not time_ok_global(5):
                break
            for ox in range(-30, 31):
                for oy in range(-30, 31):
                    if not time_ok_global(5):
                        break
                    points = []
                    for i in range(w):
                        for j in range(h):
                            x = ox + i + shear * j
                            y = oy + j
                            if -30 <= x <= 30 and -30 <= y <= 30:
                                points.append((x, y))
                    if len(points) == 30:
                        try_candidate(points)
                if not time_ok_global(5):
                    break
            if not time_ok_global(5):
                break

    # 3. Random starts
    for _ in range(20):
        if not time_ok_global(5):
            break
        points = []
        used = set()
        while len(points) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in used:
                points.append((x, y))
                used.add((x, y))
        try_candidate(points)
        
        # Quick local search on this random start if it's decent
        if best_points == points and time_ok_global(20):
            improved_points, improved_score = _local_search(points, time_limit=10.0)
            if improved_score > best_score:
                best_score = improved_score
                best_points = improved_points.tolist()

    # Run local search on the best candidate found so far
    if best_points is not None and time_ok_global(30):
        # Use remaining time for local search
        remaining_time = global_deadline - time.time() - 5
        if remaining_time > 5:
            improved_points, improved_score = _local_search(best_points, time_limit=remaining_time)
            if improved_score > best_score:
                best_score = improved_score
                best_points = improved_points.tolist()

    if best_points is None:
        # Fallback: 6x5 grid
        best_points = [(x, y) for x in range(6) for y in range(5)]

    return best_points
