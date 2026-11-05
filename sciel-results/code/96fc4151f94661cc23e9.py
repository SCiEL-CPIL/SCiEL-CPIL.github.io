"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import Counter
from itertools import combinations

def _compute_areas_fast(points):
    """Compute all doubled triangle areas (absolute determinants) using vectorized numpy."""
    n = len(points)
    if n < 3:
        return []
    pts = np.array(points, dtype=np.int64)
    x1, y1 = pts[:, 0], pts[:, 1]
    
    # Generate all index triples (i, j, k) with i < j < k
    indices = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(indices, indices, indices, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    # Compute determinants
    x1_f = x1[i_flat]
    y1_f = y1[i_flat]
    x2_f = x1[j_flat]
    y2_f = y1[j_flat]
    x3_f = x1[k_flat]
    y3_f = y1[k_flat]
    
    dets = np.abs(x1_f * (y2_f - y3_f) - x2_f * (y1_f - y3_f) + x3_f * (y1_f - y2_f))
    dets = dets[dets > 0]
    return dets.tolist()

def _score_config(points):
    """Return the maximum multiplicity of any positive doubled area."""
    areas = _compute_areas_fast(points)
    if not areas:
        return 0
    counts = Counter(areas)
    return max(counts.values())

def _score_target_fast(points, target_area):
    """Count how many triples have exactly target_area as doubled area using vectorized numpy."""
    n = len(points)
    if n < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    x1, y1 = pts[:, 0], pts[:, 1]
    
    indices = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(indices, indices, indices, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    x1_f = x1[i_flat]
    y1_f = y1[i_flat]
    x2_f = x1[j_flat]
    y2_f = y1[j_flat]
    x3_f = x1[k_flat]
    y3_f = y1[k_flat]
    
    dets = np.abs(x1_f * (y2_f - y3_f) - x2_f * (y1_f - y3_f) + x3_f * (y1_f - y2_f))
    count = np.sum(dets == target_area)
    return int(count)

def _local_search(points, target_area, max_iter=5000, time_limit=30):
    """Local search to maximize count of triples with target_area."""
    start_time = time.time()
    best_points = list(points)
    best_count = _score_target_fast(best_points, target_area)
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random point to move
        idx = random.randint(0, len(best_points) - 1)
        old_x, old_y = best_points[idx]
        
        # Try random new positions
        new_x = random.randint(-30, 30)
        new_y = random.randint(-30, 30)
        
        # Check distinctness
        is_distinct = True
        for i, (px, py) in enumerate(best_points):
            if i != idx and (px, py) == (new_x, new_y):
                is_distinct = False
                break
        
        if not is_distinct:
            continue
        
        # Try the move
        best_points[idx] = (new_x, new_y)
        new_count = _score_target_fast(best_points, target_area)
        
        if new_count >= best_count:
            best_count = new_count
        else:
            # Revert
            best_points[idx] = (old_x, old_y)
    
    return best_points, best_count

def _initial_grid_points():
    """Generate initial grid of 30 points."""
    points = []
    for x in range(6):
        for y in range(5):
            points.append((x, y))
    return points

def _initial_lattice_points():
    """Generate initial two-lattice configuration of 30 points."""
    # Use a structured lattice that may have more area multiplicities
    points = []
    # First lattice: 5x4 grid
    for x in range(5):
        for y in range(4):
            points.append((x, y))
    # Second lattice: 5x2 grid offset
    for x in range(5):
        for y in range(2):
            points.append((x + 0.5, y + 4.5))
    # Convert to integers and ensure distinctness
    int_points = []
    seen = set()
    for p in points:
        ix, iy = int(round(p[0] * 2)), int(round(p[1] * 2))
        if (ix, iy) not in seen and -60 <= ix <= 60 and -60 <= iy <= 60:
            seen.add((ix, iy))
            int_points.append((ix // 2, iy // 2))
    # If we don't have 30, add more
    while len(int_points) < 30:
        x = random.randint(-30, 30)
        y = random.randint(-30, 30)
        if (x, y) not in seen:
            seen.add((x, y))
            int_points.append((x, y))
    return int_points[:30]

def construct():
    deadline = time.time() + 50  # Safety margin before 90s limit
    
    # Try multiple initial configurations
    initial_configs = [
        _initial_grid_points(),
        _initial_lattice_points()
    ]
    
    best_score = 0
    best_points = None
    
    for init_points in initial_configs:
        if time.time() > deadline:
            break
        
        current_score = _score_config(init_points)
        if current_score > best_score:
            best_score = current_score
            best_points = list(init_points)
    
    if best_points is None:
        best_points = initial_configs[0]
        best_score = _score_config(best_points)
    
    # Determine good target areas from initial config
    areas = _compute_areas_fast(best_points)
    if areas:
        counts = Counter(areas)
        # Sort by frequency, try top candidates
        target_areas = [a for a, c in counts.most_common(15)]
    else:
        target_areas = list(range(1, 20))
    
    # For each target area, do local search
    for target in target_areas:
        if time.time() > deadline:
            break
        
        # Run local search for this target
        remaining_time = max(1, int(deadline - time.time()) - 5)
        improved_points, improved_count = _local_search(
            best_points, target, max_iter=10000, time_limit=remaining_time
        )
        
        if time.time() > deadline:
            break
        
        # Evaluate the actual max multiplicity for this config
        actual_score = _score_config(improved_points)
        if actual_score > best_score:
            best_score = actual_score
            best_points = improved_points
    
    # Final check
    if time.time() > deadline:
        return best_points
    
    return best_points
# EVOLVE-BLOCK-END
