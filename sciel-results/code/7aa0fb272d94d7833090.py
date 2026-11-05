"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import Counter
from itertools import combinations

# Precompute all unique triple indices for n=30
_TRIPLES = []
for i in range(30):
    for j in range(i+1, 30):
        for k in range(j+1, 30):
            _TRIPLES.append((i, j, k))

# Precompute which triples involve each point
# _TRIPLES_BY_POINT[p] = list of (i, j, k) triples where p is one of the indices
_TRIPLES_BY_POINT = [[] for _ in range(30)]
for t_idx, (i, j, k) in enumerate(_TRIPLES):
    _TRIPLES_BY_POINT[i].append((i, j, k))
    _TRIPLES_BY_POINT[j].append((i, j, k))
    _TRIPLES_BY_POINT[k].append((i, j, k))

def _compute_areas_and_counts(points):
    """
    Compute all doubled triangle areas and return (max_count, counts_dict, areas_list).
    Uses vectorized numpy for speed.
    """
    n = len(points)
    if n < 3:
        return 0, {}, []
    pts = np.array(points, dtype=np.int64)
    
    # Extract coordinates for all triples
    i_idx = np.array([t[0] for t in _TRIPLES])
    j_idx = np.array([t[1] for t in _TRIPLES])
    k_idx = np.array([t[2] for t in _TRIPLES])
    
    x1_f = pts[i_idx, 0]
    y1_f = pts[i_idx, 1]
    x2_f = pts[j_idx, 0]
    y2_f = pts[j_idx, 1]
    x3_f = pts[k_idx, 0]
    y3_f = pts[k_idx, 1]
    
    dets = np.abs(x1_f * (y2_f - y3_f) - x2_f * (y1_f - y3_f) + x3_f * (y1_f - y2_f))
    
    # Filter out zero areas
    valid_mask = dets > 0
    valid_dets = dets[valid_mask]
    
    if len(valid_dets) == 0:
        return 0, {}, []
        
    counts = Counter(valid_dets.tolist())
    max_count = max(counts.values())
    
    return max_count, counts, valid_dets.tolist()

def _score_move(points, idx, new_pos, counts):
    """
    Quickly score moving point idx to new_pos without modifying counts.
    Returns the new max_count if this move were made.
    Only considers the ~285 triples involving idx.
    """
    n = len(points)
    old_pos = points[idx]
    
    # Get coordinates of all other points (excluding idx)
    other_xs = [points[p][0] for p in range(n) if p != idx]
    other_ys = [points[p][1] for p in range(n) if p != idx]
    # Map from original index to position in other arrays
    other_map = {}
    pos_in_other = 0
    for p in range(n):
        if p != idx:
            other_map[p] = pos_in_other
            pos_in_other += 1
    
    old_x, old_y = old_pos
    new_x, new_y = new_pos
    
    # We need to track how the counts change for areas that involve idx
    # For each triple involving idx, compute old_area and new_area
    # Then simulate: decrement old_area counts, increment new_area counts
    # Track the max count throughout
    
    # To be efficient, we'll use a delta approach:
    # For each area value, track net change in count
    delta = {}
    
    for (i, j, k) in _TRIPLES_BY_POINT[idx]:
        # Compute old area
        xi = old_x if i == idx else points[i][0]
        yi = old_y if i == idx else points[i][1]
        xj = old_x if j == idx else points[j][0]
        yj = old_y if j == idx else points[j][1]
        xk = old_x if k == idx else points[k][0]
        yk = old_y if k == idx else points[k][1]
        
        old_area = abs(xi*(yj-yk) - xj*(yi-yk) + xk*(yi-yj))
        
        # Compute new area
        xi = new_x if i == idx else points[i][0]
        yi = new_y if i == idx else points[i][1]
        xj = new_x if j == idx else points[j][0]
        yj = new_y if j == idx else points[j][1]
        xk = new_x if k == idx else points[k][0]
        yk = new_y if k == idx else points[k][1]
        
        new_area = abs(xi*(yj-yk) - xj*(yi-yk) + xk*(yi-yj))
        
        # Track deltas
        if old_area > 0:
            delta[old_area] = delta.get(old_area, 0) - 1
        if new_area > 0:
            delta[new_area] = delta.get(new_area, 0) + 1
    
    # Now compute the new max count
    # Start from current max, then adjust for affected areas
    # The new max is max over all areas of (counts[area] + delta.get(area, 0))
    # But we only need to check areas that are in counts or in delta
    
    # Quick check: if the current max area's count doesn't decrease below current_max,
    # and no other area exceeds current_max, then new_max = current_max
    # Otherwise, we need to compute more carefully.
    
    # For simplicity and correctness, compute the new max by checking:
    # 1. The current max area (if it's affected)
    # 2. All areas in delta (they might increase)
    # 3. Areas in counts that are not in delta (their count is unchanged)
    
    # Actually, the simplest correct approach:
    # new_max = max(
    #     max((counts[a] + delta.get(a, 0)) for a in set(list(counts.keys()) + list(delta.keys()))),
    #     max(counts.values()) if counts else 0  # for areas not affected
    # )
    # But areas not affected have count = counts[a], so their max is the current max
    # minus any that decreased. This is getting complex.
    
    # Let's just compute it directly but only for affected areas + current max
    # The key insight: areas not in delta have unchanged counts, so their max is 
    # at most current_max (and exactly current_max if the current max area is not affected)
    
    # Find the current max area
    if not counts:
        return 0
    
    # Compute new counts for affected areas
    max_new = 0
    # Check all areas that could potentially be the max
    # These are: areas in delta, and the current max area(s)
    
    # Get current max value
    curr_max_val = max(counts.values())
    
    # Check if any area with count == curr_max_val is affected
    # If not, then curr_max_val is still achievable by unaffected areas
    
    # Simpler: just compute max over the union of keys
    all_keys = set(counts.keys()) | set(delta.keys())
    for a in all_keys:
        new_count = counts.get(a, 0) + delta.get(a, 0)
        if new_count > max_new:
            max_new = new_count
    
    # Also, areas not in all_keys have count = counts[a] which is <= curr_max_val
    # So if no affected area exceeds curr_max_val, the answer is curr_max_val
    # But we already computed max_new over all_keys, which includes all affected areas
    # and the current max area. Areas not in all_keys have count <= curr_max_val.
    # So the true max is max(max_new, curr_max_val) if there's an unaffected area with count curr_max_val
    # But wait, if the current max area IS in all_keys, then its new count is in max_new.
    # If it's NOT in all_keys (impossible since it's in counts), then...
    # Actually, all areas in counts are in all_keys by construction.
    # So max_new is the true new max.
    
    return max_new

def _local_search_best_improvement(points, time_limit=10):
    """
    Local search that tries to move each point to improve the max multiplicity.
    Uses fast incremental scoring via _score_move.
    """
    start_time = time.time()
    n = len(points)
    
    # Initialize counts
    _, counts, _ = _compute_areas_and_counts(points)
    current_max = max(counts.values()) if counts else 0
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        
        # Shuffle point order for diversity
        point_order = list(range(n))
        random.shuffle(point_order)
        
        for idx in point_order:
            if time.time() - start_time > time_limit:
                break
                
            old_pos = points[idx]
            best_new_pos = None
            best_new_score = current_max
            
            # Generate candidate positions
            # Mix of neighborhood and random
            candidates = set()
            
            # Neighborhood: small radius around current position
            ox, oy = old_pos
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    nx, ny = ox + dx, oy + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        candidates.add((nx, ny))
            
            # Random positions
            for _ in range(80):
                candidates.add((random.randint(-30, 30), random.randint(-30, 30)))
            
            # Also try positions that are already in the set (to swap-like moves)
            # Actually, we can't place two points at the same location, so skip occupied
            occupied = set(points)
            
            for new_pos in candidates:
                if new_pos == old_pos:
                    continue
                if new_pos in occupied:
                    continue
                
                new_max = _score_move(points, idx, new_pos, counts)
                
                if new_max > best_new_score:
                    best_new_score = new_max
                    best_new_pos = new_pos
            
            if best_new_pos is not None:
                # Accept the move: update points and counts
                old_pos = points[idx]
                points[idx] = best_new_pos
                
                # Update counts incrementally
                for (i, j, k) in _TRIPLES_BY_POINT[idx]:
                    # Old area
                    xi = old_pos[0] if i == idx else points[i][0]
                    yi = old_pos[1] if i == idx else points[i][1]
                    xj = old_pos[0] if j == idx else points[j][0]
                    yj = old_pos[1] if j == idx else points[j][1]
                    xk = old_pos[0] if k == idx else points[k][0]
                    yk = old_pos[1] if k == idx else points[k][1]
                    old_area = abs(xi*(yj-yk) - xj*(yi-yk) + xk*(yi-yj))
                    if old_area > 0:
                        counts[old_area] -= 1
                        if counts[old_area] == 0:
                            del counts[old_area]
                    
                    # New area
                    xi = best_new_pos[0] if i == idx else points[i][0]
                    yi = best_new_pos[1] if i == idx else points[i][1]
                    xj = best_new_pos[0] if j == idx else points[j][0]
                    yj = best_new_pos[1] if j == idx else points[j][1]
                    xk = best_new_pos[0] if k == idx else points[k][0]
                    yk = best_new_pos[1] if k == idx else points[k][1]
                    new_area = abs(xi*(yj-yk) - xj*(yi-yk) + xk*(yi-yj))
                    if new_area > 0:
                        counts[new_area] = counts.get(new_area, 0) + 1
                
                current_max = best_new_score
                improved = True
                
    return points, current_max

def construct():
    deadline = time.time() + 55  # Safety margin before 90s limit
    
    # Generate initial configurations
    initial_configs = []
    
    # Structured grid: 6x5
    pts = []
    for x in range(6):
        for y in range(5):
            pts.append((x, y))
    initial_configs.append(pts)
    
    # Another structured grid: 5x6
    pts = []
    for x in range(5):
        for y in range(6):
            pts.append((x, y))
    initial_configs.append(pts)
    
    # Random configurations
    for _ in range(8):
        pts = set()
        while len(pts) < 30:
            pts.add((random.randint(-30, 30), random.randint(-30, 30)))
        initial_configs.append(list(pts))
    
    # Small dense grid
    pts = []
    for x in range(-2, 3):
        for y in range(-2, 3):
            pts.append((x, y))
    # That's 25, add 5 more
    for i in range(5):
        pts.append((3, i - 2))
    initial_configs.append(pts)
    
    best_score = 0
    best_points = None
    
    for init_points in initial_configs:
        if time.time() > deadline:
            break
        
        score, _, _ = _compute_areas_and_counts(init_points)
        if score > best_score:
            best_score = score
            best_points = list(init_points)
            
    if best_points is None:
        best_points = initial_configs[0]
        best_score, _, _ = _compute_areas_and_counts(best_points)
        
    # Local search with multiple restarts if time permits
    remaining_time = deadline - time.time()
    
    # Run local search on the best initial config
    if remaining_time > 3:
        improved_points, improved_score = _local_search_best_improvement(
            list(best_points), time_limit=remaining_time - 5)
        
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved_points
    
    # If we still have time, try local search on other initial configs
    remaining_time = deadline - time.time()
    if remaining_time > 3:
        for init_points in initial_configs:
            if time.time() > deadline - 3:
                break
            if init_points == best_points:
                continue
            score, _, _ = _compute_areas_and_counts(init_points)
            # Only run local search if this initial is competitive
            if score >= best_score - 2:
                remaining_time = deadline - time.time()
                if remaining_time > 2:
                    improved_points, improved_score = _local_search_best_improvement(
                        list(init_points), time_limit=remaining_time - 3)
                    if improved_score > best_score:
                        best_score = improved_score
                        best_points = improved_points
    
    return best_points
# EVOLVE-BLOCK-END
