"""
Program to construct 30 distinct integer coordinate pairs in [-30,30]^2
maximizing the largest multiplicity of a single positive doubled area.
"""

import time
import itertools
import numpy as np
from collections import Counter
import random
import math

def _valid(points):
    """Check 30 distinct integer points in [-30,30]^2."""
    if len(points) != 30:
        return False
    if len(set(points)) != 30:
        return False
    for x, y in points:
        if not (isinstance(x, int) and isinstance(y, int)):
            return False
        if x < -30 or x > 30 or y < -30 or y > 30:
            return False
    return True

def _precompute_indices():
    """Precompute triple indices for all i < j < k with 30 points."""
    i_idx, j_idx, k_idx = np.indices((30, 30, 30))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    return i_idx[mask], j_idx[mask], k_idx[mask]

# Precompute once at module load
_I_IDX, _J_IDX, _K_IDX = _precompute_indices()

def _score(points):
    """
    Compute the maximum multiplicity of a single positive doubled area.
    Uses vectorized NumPy operations for speed.
    """
    if not _valid(points):
        return 0
    
    pts = np.array(points, dtype=np.int64)
    n = len(pts)
    
    # Use precomputed indices
    i_idx = _I_IDX
    j_idx = _J_IDX
    k_idx = _K_IDX
    
    # Coordinates
    xi = pts[i_idx, 0]
    yi = pts[i_idx, 1]
    xj = pts[j_idx, 0]
    yj = pts[j_idx, 1]
    xk = pts[k_idx, 0]
    yk = pts[k_idx, 1]
    
    # Doubled Area = | (xj - xi)(yk - yi) - (yj - yi)(xk - xi) |
    dx1 = xj - xi
    dy1 = yj - yi
    dx2 = xk - xi
    dy2 = yk - yi
    
    areas = np.abs(dx1 * dy2 - dy1 * dx2)
    
    # Filter out zero areas
    positive_areas = areas[areas > 0]
    
    if len(positive_areas) == 0:
        return 0
        
    # Count frequencies
    # np.unique returns sorted unique values and their counts
    unique_areas, counts = np.unique(positive_areas, return_counts=True)
    max_mult = np.max(counts)
    
    return int(max_mult)

def _get_area_counter(points):
    """
    Returns a Counter of all positive doubled areas using precomputed indices.
    """
    if not _valid(points):
        return Counter()
    
    pts = np.array(points, dtype=np.int64)
    
    # Use precomputed indices
    i_idx = _I_IDX
    j_idx = _J_IDX
    k_idx = _K_IDX
    
    xi = pts[i_idx, 0]
    yi = pts[i_idx, 1]
    xj = pts[j_idx, 0]
    yj = pts[j_idx, 1]
    xk = pts[k_idx, 0]
    yk = pts[k_idx, 1]
    
    dx1 = xj - xi
    dy1 = yj - yi
    dx2 = xk - xi
    dy2 = yk - yi
    
    areas = np.abs(dx1 * dy2 - dy1 * dx2)
    positive_areas = areas[areas > 0]
    
    if len(positive_areas) == 0:
        return Counter()
        
    unique_areas, counts = np.unique(positive_areas, return_counts=True)
    return Counter(dict(zip(unique_areas.tolist(), counts.tolist())))

# Alias for backward compatibility
_get_area_counts = _get_area_counter

def _score_target(points, target_area):
    """
    Count the number of triangles with exactly target_area.
    Uses precomputed indices for speed.
    """
    if not _valid(points):
        return 0
    
    pts = np.array(points, dtype=np.int64)
    
    i_idx = _I_IDX
    j_idx = _J_IDX
    k_idx = _K_IDX
    
    xi = pts[i_idx, 0]
    yi = pts[i_idx, 1]
    xj = pts[j_idx, 0]
    yj = pts[j_idx, 1]
    xk = pts[k_idx, 0]
    yk = pts[k_idx, 1]
    
    dx1 = xj - xi
    dy1 = yj - yi
    dx2 = xk - xi
    dy2 = yk - yi
    
    areas = np.abs(dx1 * dy2 - dy1 * dx2)
    
    return int(np.sum(areas == target_area))

def _local_search_target(points, target_area, time_limit=10.0):
    """
    Local search that ONLY accepts moves increasing the count of triangles
    with the specific target_area.
    Includes both single-point moves and two-point swaps for better exploration.
    """
    start_time = time.time()
    best_points = list(points)
    best_count = _score_target(best_points, target_area)
    
    if best_count == 0:
        return best_points
    
    occupied = set(best_points)
    no_improve_count = 0
    max_no_improve = 150  # Allow more exploration
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Decide between single-point move and two-point swap
        use_swap = random.random() < 0.4  # 40% chance of swap
        
        if use_swap:
            # Two-point swap operation
            i = random.randint(0, 29)
            j = random.randint(0, 29)
            if i == j:
                continue
                
            pi, pj = best_points[i], best_points[j]
            
            # Try swapping with nearby points
            for _ in range(40):
                dx1 = random.randint(-5, 5)
                dy1 = random.randint(-5, 5)
                dx2 = random.randint(-5, 5)
                dy2 = random.randint(-5, 5)
                
                ni = (pi[0] + dx1, pi[1] + dy1)
                nj = (pj[0] + dx2, pj[1] + dy2)
                
                # Check bounds
                if ni[0] < -30 or ni[0] > 30 or ni[1] < -30 or ni[1] > 30:
                    continue
                if nj[0] < -30 or nj[0] > 30 or nj[1] < -30 or nj[1] > 30:
                    continue
                
                # Check for collisions
                if ni in occupied or nj in occupied:
                    # Allow if they swap into each other's positions
                    if not (ni == pj and nj == pi):
                        continue
                
                # Apply swap
                best_points[i] = ni
                best_points[j] = nj
                occupied.discard(pi)
                occupied.discard(pj)
                occupied.add(ni)
                occupied.add(nj)
                
                s = _score_target(best_points, target_area)
                
                if s > best_count:
                    best_count = s
                    improved = True
                    no_improve_count = 0
                    break
                else:
                    # Revert
                    best_points[i] = pi
                    best_points[j] = pj
                    occupied.discard(ni)
                    occupied.discard(nj)
                    occupied.add(pi)
                    occupied.add(pj)
            
            if not improved and no_improve_count > max_no_improve:
                # Aggressive random swap
                i = random.randint(0, 29)
                j = random.randint(0, 29)
                if i != j:
                    ni = (random.randint(-30, 30), random.randint(-30, 30))
                    nj = (random.randint(-30, 30), random.randint(-30, 30))
                    
                    if ni not in occupied and nj not in occupied and ni != nj:
                        pi, pj = best_points[i], best_points[j]
                        best_points[i] = ni
                        best_points[j] = nj
                        occupied.discard(pi)
                        occupied.discard(pj)
                        occupied.add(ni)
                        occupied.add(nj)
                        
                        s = _score_target(best_points, target_area)
                        if s > best_count:
                            best_count = s
                            improved = True
                            no_improve_count = 0
                        else:
                            best_points[i] = pi
                            best_points[j] = pj
                            occupied.discard(ni)
                            occupied.discard(nj)
                            occupied.add(pi)
                            occupied.add(pj)
                
                if not improved:
                    break
        else:
            # Single-point move
            random_idx = random.randint(0, 29)
            px, py = best_points[random_idx]
            
            # Try a few random neighbor positions with wider radius
            for _ in range(40):
                # Wider search radius to escape local minima
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
                nx, ny = px + dx, py + dy
                
                if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                    continue
                
                new_point = (nx, ny)
                if new_point in occupied:
                    continue
                
                # Apply move
                old_point = best_points[random_idx]
                best_points[random_idx] = new_point
                occupied.discard(old_point)
                occupied.add(new_point)
                
                s = _score_target(best_points, target_area)
                
                if s > best_count:
                    best_count = s
                    improved = True
                    no_improve_count = 0
                    break  # Accept and restart
                else:
                    # Revert
                    best_points[random_idx] = old_point
                    occupied.discard(new_point)
                    occupied.add(old_point)
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Perturb more aggressively
                    random_idx = random.randint(0, 29)
                    px, py = best_points[random_idx]
                    nx = random.randint(-30, 30)
                    ny = random.randint(-30, 30)
                    new_point = (nx, ny)
                    if new_point not in occupied:
                        old_point = best_points[random_idx]
                        best_points[random_idx] = new_point
                        occupied.discard(old_point)
                        occupied.add(new_point)
                        s = _score_target(best_points, target_area)
                        if s > best_count:
                            best_count = s
                            improved = True
                            no_improve_count = 0
                        else:
                            best_points[random_idx] = old_point
                            occupied.discard(new_point)
                            occupied.add(old_point)
                    
                    if not improved:
                        break  # Local optimum for this target
                        
    return best_points

def _generate_candidate(type_id, params=None):
    """
    Generate structured candidates known to have high symmetry.
    """
    if params is None:
        params = {}
        
    points = []
    
    if type_id == 'regular_polygon_concentric':
        # Concentric regular polygons
        ring_sizes = params.get('ring_sizes', [10, 10, 10])
        radii = params.get('radii', [10, 20, 30])
        
        for r, size in zip(radii, ring_sizes):
            for k in range(size):
                angle = 2 * np.pi * k / size
                x = int(round(r * np.cos(angle)))
                y = int(round(r * np.sin(angle)))
                points.append((x, y))
                
    elif type_id == 'triangular_lattice':
        # Simpler structured grid with shifts
        pts = []
        for r in range(5):
            y = (r - 2) * 4 # -8, -4, 0, 4, 8
            for c in range(6):
                x = (c - 2.5) * 4 # -10, -6, -2, 2, 6, 10
                if r % 2 == 1:
                    x += 2 
                pts.append((int(x), y))
        points = pts

    elif type_id == 'circle_grid':
        # Points on concentric circles
        for r, count in [(10, 10), (20, 10), (30, 10)]:
            for k in range(count):
                angle = 2 * np.pi * k / count
                x = int(round(r * np.cos(angle)))
                y = int(round(r * np.sin(angle)))
                points.append((x, y))
                
    elif type_id == 'square_grid':
        # Standard 5x6 grid
        pts = []
        for x in range(-10, 11, 4):
            for y in range(-8, 9, 4):
                pts.append((x, y))
        points = pts

    return points

def construct():
    """
    Construct 30 points maximizing the multiplicity of a single doubled area.
    Strategy: 
    1. Generate diverse initial candidates.
    2. Identify top candidate areas (targets) from these candidates.
    3. Run specialized local search for each target area with swap operations.
    4. Return the best configuration found.
    """
    start_time = time.time()
    deadline = start_time + 85 # Leave 5s margin
    
    candidates = []
    
    # 1. Generate diverse candidates
    # Regular polygons
    candidates.append(_generate_candidate('regular_polygon_concentric', {'ring_sizes': [15, 15], 'radii': [15, 30]}))
    candidates.append(_generate_candidate('regular_polygon_concentric', {'ring_sizes': [10, 10, 10], 'radii': [10, 20, 30]}))
    candidates.append(_generate_candidate('regular_polygon_concentric', {'ring_sizes': [6, 6, 6, 6, 6], 'radii': [6, 12, 18, 24, 30]}))
    
    # Grids
    candidates.append(_generate_candidate('square_grid'))
    candidates.append(_generate_candidate('triangular_lattice'))
    candidates.append(_generate_candidate('circle_grid'))
    
    # Random
    for _ in range(10):
        pts = []
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in pts:
                pts.append((x, y))
        candidates.append(pts)
        
    # Filter valid candidates
    valid_candidates = [c for c in candidates if _valid(c)]
    
    if not valid_candidates:
        return [(x, y) for x in range(6) for y in range(5)]
        
    # 2. Analyze candidates to find promising target areas
    # We want to find areas that appear frequently in the "best" configurations.
    # Instead of just max score, we look at the top areas from a few good candidates.
    # Let's score all and take the top 5 candidates, then collect their top areas.
    
    scored_candidates = []
    for cand in valid_candidates:
        s = _score(cand)
        if s > 0:
            scored_candidates.append((s, cand))
            
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 5 candidates (or fewer if not enough)
    top_candidates = [c[1] for c in scored_candidates[:5]]
    
    # Collect frequent areas from these top candidates
    area_counts_aggregate = Counter()
    for cand in top_candidates:
        counts = _get_area_counts(cand)
        # Add counts to aggregate
        area_counts_aggregate.update(counts)
        
    # Select top 5 most frequent areas as targets
    if area_counts_aggregate:
        top_areas = [area for area, _ in area_counts_aggregate.most_common(5)]
    else:
        # Fallback to small even numbers
        top_areas = [2, 4, 6, 8, 10]
        
    # 3. Run specialized local search for each target
    best_points = None
    best_global_score = 0
    
    # Allocate time for each target search
    # We have ~80s. Let's give each target search ~12-15s if there are 5 targets.
    # But we should stop if we find a very good score.
    
    for target in top_areas:
        if time.time() > deadline - 5:
            break
            
        # Pick the best initial candidate for this specific target?
        # Or just use the overall best candidate?
        # Using the candidate that already has the most of this target is smart.
        best_init_for_target = None
        max_count_for_target = 0
        for cand in valid_candidates:
            c = _score_target(cand, target)
            if c > max_count_for_target:
                max_count_for_target = c
                best_init_for_target = cand
                
        if best_init_for_target is None:
            best_init_for_target = valid_candidates[0]
            
        # Run local search
        # Time limit: distribute remaining time
        remaining = deadline - time.time()
        # If this is the first target, give it more time?
        # Let's give each target min(15s, remaining/2)
        search_time = min(15.0, remaining / 2.0)
        if search_time < 1.0:
            break
            
        result_points = _local_search_target(best_init_for_target, target, time_limit=search_time)
        
        # Evaluate the result with the GLOBAL score (max multiplicity of ANY area)
        # Because the problem asks for max multiplicity of a single positive doubled area.
        # It's possible that optimizing for Target A leads to a configuration where
        # Area B has an even higher multiplicity, or Target A itself is the max.
        global_s = _score(result_points)
        
        if global_s > best_global_score:
            best_global_score = global_s
            best_points = result_points
            
    # If no search was run or failed, use the best initial candidate
    if best_points is None:
        best_points = valid_candidates[0]
        
    # Final check
    if not _valid(best_points):
        return [(x, y) for x in range(6) for y in range(5)]
        
    return best_points

# EVOLVE-BLOCK-END
