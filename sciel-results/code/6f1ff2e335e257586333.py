"""
Optimized packing of 100 spheres in a unit cube.
Combines FCC lattice initialization with constraint-relaxation (push-apart) refinement.
"""
import numpy as np
import time

N = 100

def _compute_radius(pts):
    """Compute the maximum common radius for given centers."""
    if len(pts) == 0:
        return 0.0
    # Distance to walls
    # pts are in [-0.5, 0.5]
    wall_dists = 0.5 - np.abs(pts)  # shape (n, 3)
    min_wall_per_point = np.min(wall_dists, axis=1)  # shape (n,)
    wall_radius = np.min(min_wall_per_point)
    
    # Pairwise distances
    n = len(pts)
    if n < 2:
        return wall_radius
        
    # Compute all pairwise squared distances
    # Using broadcasting for vectorization
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=2)
    
    # Only consider upper triangle to avoid self-distance (0) and double counting
    iu = np.triu_indices(n, k=1)
    pair_dists = np.sqrt(sq_dists[iu])
    min_pair = np.min(pair_dists)
    pair_radius = min_pair / 2.0
    
    return min(wall_radius, pair_radius)

def _get_min_constraints(pts):
    """
    Find the indices and values of the minimum pair distance and minimum wall distance.
    Returns:
        min_pair_dist: float
        i, j: indices of the closest pair
        min_wall_dist: float
        wi, wk: indices of the point and wall closest to boundary
    """
    n = len(pts)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    min_pair_val = np.min(dists)
    idx = np.unravel_index(np.argmin(dists), dists.shape)
    i, j = idx[0], idx[1]
    
    # Wall distances
    wall_dists = 0.5 - np.abs(pts)
    min_wall_val = np.min(wall_dists)
    wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
    wi, wk = wall_idx[0], wall_idx[1]
    
    return min_pair_val, i, j, min_wall_val, wi, wk

def _push_apart_refine(pts, time_limit=10.0, initial_step=0.05):
    """
    Refine the packing by iteratively pushing the most constrained elements apart.
    This is a form of gradient ascent on the minimum radius.
    """
    start_time = time.time()
    current_pts = pts.copy()
    best_pts = pts.copy()
    best_score = _compute_radius(best_pts)
    
    step = initial_step
    decay = 0.999
    
    while time.time() - start_time < time_limit:
        min_pair_dist, i, j, min_wall_dist, wi, wk = _get_min_constraints(current_pts)
        
        r_pair = min_pair_dist / 2.0
        r_wall = min_wall_dist
        
        # Determine which constraint is limiting
        if r_pair < r_wall:
            # Push pair i, j apart
            v = current_pts[i] - current_pts[j]
            norm_v = np.linalg.norm(v)
            if norm_v > 1e-10:
                direction = v / norm_v
                # Move each point by half the step in opposite directions
                shift = 0.5 * step * direction
                current_pts[i] += shift
                current_pts[j] -= shift
        else:
            # Push point wi away from wall wk
            val = current_pts[wi, wk]
            if val > 0:
                direction = -1.0
            else:
                direction = 1.0
            
            current_pts[wi, wk] += step * direction
        
        # Clamp to cube
        current_pts = np.clip(current_pts, -0.5, 0.5)
        
        # Check score
        score = _compute_radius(current_pts)
        if score > best_score:
            best_score = score
            best_pts = current_pts.copy()
            
        # Decay step size
        step *= decay
        
        # If step becomes too small, break
        if step < 1e-8:
            break
            
    return best_pts, best_score

def _final_polish(pts, time_limit=10.0):
    """
    High-precision coordinate-wise polish.
    """
    start_time = time.time()
    best_pts = pts.copy()
    best_score = _compute_radius(best_pts)
    step = 1e-4
    n = len(pts)
    
    while time.time() - start_time < time_limit and step > 1e-9:
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            for d in range(3):
                # Try +step
                new_pts = best_pts.copy()
                new_pts[i, d] += step
                if new_pts[i, d] <= 0.5 and new_pts[i, d] >= -0.5:
                    new_score = _compute_radius(new_pts)
                    if new_score > best_score + 1e-12:
                        best_pts = new_pts
                        best_score = new_score
                        improved = True
                # Try -step
                new_pts = best_pts.copy()
                new_pts[i, d] -= step
                if new_pts[i, d] <= 0.5 and new_pts[i, d] >= -0.5:
                    new_score = _compute_radius(new_pts)
                    if new_score > best_score + 1e-12:
                        best_pts = new_pts
                        best_score = new_score
                        improved = True
        if not improved:
            step *= 0.5
            
    return best_pts, best_score

def _generate_fcc_candidates():
    """
    Generate FCC lattice points within a slightly larger cube, 
    then select the best 100 based on distance to boundary.
    """
    # FCC lattice vectors
    # Basis vectors for FCC in cubic unit cell of side 'a':
    # (0,0,0), (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)
    # To get a dense set, we can use a smaller 'a' and generate many points, 
    # then scale to fit [-0.5, 0.5].
    
    # Let's generate a grid of FCC points.
    # The density of FCC is higher than simple cubic.
    # We want ~100 points in volume 1.
    # Simple cubic with m=5 gives 125 points.
    # FCC has 4 points per cubic cell of side 'a'.
    # If we use a lattice constant 'a', the number of points in [-0.5, 0.5]^3 is roughly (1/a)^3 * 4.
    # We want ~100. (1/a)^3 * 4 = 100 => 1/a^3 = 25 => a = 1/25^(1/3) ≈ 1/2.92 ≈ 0.34.
    # Let's try a few values of 'a' or just generate a fine grid and filter.
    
    # Alternative: Generate a simple cubic grid of "cells" and add FCC offsets.
    # Let's define the lattice in a larger coordinate system and scale.
    
    # Let's use a grid size that ensures we have enough candidates.
    # m=6 for simple cubic gives 216 points.
    # FCC in a 6x6x6 grid of cells?
    # Let's just generate points on an FCC lattice with spacing s.
    # s = 0.15 gives approx (1/0.15)^3 * 4 = 455 points. Enough.
    
    s = 0.15
    # Generate grid indices
    max_idx = int(0.5 / s) + 2
    indices = np.arange(-max_idx, max_idx + 1)
    
    # FCC basis offsets
    offsets = np.array([
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ]) * s
    
    candidates = []
    for x in indices:
        for y in indices:
            for z in indices:
                base = np.array([x, y, z]) * s
                for off in offsets:
                    pt = base + off
                    # Check if inside cube [-0.5, 0.5]
                    if np.all(pt >= -0.5) and np.all(pt <= 0.5):
                        candidates.append(pt)
    
    candidates = np.array(candidates)
    
    if len(candidates) < N:
        # Fallback to simple cubic if FCC generation failed or too few
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()
        
    # Select N points with largest minimum distance to boundary
    # To do this efficiently, we can just compute min dist to wall for each candidate
    wall_dists = 0.5 - np.abs(candidates)
    min_wall_per_pt = np.min(wall_dists, axis=1)
    
    # Sort by min wall dist descending
    # However, we also want good inter-point spacing.
    # A greedy selection based on wall dist might pick points that are close to each other.
    # Better: Sort by wall dist, pick the first, then for subsequent, pick the one with max min dist to already picked points AND good wall dist.
    # Given the lattice structure, points with similar wall dists are likely well-separated.
    # Let's try sorting by wall dist and taking top N.
    
    sorted_indices = np.argsort(min_wall_per_pt)[::-1]
    selected = candidates[sorted_indices[:N]]
    
    return selected

def construct():
    start_time = time.time()
    total_time_limit = 130.0  # Safety margin below 150s
    
    # 1. Initialization
    initial_pts = _generate_fcc_candidates()
    
    # Add small jitter to break symmetry if needed, though FCC is symmetric.
    # Jitter can help escape local minima caused by lattice symmetry.
    np.random.seed(42)
    # Small jitter relative to lattice spacing
    initial_pts += np.random.uniform(-0.001, 0.001, size=initial_pts.shape)
    initial_pts = np.clip(initial_pts, -0.5, 0.5)
    
    best_overall = initial_pts.copy()
    best_score = _compute_radius(best_overall)
    
    # 2. Push-Apart Refinement
    # This is the main workhorse. It's fast and effective.
    time_left = total_time_limit - (time.time() - start_time) - 10.0
    if time_left > 5.0:
        refined_pts, refined_score = _push_apart_refine(initial_pts, time_limit=min(60.0, time_left), initial_step=0.02)
        if refined_score > best_score:
            best_score = refined_score
            best_overall = refined_pts.copy()
    
    # 3. Final Polish
    time_left = total_time_limit - (time.time() - start_time)
    if time_left > 2.0:
        polished_pts, polished_score = _final_polish(best_overall, time_limit=min(10.0, time_left))
        if polished_score > best_score:
            best_score = polished_score
            best_overall = polished_pts.copy()
            
    # 4. Optional: Second round of push-apart if time remains
    time_left = total_time_limit - (time.time() - start_time)
    if time_left > 10.0:
        refined_pts2, refined_score2 = _push_apart_refine(best_overall, time_limit=min(20.0, time_left), initial_step=0.005)
        if refined_score2 > best_score:
            best_score = refined_score2
            best_overall = refined_pts2.copy()
            
        # Polish again
        time_left = total_time_limit - (time.time() - start_time)
        if time_left > 2.0:
            polished_pts2, polished_score2 = _final_polish(best_overall, time_limit=min(5.0, time_left))
            if polished_score2 > best_score:
                best_score = polished_score2
                best_overall = polished_pts2.copy()
                
    return best_overall

# EVOLVE-BLOCK-END
