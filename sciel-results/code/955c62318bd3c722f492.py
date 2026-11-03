"""
Packing 100 equal spheres in a unit cube using a discrete lattice + greedy MIS approach.

Strategy:
1. Place 100 centers on a dense 20x20x20 lattice
2. Use greedy maximal independent set to select non-overlapping positions for target radius
3. Local refinement: move each point away from its nearest neighbor
4. Multi-start with different lattice offsets and target radii
5. Deadline guard
"""
import numpy as np
import time
import random

N = 100

def _score(points):
    """Return the maximal radius for the given set of centres."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    margin = 0.5 - np.abs(pts).max(axis=1)
    r_margin = margin.min()
    if r_margin <= 0:
        return 0.0
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    r_pair = min_dist / 2.0
    return min(r_pair, r_margin)

def _lattice_positions(grid_size=20):
    """Generate all lattice positions in [-0.45, 0.45]^3."""
    coords = np.linspace(-0.45, 0.45, grid_size)
    xs, ys, zs = np.meshgrid(coords, coords, coords, indexing='ij')
    positions = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])
    return positions

def _greedy_mis_select(positions, target_r, n_select=100, seed=42):
    """
    Greedy maximal independent set: select n_select positions from lattice
    such that all pairwise distances >= 2*target_r.
    Uses a randomized greedy approach with multiple restarts.
    """
    rng = random.Random(seed)
    best_selection = None
    best_min_dist = 0.0
    
    n_pos = len(positions)
    
    for attempt in range(20):
        # Random order for greedy selection
        order = list(range(n_pos))
        rng.shuffle(order)
        
        selected = []
        selected_indices = []
        
        for idx in order:
            if len(selected) >= n_select:
                break
            
            pos = positions[idx]
            
            # Check if this position is valid (far enough from all selected)
            valid = True
            for sel_pos in selected:
                d = np.linalg.norm(pos - sel_pos)
                if d < 2 * target_r - 1e-9:
                    valid = False
                    break
            
            if valid:
                selected.append(pos)
                selected_indices.append(idx)
        
        if len(selected) >= n_select:
            # Compute actual min distance
            sel_arr = np.array(selected)
            diff = sel_arr[:, np.newaxis, :] - sel_arr[np.newaxis, :, :]
            dists = np.sqrt((diff ** 2).sum(axis=2))
            np.fill_diagonal(dists, np.inf)
            min_d = dists.min()
            
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_selection = sel_arr.copy()
    
    return best_selection

def _local_refine(points, n_iter=50, step_size=0.02):
    """
    Local refinement: for each point, move it away from its nearest neighbor.
    This is a fast combinatorial-style refinement without gradient computation.
    """
    pts = points.copy()
    
    for iteration in range(n_iter):
        improved = False
        for i in range(N):
            # Find nearest neighbor
            dists = np.linalg.norm(pts - pts[i], axis=1)
            np.fill_diagonal(dists, 0)
            j = np.argmin(dists)
            
            if j == i:
                continue
            
            d_ij = pts[i] - pts[j]
            dist_ij = np.linalg.norm(d_ij)
            
            if dist_ij < 1e-12:
                continue
            
            # Move i away from j
            direction = d_ij / dist_ij
            new_pos = pts[i] + direction * step_size
            
            # Keep within bounds
            new_pos = np.clip(new_pos, -0.49, 0.49)
            
            # Check if this improves the minimum distance
            old_min = min(np.linalg.norm(pts[i] - pts[k]) for k in range(N) if k != i)
            new_min = min(np.linalg.norm(new_pos - pts[k]) for k in range(N) if k != i)
            
            if new_min > old_min + 1e-10:
                pts[i] = new_pos
                improved = True
        
        if not improved:
            break
    
    return pts

def _refine_pairwise(points, n_sweeps=10):
    """
    Pairwise refinement: for each pair of closest points, try to separate them.
    """
    pts = points.copy()
    
    for sweep in range(n_sweeps):
        # Find the closest pair
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        min_d = dists[i, j]
        
        d_ij = pts[i] - pts[j]
        dist_ij = np.linalg.norm(d_ij)
        
        if dist_ij < 1e-12:
            continue
        
        direction = d_ij / dist_ij
        
        # Try moving both apart
        move_amount = 0.005
        new_i = pts[i] + direction * move_amount
        new_j = pts[j] - direction * move_amount
        
        # Check bounds
        if np.all(np.abs(new_i) <= 0.49) and np.all(np.abs(new_j) <= 0.49):
            # Check if this improves things
            old_score = _score(pts)
            pts[i] = new_i
            pts[j] = new_j
            new_score = _score(pts)
            
            if new_score <= old_score + 1e-12:
                # Revert
                pts[i] -= direction * move_amount
                pts[j] += direction * move_amount
    
    return pts

def _multi_start_lattice(grid_size=20, target_r=0.11, n_starts=5, seed=42):
    """
    Multi-start: use different lattice offsets and target radii.
    """
    rng = random.Random(seed)
    best_points = None
    best_score = 0.0
    
    for start in range(n_starts):
        # Slightly different target radius
        r_try = target_r + rng.uniform(-0.005, 0.005)
        
        # Get lattice positions with slight random offset
        positions = _lattice_positions(grid_size)
        offset = rng.uniform(-0.02, 0.02)
        positions = positions + offset
        
        # Greedy selection
        selection = _greedy_mis_select(positions, r_try, n_select=N, seed=seed + start)
        
        if selection is not None:
            # Local refinement
            refined = _local_refine(selection, n_iter=30, step_size=0.015)
            refined = _refine_pairwise(refined, n_sweeps=5)
            
            score = _score(refined)
            if score > best_score:
                best_score = score
                best_points = refined.copy()
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_points = None
    best_score = 0.0
    
    # Phase 1: Lattice-based greedy MIS with multiple target radii
    for target_r in [0.105, 0.108, 0.111, 0.113, 0.115]:
        if time.time() > deadline - 20:
            break
        
        points, score = _multi_start_lattice(
            grid_size=20, 
            target_r=target_r, 
            n_starts=8, 
            seed=42 + int(target_r * 1000)
        )
        
        if points is not None and score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Phase 2: Local refinement on best solution
    if best_points is not None and time.time() < deadline - 10:
        refined = _local_refine(best_points, n_iter=100, step_size=0.01)
        refined_score = _score(refined)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined
    
    # Phase 3: Pairwise refinement
    if best_points is not None and time.time() < deadline - 5:
        refined = _refine_pairwise(best_points, n_sweeps=20)
        refined_score = _score(refined)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined
    
    # Final: ensure valid output
    if best_points is None:
        # Fallback: simple grid
        coords = np.linspace(-0.4, 0.4, 10)
        xs, ys, zs = np.meshgrid(coords[:5], coords[:5], coords[:4], indexing='ij')
        best_points = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])
    
    return best_points

# EVOLVE-BLOCK-END
