"""Pack 200 equal circles in a unit square using Simulated Annealing with incremental updates."""
import numpy as np
import time
import random
from scipy.spatial import KDTree

N = 200
_TIME_LIMIT = 145  # seconds, leave margin before 150s kill

def _compute_pair_indices(n):
    """Generate indices for all unique pairs."""
    i_idx, j_idx = np.triu_indices(n, k=1)
    return i_idx, j_idx

def _generate_initial_points(n_points, n_candidates=2000):
    """
    Generate n_candidates random points, then greedily select n_points 
    that maximize the minimum distance.
    """
    # Generate candidates
    # Mix of random and structured (hex grid)
    rand_pts = np.random.uniform(-0.48, 0.48, size=(n_candidates // 2, 2))
    
    hex_pts = []
    spacing = 0.04
    row_h = spacing * np.sqrt(3) / 2
    y = -0.5
    row_idx = 0
    while y <= 0.5:
        shift = (spacing / 2.0) if row_idx % 2 == 1 else 0.0
        x = -0.5 + shift
        while x <= 0.5:
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                hex_pts.append([x, y])
            x += spacing
        y += row_h
        row_idx += 1
    
    if hex_pts:
        hex_pts = np.array(hex_pts)
        candidates = np.vstack([rand_pts, hex_pts])
    else:
        candidates = rand_pts
        
    # Greedy selection
    selected = []
    available_idx = list(range(len(candidates)))
    
    for _ in range(n_points):
        if not available_idx:
            break
            
        if len(selected) == 0:
            # First point: pick one with max min-wall-dist
            # Or just center?
            # Let's pick the candidate with max min-wall-dist
            wall_dists = np.min([
                candidates[:, 0] + 0.5,
                0.5 - candidates[:, 0],
                candidates[:, 1] + 0.5,
                0.5 - candidates[:, 1]
            ], axis=0)
            best_local_idx = np.argmax(wall_dists)
            best_idx = available_idx[best_local_idx]
        else:
            sel_arr = np.array(selected)
            tree = KDTree(sel_arr)
            
            # Query min distance from each candidate to selected set
            dists, _ = tree.query(candidates, k=1)
            
            # Wall distances
            wall_dists = np.min([
                candidates[:, 0] + 0.5,
                0.5 - candidates[:, 0],
                candidates[:, 1] + 0.5,
                0.5 - candidates[:, 1]
            ], axis=0)
            
            # Effective radius for this candidate
            scores = np.minimum(dists / 2.0, wall_dists)
            
            # Find best score among available
            mask = np.zeros(len(candidates), dtype=bool)
            mask[available_idx] = True
            
            masked_scores = np.where(mask, scores, -1)
            
            best_local_idx = np.argmax(masked_scores)
            if masked_scores[best_local_idx] < 0:
                break
            best_idx = available_idx[best_local_idx]
        
        selected.append(candidates[best_idx])
        available_idx.remove(best_idx)
        
    return np.array(selected)

def _sa_optimize(pts, time_limit):
    """
    Simulated Annealing to maximize min radius.
    Uses incremental updates of pair distances and wall distances.
    """
    n = len(pts)
    pts = pts.copy()
    
    i_idx, j_idx = _compute_pair_indices(n)
    
    # Precompute pair indices for each point for faster update
    pairs_for_point = [[] for _ in range(n)]
    for k in range(len(i_idx)):
        pairs_for_point[i_idx[k]].append(k)
        pairs_for_point[j_idx[k]].append(k)
        
    def _calc_wall_dists(p):
        return np.min([
            p[:, 0] + 0.5,
            0.5 - p[:, 0],
            p[:, 1] + 0.5,
            0.5 - p[:, 1]
        ], axis=0)
    
    wall_dists = _calc_wall_dists(pts)
    pi = pts[i_idx]
    pj = pts[j_idx]
    diffs = pi - pj
    pair_dists = np.linalg.norm(diffs, axis=1)
    
    current_r = min(np.min(pair_dists) / 2.0, np.min(wall_dists))
    
    T_start = 0.001
    T_end = 1e-5
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Cooling based on time
        progress = (time.time() - start_time) / time_limit
        T = T_start * (T_end / T_start) ** progress
        
        i = np.random.randint(n)
        
        # Random move
        # Scale move size with T?
        move_scale = 0.005 + 0.01 * (1 - progress)
        move = np.random.randn(2) * move_scale
        new_pos = pts[i] + move
        
        # Bounds check
        if new_pos[0] < -0.5: new_pos[0] = -0.5
        if new_pos[0] > 0.5: new_pos[0] = 0.5
        if new_pos[1] < -0.5: new_pos[1] = -0.5
        if new_pos[1] > 0.5: new_pos[1] = 0.5
        
        if np.allclose(new_pos, pts[i]):
            continue
            
        # Update distances
        idxs = pairs_for_point[i]
        
        # Store old values
        old_pair_vals = pair_dists[idxs].copy()
        old_wall_val = wall_dists[i]
        
        # New wall dist
        new_wall_i = min(
            new_pos[0] + 0.5,
            0.5 - new_pos[0],
            new_pos[1] + 0.5,
            0.5 - new_pos[1]
        )
        wall_dists[i] = new_wall_i
        
        # New pair dists
        new_pair_vals = np.zeros(len(idxs))
        for m, k in enumerate(idxs):
            if i_idx[k] == i:
                other = j_idx[k]
            else:
                other = i_idx[k]
            diff = new_pos - pts[other]
            new_pair_vals[m] = np.linalg.norm(diff)
            
        pair_dists[idxs] = new_pair_vals
        
        # New R
        new_r = min(np.min(pair_dists) / 2.0, np.min(wall_dists))
        
        # Accept?
        if new_r > current_r or (T > 1e-9 and np.random.rand() < np.exp((new_r - current_r) / T)):
            # Accept
            pts[i] = new_pos
            current_r = new_r
        else:
            # Reject
            pair_dists[idxs] = old_pair_vals
            wall_dists[i] = old_wall_val
            
    return pts

def construct():
    t0 = time.time()
    
    # 1. Initialization
    np.random.seed(42)
    pts = _generate_initial_points(N, n_candidates=3000)
    
    # 2. Optimization
    time_for_sa = _TIME_LIMIT - (time.time() - t0) - 5
    if time_for_sa > 1:
        pts = _sa_optimize(pts, time_for_sa)
        
    return pts

if __name__ == "__main__":
    pts = construct()
    i_idx, j_idx = _compute_pair_indices(N)
    pi = pts[i_idx]
    pj = pts[j_idx]
    pair_dists = np.linalg.norm(pi - pj, axis=1)
    wall_dists = np.min([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1]
    ], axis=0)
    r = min(np.min(pair_dists) / 2.0, np.min(wall_dists))
    print(f"Best Radius: {r}")
