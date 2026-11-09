import numpy as np

N = 50

def construct():
    rng = np.random.default_rng(42)
    
    # 1. Initialization: 4x4x4 Grid subset
    # Create grid
    coords = np.linspace(-0.45, 0.45, 4)
    grid_points = np.array(np.meshgrid(coords, coords, coords)).T.reshape(-1, 3)
    
    # Select 50 points
    # To ensure good distribution, we can pick the 50 points closest to the center?
    # Or just random. Random is fine as long as we have repulsion.
    idx = rng.choice(len(grid_points), N, replace=False)
    centers = grid_points[idx].copy()
    
    # Jitter
    centers += rng.normal(0, 0.02, (N, 3))
    centers = np.clip(centers, -0.5 + 1e-6, 0.5 - 1e-6)
    
    def get_min_dist_and_indices(centers):
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Find min pairwise
        i_idx, j_idx = np.triu_indices(N, k=1)
        min_pair_val = np.min(dists[i_idx, j_idx])
        min_flat_idx = np.argmin(dists[i_idx, j_idx])
        i, j = i_idx[min_flat_idx], j_idx[min_flat_idx]
        
        # Wall distances
        # Distance to each of the 6 walls
        # +x: 0.5 - x, -x: 0.5 + x, etc.
        wall_dists = np.min(np.stack([
            0.5 - centers, # +x, +y, +z
            0.5 + centers  # -x, -y, -z
        ], axis=1), axis=1)
        
        min_wall_val = np.min(wall_dists)
        min_wall_idx = np.argmin(wall_dists)
        
        if min_pair_val < min_wall_val:
            return min_pair_val, i, j, False
        else:
            return min_wall_val, min_wall_idx, -1, True

    best_centers = centers.copy()
    best_score, _, _, _ = get_min_dist_and_indices(centers)
    current_score = best_score
    
    T = 0.02
    T_min = 1e-5
    cooling = 0.9995
    n_iter = 8000
    step_size = 0.01
    
    for _ in range(n_iter):
        if T < T_min:
            break
            
        min_dist, i, j, is_wall = get_min_dist_and_indices(centers)
        
        if not is_wall:
            # Move i and j apart
            vec = centers[j] - centers[i]
            norm = np.linalg.norm(vec)
            if norm < 1e-9:
                vec = rng.normal(0, 1, 3)
                norm = np.linalg.norm(vec)
                vec /= norm
            else:
                vec /= norm
                
            # Move i by -vec * step, j by +vec * step
            # Randomize step direction slightly to escape local minima?
            # Or just deterministic push.
            
            old_i = centers[i].copy()
            old_j = centers[j].copy()
            
            centers[i] = old_i - vec * step_size
            centers[j] = old_j + vec * step_size
            
            # Bounds check
            if np.any(centers[i] < -0.5) or np.any(centers[i] > 0.5) or \
               np.any(centers[j] < -0.5) or np.any(centers[j] > 0.5):
                centers[i] = old_i
                centers[j] = old_j
                continue
                
            new_score, _, _, _ = get_min_dist_and_indices(centers)
            
        else:
            # Move i away from wall
            # Determine direction
            # We know i is the index.
            # We need to know which wall.
            # Recalculate wall dists for i
            d_pos = 0.5 - centers[i]
            d_neg = 0.5 + centers[i]
            
            # Find the min wall dist for i
            # It's the min of d_pos and d_neg across 3 dims
            # Let's just push towards the center of the cube for simplicity
            # This is a safe move
            vec = -centers[i]
            norm = np.linalg.norm(vec)
            if norm < 1e-9:
                vec = rng.normal(0, 1, 3)
                norm = np.linalg.norm(vec)
                vec /= norm
            else:
                vec /= norm
                
            old_i = centers[i].copy()
            centers[i] = old_i + vec * step_size
            
            if np.any(centers[i] < -0.5) or np.any(centers[i] > 0.5):
                centers[i] = old_i
                continue
                
            new_score, _, _, _ = get_min_dist_and_indices(centers)
        
        # SA Acceptance
        delta = new_score - current_score
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_centers = centers.copy()
        else:
            # Revert
            if not is_wall:
                centers[i] = old_i
                centers[j] = old_j
            else:
                centers[i] = old_i
                
        T *= cooling
        
    return best_centers
