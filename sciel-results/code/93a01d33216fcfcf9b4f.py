"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
CUBE_HALF = 0.5

# EVOLVE-BLOCK-START
import numpy as np
import time

def _radius(centers):
    """Calculate the maximum possible radius for the given centers."""
    # Pairwise distances
    dist_matrix = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair_dist = np.min(dist_matrix)
    
    # Distance to walls
    dist_to_walls = np.min(np.stack([
        CUBE_HALF - np.abs(centers[:, 0]),
        CUBE_HALF - np.abs(centers[:, 1]),
        CUBE_HALF - np.abs(centers[:, 2])
    ], axis=1), axis=1)
    min_wall_dist = np.min(dist_to_walls)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _local_search(centers, max_iters=50000, time_limit=None):
    """
    Local search to maximize the minimum radius.
    """
    centers = centers.copy()
    current_r = _radius(centers)
    step_size = 0.005
    min_step_size = 1e-6
    stagnation = 0
    best_centers = centers.copy()
    best_r = current_r
    
    if time_limit is None:
        time_limit = time.time() + 5
    
    for i in range(max_iters):
        if time.time() > time_limit:
            break
            
        # Calculate gradients
        grad = np.zeros_like(centers)
        
        # 1. Pairwise constraints
        dist_matrix = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dist_matrix, np.inf)
        
        min_dist = np.min(dist_matrix)
        eps = 1e-8
        # Active pairs: those with distance close to the minimum
        active_pairs = dist_matrix < (min_dist + eps)
        rows, cols = np.where(active_pairs)
        
        for r_idx, c_idx in zip(rows, cols):
            if r_idx == c_idx: continue
            vec = centers[r_idx] - centers[c_idx]
            dist = np.linalg.norm(vec)
            if dist > 1e-12:
                grad[r_idx] += vec / dist
                grad[c_idx] -= vec / dist
        
        # 2. Wall constraints
        # Check if wall distance is limiting
        # We need to compare min_wall_dist with min_pair_dist / 2
        # But we already have min_dist (pair).
        # Let's calculate min_wall_dist again or store it?
        # For efficiency, we can just check each particle.
        
        # To be precise, we should check if the wall constraint is active.
        # A wall constraint is active if dist_to_wall < min_pair_dist / 2 + eps
        # OR if it is the global minimum.
        
        # Let's just check for each particle if it's close to a wall
        # and if that wall distance is close to the current limiting radius.
        
        # Current limiting radius is min(min_dist/2, min_wall_dist_global)
        # We don't have min_wall_dist_global here without extra computation.
        # But we can approximate: if a particle is very close to a wall, it's likely active.
        
        # Better: Calculate min_wall_dist_global once?
        # It's O(N). Cheap.
        dist_to_walls = np.min(np.stack([
            CUBE_HALF - np.abs(centers[:, 0]),
            CUBE_HALF - np.abs(centers[:, 1]),
            CUBE_HALF - np.abs(centers[:, 2])
        ], axis=1), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        
        current_limit = min(min_dist / 2.0, min_wall_dist)
        
        for i in range(N):
            if dist_to_walls[i] < current_limit + eps:
                x, y, z = centers[i]
                dx = CUBE_HALF - abs(x)
                dy = CUBE_HALF - abs(y)
                dz = CUBE_HALF - abs(z)
                
                # Find the closest wall axis
                if dx <= dy and dx <= dz:
                    grad[i, 0] += 1.0 if x >= 0 else -1.0
                elif dy <= dx and dy <= dz:
                    grad[i, 1] += 1.0 if y >= 0 else -1.0
                else:
                    grad[i, 2] += 1.0 if z >= 0 else -1.0
        
        # Normalize gradient
        norm = np.linalg.norm(grad)
        if norm > 1e-12:
            grad /= grad_norm if False else norm # Fix syntax
            grad /= norm
            
        # Propose new centers
        new_centers = centers + step_size * grad
        new_centers = np.clip(new_centers, -CUBE_HALF, CUBE_HALF)
        
        # Calculate new radius
        new_r = _radius(new_centers)
        
        if new_r > current_r + 1e-12:
            centers = new_centers
            current_r = new_r
            stagnation = 0
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        else:
            stagnation += 1
            if stagnation > 100:
                # Random perturbation
                centers = centers + np.random.normal(0, step_size, size=centers.shape)
                centers = np.clip(centers, -CUBE_HALF, CUBE_HALF)
                current_r = _radius(centers)
                if current_r < best_r - 1e-3:
                    centers = best_centers.copy()
                    current_r = best_r
                stagnation = 0
                step_size = max(min_step_size, step_size * 0.8)
                
    return best_centers, best_r

def construct():
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    best_centers = None
    best_r = 0.0
    
    # Strategy: Try a few different initializations
    # 1. Jittered Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:N]
    
    np.random.seed(42)
    init_configs = []
    # Jittered grid
    init_configs.append(grid + np.random.normal(0, 0.02, size=grid.shape))
    
    # Random configs
    for _ in range(9):
        rand_c = np.random.uniform(-CUBE_HALF * 0.8, CUBE_HALF * 0.8, size=(N, 3))
        init_configs.append(rand_c)
        
    # Run local search on each
    for idx, config in enumerate(init_configs):
        if time.time() > time_limit:
            break
            
        remaining = time_limit - time.time()
        allocated = remaining / (len(init_configs) - idx)
        
        c, r = _local_search(config, max_iters=50000, time_limit=time.time() + allocated)
        
        if r > best_r:
            best_r = r
            best_centers = c
    
    if best_centers is None:
        return grid
        
    return best_centers
# EVOLVE-BLOCK-END
