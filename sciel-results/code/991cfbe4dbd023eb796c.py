"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(centers):
    """Calculate the score (radius) for a given set of centers."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Min pairwise distance
    min_pair_dist = np.min(dists[np.triu_indices(N, k=1)])
    
    # Min distance to walls
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _local_search(centers, time_limit=5.0):
    """
    Structural change: Direct local search maximizing the minimum distance.
    Instead of gradient descent on a potential, we directly perturb spheres
    involved in the current minimum distance constraints.
    """
    start_time = time.time()
    centers = centers.copy()
    current_score = _score(centers)
    best_centers = centers.copy()
    best_score = current_score
    
    iu = np.triu_indices(N, k=1)
    step_size = 0.02
    
    for iteration in range(5000):
        if time.time() - start_time > time_limit:
            break
            
        # Compute min pair and min wall
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        pair_dists_sq = dists_sq[iu]
        min_pair_dist_sq = np.min(pair_dists_sq)
        min_pair_dist = np.sqrt(min_pair_dist_sq)
        
        wall_dists = 0.5 - np.abs(centers)
        min_wall_dist_per_sphere = np.min(wall_dists, axis=1)
        global_min_wall_dist = np.min(min_wall_dist_per_sphere)
        
        current_r = min(min_pair_dist / 2.0, global_min_wall_dist)
        
        # Find critical indices
        min_pair_idx = np.argmin(pair_dists_sq)
        i, j = iu[0][min_pair_idx], iu[1][min_pair_idx]
        min_wall_idx = np.argmin(min_wall_dist_per_sphere)
        
        candidates = [i, j, min_wall_idx]
        # Add random sphere
        if np.random.rand() < 0.1:
            candidates.append(np.random.randint(N))
        
        improved = False
        for idx in candidates:
            # Try to move idx
            for _ in range(10): # 10 random trials per candidate
                direction = np.random.normal(0, 1, 3)
                direction /= np.linalg.norm(direction)
                new_pos = centers[idx] + direction * step_size
                
                if np.any(new_pos < -0.5) or np.any(new_pos > 0.5):
                    continue
                    
                old_pos = centers[idx]
                centers[idx] = new_pos
                
                new_score = _score(centers)
                
                if new_score > current_r:
                    current_r = new_score
                    if current_r > best_score:
                        best_score = current_r
                        best_centers = centers.copy()
                    improved = True
                    break # Accept this move, try next candidate
                else:
                    centers[idx] = old_pos
            
            if improved:
                step_size *= 0.9 # Refine
            else:
                step_size *= 1.05 # Explore
                step_size = min(step_size, 0.1)
                
        # Reset step size occasionally
        if iteration % 50 == 0:
             step_size = 0.02

    return best_centers, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_score = 0.0
    best_centers = None
    
    rng = np.random.default_rng(42)
    
    # Strategy: Mix of Grid Pruning and Random, followed by Local Search
    ticks = np.array([-0.375, -0.125, 0.125, 0.375])
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        if restart < 10:
            # Grid Pruning
            idx = rng.permutation(len(grid))[:N]
            centers = grid[idx].copy()
            centers += rng.normal(0, 0.01, centers.shape)
        else:
            # Random
            centers = rng.uniform(-0.45, 0.45, (N, 3))
            
        centers = np.clip(centers, -0.49, 0.49)
        
        # Run Local Search
        time_left = deadline - time.time()
        ls_time = min(5.0, time_left / (20 - restart))
        
        refined_centers, score = _local_search(centers, time_limit=ls_time)
        
        if score > best_score:
            best_score = score
            best_centers = refined_centers.copy()
            
    # Final Polish on the best candidate
    if best_centers is not None and (deadline - time.time()) > 10:
        final_time = deadline - time.time() - 5
        refined_centers, score = _local_search(best_centers, time_limit=final_time)
        if score > best_score:
            best_score = score
            best_centers = refined_centers.copy()

    if best_centers is None:
        # Fallback
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
# EVOLVE-BLOCK-END
