"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree
import random

START_TIME = time.time()
TIME_LIMIT = 140.0  # Leave 10s safety margin before 150s kill
SAFETY_MARGIN = 10.0


def _remaining_time():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)


def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    if len(centers) == 0:
        return 0.0
    
    # Distance to walls: min_i min_k (0.5 - |c_ik|)
    # For each point, dist to nearest wall
    dw = np.minimum(np.minimum(0.5 + centers[:, 0], 0.5 - centers[:, 0]),
                    np.minimum(0.5 + centers[:, 1], 0.5 - centers[:, 1]))
    wall_term = np.min(dw)
    
    # Pairwise distances
    n = len(centers)
    if n < 2:
        return float(wall_term)
    
    # Use KDTree for efficiency in finding min distance, but pdist is often faster for full matrix if N is smallish
    # N=200, pdist is 200*199/2 ~ 20k ops, very fast.
    from scipy.spatial.distance import pdist
    d = pdist(centers)
    min_pair = np.min(d)
    pair_term = min_pair / 2.0
    
    return min(float(wall_term), float(pair_term))


def _initial_grid():
    """Create initial grid arrangement with slight jitter to break symmetry."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    # Add small random jitter
    centers += np.random.normal(0, 0.001, centers.shape)
    # Clip to ensure inside
    centers = np.clip(centers, -0.5 + 1e-6, 0.5 - 1e-6)
    return centers


def _optimize_repulsion(centers, max_iters=50000):
    """
    Optimize positions by iteratively pushing apart the closest pairs
    and away from walls.
    """
    centers = centers.copy()
    r = _compute_radius(centers)
    no_improve_count = 0
    best_r = r
    best_centers = centers.copy()
    
    # Adaptive step size
    step_size = 0.01
    
    for it in range(max_iters):
        if _remaining_time() < 2.0:
            break
            
        # 1. Identify bottlenecks
        # Wall constraints
        dw = np.minimum(np.minimum(0.5 + centers[:, 0], 0.5 - centers[:, 0]),
                        np.minimum(0.5 + centers[:, 1], 0.5 - centers[:, 1]))
        min_wall_idx = np.argmin(dw)
        min_wall_dist = dw[min_wall_idx]
        
        # Pair constraints
        from scipy.spatial.distance import pdist
        d = pdist(centers)
        min_pair_idx = np.argmin(d)
        min_pair_dist = d[min_pair_idx]
        
        # Determine which constraint is the bottleneck
        # The radius is min(min_wall_dist, min_pair_dist/2)
        # If wall is bottleneck, push wall point away.
        # If pair is bottleneck, push pair apart.
        
        # We want to increase the minimum.
        # Let's find the actual limiting factor.
        current_r = min(min_wall_dist, min_pair_dist / 2.0)
        
        # If we are limited by a pair
        if min_pair_dist / 2.0 <= min_wall_dist:
            # Find the indices of the closest pair
            # pdist returns condensed form. Index mapping:
            # k = i * N - i * (i + 1) // 2 + j - i - 1 for i < j
            # Inverse:
            # i = floor((2*N*k - k*(k+1)) / (2*N - 1)) ... complex.
            # Easier: use cKDTree to find nearest neighbor for each point, or just loop if N is small?
            # N=200, we can just use a loop to find the pair or use the inverse formula.
            # Let's use a helper to get indices from pdist index.
            i, j = _pdist_index_to_pair(min_pair_idx, N)
            
            # Move i and j apart
            vec = centers[i] - centers[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-9:
                # Random direction
                vec = np.random.randn(2)
                norm = np.linalg.norm(vec)
            direction = vec / norm
            
            # Push apart
            move = step_size * direction
            centers[i] += move
            centers[j] -= move
            
        else:
            # Limited by wall
            idx = min_wall_idx
            # Determine which wall
            x, y = centers[idx]
            # Distances to walls
            dl = x + 0.5
            dr = 0.5 - x
            db = y + 0.5
            dt = 0.5 - y
            
            # Find min
            min_d = min(dl, dr, db, dt)
            if min_d == dl:
                move = np.array([step_size, 0])
            elif min_d == dr:
                move = np.array([-step_size, 0])
            elif min_d == db:
                move = np.array([0, step_size])
            else:
                move = np.array([0, -step_size])
                
            centers[idx] += move
            
        # Clip to square (soft clip, allow slight violation during optimization? No, must stay inside)
        centers = np.clip(centers, -0.5, 0.5)
        
        # Calculate new radius
        new_r = _compute_radius(centers)
        
        if new_r > best_r:
            best_r = new_r
            best_centers = centers.copy()
            no_improve_count = 0
            # Optionally increase step size slightly? Or keep constant.
            # If we made significant progress, maybe keep step size.
        else:
            no_improve_count += 1
            # If stuck, reduce step size or randomize
            if no_improve_count > 100:
                # Random kick
                idx = np.random.randint(N)
                centers[idx] += np.random.randn(2) * 0.01
                centers[idx] = np.clip(centers[idx], -0.5, 0.5)
                no_improve_count = 0
                step_size = max(0.001, step_size * 0.9)
            elif no_improve_count > 50:
                step_size = max(0.001, step_size * 0.95)

    return best_centers, best_r


def _pdist_index_to_pair(k, n):
    """Convert pdist condensed index to (i, j) with i < j."""
    # Formula: k = i * n - i * (i + 1) // 2 + j - i - 1
    # We can solve for i by checking ranges or using binary search.
    # Simpler: iterate i from 0 to n-1 and check if k falls in the block for i.
    # Block i has size n - i - 1.
    # Start of block i: sum_{m=0}^{i-1} (n - m - 1)
    # This is O(N) which is fine for N=200.
    start = 0
    for i in range(n - 1):
        block_size = n - i - 1
        if k < start + block_size:
            j = i + 1 + (k - start)
            return i, j
        start += block_size
    return 0, 1 # Fallback


def construct():
    """Construct optimal circle packing."""
    # Start with grid
    centers = _initial_grid()
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Optimization loop
    if _remaining_time() > 5.0:
        try:
            optimized, opt_radius = _optimize_repulsion(centers)
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = optimized
        except Exception:
            pass
    
    # Ensure all centers are within bounds
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
