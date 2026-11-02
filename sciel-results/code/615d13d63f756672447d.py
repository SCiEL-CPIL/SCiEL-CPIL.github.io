"""Optimized 50-sphere packing in unit cube via simulated annealing + polish."""
import numpy as np
import time

N = 50
DEADLINE = 115.0  # safety margin before 120s kill

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Wall distances
    wall_dists = np.minimum(0.5 - np.abs(centers), axis=1)
    min_pair = np.min(dists[np.triu_indices(N, k=1)]) / 2.0
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)

def _pairwise_dists(centers):
    """Return upper-triangular pairwise distances and wall distances."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    wall_dists = np.minimum(0.5 - np.abs(centers), axis=1)
    return dists, wall_dists

def _anneal(centers, start_temp, end_temp, steps, deadline):
    """Simulated annealing to maximize minimum clearance."""
    centers = centers.copy()
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    # Precompute index pairs for efficiency
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    for step in range(steps):
        if time.time() > deadline:
            break
        
        # Temperature schedule (geometric cooling)
        t = start_temp * (end_temp / start_temp) ** (step / max(steps - 1, 1))
        
        # Generate random perturbation
        # Move a random subset of points
        n_move = max(1, int(N * 0.1))
        move_idx = np.random.choice(N, size=n_move, replace=False)
        old_positions = centers[move_idx].copy()
        
        # Random displacement
        scale = t * 0.5
        centers[move_idx] += np.random.randn(n_move, 3) * scale
        
        # Clamp to cube
        centers[move_idx] = np.clip(centers[move_idx], -0.49, 0.49)
        
        # Compute new radius
        new_radius = _compute_radius(centers)
        
        # Acceptance criterion
        if new_radius > best_radius or np.random.random() < np.exp((new_radius - best_radius) / max(t, 1e-10)):
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = centers.copy()
        else:
            centers[move_idx] = old_positions
    
    return best_centers, best_radius

def _polish(centers, deadline):
    """Local refinement using gradient-like ascent on minimum clearance."""
    centers = centers.copy()
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    # Find the bottleneck pair and wall
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        dists, wall_dists = _pairwise_dists(centers)
        pair_dists = dists[i_idx, j_idx]
        min_pair_idx = np.argmin(pair_dists)
        min_pair_val = pair_dists[min_pair_idx] / 2.0
        min_wall_idx = np.argmin(wall_dists)
        min_wall_val = wall_dists[min_wall_idx]
        
        current_radius = min(min_pair_val, min_wall_val)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = centers.copy()
        
        # Determine which constraint is active
        if min_pair_val < min_wall_val:
            # Push apart the closest pair
            i, j = i_idx[min_pair_idx], j_idx[min_pair_idx]
            diff = centers[i] - centers[j]
            norm = np.linalg.norm(diff)
            if norm < 1e-12:
                continue
            diff /= norm
            step_size = 0.001
            centers[i] += diff * step_size
            centers[j] -= diff * step_size
        else:
            # Push away from wall
            k = min_wall_idx
            coord = np.argmin(np.abs(centers[k]))
            # Determine which side of the wall
            if centers[k, coord] > 0:
                centers[k, coord] -= 0.001
            else:
                centers[k, coord] += 0.001
        
        # Clamp to cube
        centers = np.clip(centers, -0.49, 0.49)
    
    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Initial guess: slightly perturbed grid
    m = 4  # 4x4x4 = 64 positions, pick 50
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Pick 50 from the 64 grid points with some randomness
    np.random.seed(42)
    perm = np.random.permutation(len(grid))
    centers = grid[perm[:N]].copy()
    
    # Add small random perturbation to break symmetry
    centers += np.random.randn(N, 3) * 0.01
    centers = np.clip(centers, -0.49, 0.49)
    
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    # Phase 1: Coarse annealing
    remaining = deadline - time.time()
    if remaining > 5:
        steps1 = int(min(5000, remaining * 100))
        best_centers, best_radius = _anneal(
            best_centers, start_temp=0.05, end_temp=0.001,
            steps=steps1, deadline=deadline - 10
        )
    
    # Phase 2: Fine annealing
    remaining = deadline - time.time()
    if remaining > 5:
        steps2 = int(min(5000, remaining * 100))
        best_centers, best_radius = _anneal(
            best_centers, start_temp=0.01, end_temp=0.0001,
            steps=steps2, deadline=deadline - 5
        )
    
    # Phase 3: Polish
    remaining = deadline - time.time()
    if remaining > 2:
        best_centers, best_radius = _polish(
            best_centers, deadline=deadline - 1
        )
    
    # Final validation and clamping
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
