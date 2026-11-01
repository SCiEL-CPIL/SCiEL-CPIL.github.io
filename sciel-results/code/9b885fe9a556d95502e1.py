"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_full_distances(centers):
    """
    Compute the full distance matrix and wall distances.
    Returns (dist_matrix, wall_dists) where dist_matrix[i,j] = |c_i - c_j| for i!=j, inf for i=j.
    """
    diff = centers[:, np.newaxis] - centers[np.newaxis, :]
    dist_matrix = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    
    return dist_matrix, wall_dists


def _sa_run(centers_init, rng, time_limit, T_start=0.03, T_end=1e-10):
    """
    Run simulated annealing starting from centers_init.
    Returns (best_centers, best_score).
    """
    start = time.time()
    max_time = time_limit
    
    centers = centers_init.copy()
    
    # Precompute distance matrix and wall distances
    dist_matrix, wall_dists = _compute_full_distances(centers)
    
    min_pair = np.min(dist_matrix)
    min_wall = np.min(wall_dists)
    current_score = min(min_pair / 2.0, min_wall)
    
    best_centers = centers.copy()
    best_score = current_score
    
    # Adaptive parameters
    step_size = 0.04
    accept_count = 0
    total_count = 0
    no_improve = 0
    
    it = 0
    while time.time() - start < max_time:
        it += 1
        
        # Check time every 5000 iterations
        if it % 5000 == 0:
            if time.time() - start >= max_time:
                break
        
        # Temperature schedule
        progress = min(1.0, it / 500000.0)
        T = T_start * (T_end / T_start) ** progress
        
        # Pick a random center
        i = rng.integers(N)
        old_pos = centers[i].copy()
        
        # Perturb
        centers[i] += rng.normal(0, step_size, 2)
        centers[i] = np.clip(centers[i], -0.4999, 0.4999)
        
        # Compute new distances involving i
        diffs = centers[i] - centers
        new_dists_i = np.sqrt(np.sum(diffs * diffs, axis=1))
        new_dists_i[i] = np.inf
        
        # Update distance matrix row and column for i
        dist_matrix[i, :] = new_dists_i
        dist_matrix[:, i] = new_dists_i
        
        # New wall distance for i
        new_wall_i = 0.5 - np.max(np.abs(centers[i]))
        old_wall_i = wall_dists[i]
        wall_dists[i] = new_wall_i
        
        # Compute new global min_pair and min_wall
        # min_pair = min of all off-diagonal elements
        # Since we only changed row/col i, the new min is:
        #   min(old_min_pair_excluding_i, min(new_dists_i))
        # But we don't know old_min_pair_excluding_i easily.
        # For N=50, just compute the full min - it's fast.
        new_min_pair = np.min(dist_matrix)
        
        # For wall, similarly:
        new_min_wall = np.min(wall_dists)
        
        new_score = min(new_min_pair / 2.0, new_min_wall)
        
        # Acceptance
        if new_score >= current_score:
            current_score = new_score
            accept_count += 1
            total_count += 1
            no_improve = 0
            if new_score > best_score:
                best_score = new_score
                best_centers = centers.copy()
        else:
            delta = new_score - current_score
            total_count += 1
            # Accept with prob exp(delta/T)
            if T > 1e-15 and delta / T > -50:
                if rng.random() < np.exp(delta / T):
                    current_score = new_score
                    accept_count += 1
                    no_improve = 0
                else:
                    # Revert
                    centers[i] = old_pos
                    dist_matrix[i, :] = new_dists_i  # This is wrong, need to revert properly
                    # Actually, let's just revert the center and recompute row/col
                    centers[i] = old_pos
                    diffs = centers[i] - centers
                    new_dists_i = np.sqrt(np.sum(diffs * diffs, axis=1))
                    new_dists_i[i] = np.inf
                    dist_matrix[i, :] = new_dists_i
                    dist_matrix[:, i] = new_dists_i
                    wall_dists[i] = old_wall_i
                    no_improve += 1
            else:
                # Revert
                centers[i] = old_pos
                diffs = centers[i] - centers
                new_dists_i = np.sqrt(np.sum(diffs * diffs, axis=1))
                new_dists_i[i] = np.inf
                dist_matrix[i, :] = new_dists_i
                dist_matrix[:, i] = new_dists_i
                wall_dists[i] = old_wall_i
                no_improve += 1
        
        # Adaptive step size based on acceptance rate
        if total_count % 1000 == 0:
            accept_rate = accept_count / total_count
            if accept_rate < 0.1:
                step_size *= 0.8
            elif accept_rate > 0.8:
                step_size *= 1.2
            step_size = np.clip(step_size, 0.001, 0.1)
            accept_count = 0
            total_count = 0
        
        # If stuck for too long, increase step size to escape
        if no_improve > 5000:
            step_size *= 1.5
            step_size = min(step_size, 0.1)
            no_improve = 0
    
    return best_centers, best_score


def _grid_init(rng):
    """
    Create a grid-based initialization with jitter.
    """
    # 7x8 grid covering most of the square
    rows, cols = 7, 8
    x_ticks = np.linspace(-0.44, 0.44, cols)
    y_ticks = np.linspace(-0.44, 0.44, rows)
    
    centers = np.zeros((N, 2))
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < N:
                centers[idx] = [x_ticks[c], y_ticks[r]]
                idx += 1
            else:
                break
        if idx >= N:
            break
    
    # Add jitter
    centers += rng.normal(0, 0.005, (N, 2))
    centers = np.clip(centers, -0.4999, 0.4999)
    
    return centers


def construct():
    """
    Use simulated annealing with multiple restarts to maximize the packing radius.
    """
    start_time = time.time()
    max_time = 110.0  # Leave 10s margin
    
    rng = np.random.default_rng(42)
    
    best_centers = None
    best_score = -np.inf
    
    # Multiple restarts strategy
    # Use ~30s for first run, then shorter runs with different seeds
    time_allocations = [30.0, 20.0, 15.0, 15.0, 15.0, 10.0, 5.0]
    
    for run_idx, alloc in enumerate(time_allocations):
        if time.time() - start_time >= max_time:
            break
        
        # Different initialization for each restart
        if run_idx == 0:
            # First run: standard grid
            centers_init = _grid_init(rng)
        else:
            # Subsequent runs: more randomized initialization
            if run_idx % 2 == 1:
                # Hexagonal-like pattern with random offset
                centers_init = np.random.default_rng(42 + run_idx).normal(0, 0.15, (N, 2))
                centers_init = np.clip(centers_init, -0.49, 0.49)
            else:
                # Grid with larger jitter
                centers_init = _grid_init(np.random.default_rng(42 + run_idx * 100))
                centers_init += np.random.default_rng(42 + run_idx * 200).normal(0, 0.02, (N, 2))
                centers_init = np.clip(centers_init, -0.4999, 0.4999)
        
        # Run SA
        centers_out, score = _sa_run(centers_init, rng, alloc)
        
        if score > best_score:
            best_score = score
            best_centers = centers_out
    
    # Final verification and cleanup
    if best_centers is None:
        # Fallback: return grid
        best_centers = _grid_init(rng)
    
    # Ensure centers are within bounds
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
# EVOLVE-BLOCK-END
