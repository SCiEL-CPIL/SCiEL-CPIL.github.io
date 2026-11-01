"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations


def _full_score(centers):
    """Compute the packing radius: min(min_pair_dist/2, min_wall_dist)."""
    min_wall = np.min(0.5 - np.abs(centers))
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair = np.sqrt(np.min(dists_sq))
    return min(min_pair / 2.0, min_wall)


def _sa_run(centers, rng, time_budget):
    """
    Run simulated annealing with efficient incremental scoring.
    """
    start_time = time.time()
    centers = centers.copy()
    
    # Precompute initial state
    wall_dists = 0.5 - np.abs(centers)
    current_score = _full_score(centers)
    
    best_centers = centers.copy()
    best_score = current_score
    
    # SA parameters
    n_iters = 1000000
    T_start = 0.02
    T_end = 1e-10
    alpha = (T_end / T_start) ** (1.0 / n_iters)
    
    # Adaptive step size
    step = 0.015
    base_step = 0.015
    no_improve = 0
    max_no_improve = 1500
    
    # Track distances efficiently
    # For each circle, store distances to all others
    # We'll recompute the full distance matrix periodically
    
    for it in range(n_iters):
        # Time check every 200 iterations
        if it % 200 == 0:
            if time.time() - start_time > time_budget:
                break
        
        T = T_start * (alpha ** it)
        
        # Pick random circle to move
        i = rng.integers(N)
        old_pos = centers[i].copy()
        
        # Generate new position with adaptive step
        new_pos = old_pos + rng.normal(0, step, 2)
        new_pos = np.clip(new_pos, -0.4999, 0.4999)
        
        # Compute new distances from i to all others
        new_dists_sq_i = np.sum((centers - new_pos) ** 2, axis=1)
        new_dists_sq_i[i] = np.inf
        new_min_pair_inv_i = np.min(new_dists_sq_i)
        
        # New wall distance for circle i
        new_wall_i = np.min(0.5 - np.abs(new_pos))
        
        # Quick upper bound on new score
        # The new score is at most min(new_min_pair_inv_i/2, new_wall_i)
        # but could be limited by pairs not involving i
        candidate_upper = min(np.sqrt(new_min_pair_inv_i) / 2.0, new_wall_i)
        
        # If upper bound is much worse, reject quickly
        if candidate_upper < current_score * 0.95:
            no_improve += 1
            continue
        
        # Apply the move temporarily
        centers[i] = new_pos
        
        # Compute actual new score (full recompute for correctness)
        new_score = _full_score(centers)
        
        if new_score >= current_score:
            # Accept improving move
            current_score = new_score
            no_improve = 0
            if new_score > best_score:
                best_score = new_score
                best_centers = centers.copy()
        else:
            # Metropolis acceptance for worse moves
            delta = new_score - current_score
            if T > 1e-12 and rng.random() < np.exp(delta / T):
                current_score = new_score
                no_improve += 1
            else:
                # Revert
                centers[i] = old_pos
                no_improve += 1
        
        # Adaptive step size
        if no_improve > max_no_improve:
            step *= 0.7
            no_improve = 0
        elif it % 500 == 0 and it > 0:
            step *= 0.999
        
        # Periodic reheating to escape local minima
        if it % 30000 == 0 and it > 0:
            step = base_step * (0.8 + 0.4 * rng.random())
            no_improve = 0
    
    # Final local refinement using coordinate descent
    if time.time() - start_time < time_budget:
        centers = best_centers.copy()
        refine_budget = min(20.0, time_budget - (time.time() - start_time))
        refine_start = time.time()
        cur_score = _full_score(centers)
        
        # Multi-scale coordinate descent
        scales = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
        for scale in scales:
            if time.time() - refine_start > refine_budget:
                break
            
            for _ in range(200):
                if time.time() - refine_start > refine_budget:
                    break
                
                improved = False
                for i in range(N):
                    old_pos = centers[i].copy()
                    
                    # Try moving in each direction
                    for dx in [-scale, 0, scale]:
                        for dy in [-scale, 0, scale]:
                            if dx == 0 and dy == 0:
                                continue
                            new_pos = old_pos + [dx, dy]
                            new_pos = np.clip(new_pos, -0.4999, 0.4999)
                            centers[i] = new_pos
                            new_score = _full_score(centers)
                            
                            if new_score > cur_score:
                                cur_score = new_score
                                improved = True
                                if cur_score > best_score:
                                    best_score = cur_score
                                    best_centers = centers.copy()
                            else:
                                centers[i] = old_pos
                    
                    # Also try random perturbations at this scale
                    new_pos = old_pos + rng.normal(0, scale * 0.5, 2)
                    new_pos = np.clip(new_pos, -0.4999, 0.4999)
                    centers[i] = new_pos
                    new_score = _full_score(centers)
                    if new_score > cur_score:
                        cur_score = new_score
                        improved = True
                        if cur_score > best_score:
                            best_score = cur_score
                            best_centers = centers.copy()
                    else:
                        centers[i] = old_pos
                
                if not improved:
                    break
    
    return best_centers, best_score


def _make_grid_init(rows, cols, jitter=0.003, rng=None):
    """Create a grid initialization."""
    centers = np.zeros((N, 2))
    x_ticks = np.linspace(-0.48, 0.48, cols)
    y_ticks = np.linspace(-0.48, 0.48, rows)
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
    if rng is not None:
        centers += rng.normal(0, jitter, (N, 2))
    return np.clip(centers, -0.4999, 0.4999)


def _make_hex_init(spacing=0.19, jitter=0.003, rng=None):
    """Create a hexagonal packing initialization."""
    centers = np.zeros((N, 2))
    hex_row_h = spacing * np.sqrt(3) / 2
    idx = 0
    row = 0
    while idx < N and row < 12:
        offset = 0 if row % 2 == 0 else spacing / 2
        col = 0
        while idx < N and col < 10:
            x = -0.48 + offset + col * spacing
            y = -0.48 + row * hex_row_h
            if -0.4999 <= x <= 0.4999 and -0.4999 <= y <= 0.4999:
                centers[idx] = [x, y]
                idx += 1
            col += 1
        row += 1
    if idx == N and rng is not None:
        centers += rng.normal(0, jitter, (N, 2))
    return np.clip(centers, -0.4999, 0.4999)


def construct():
    """
    Multi-start simulated annealing to maximize the packing radius.
    """
    rng = np.random.default_rng(42)
    
    start_time = time.time()
    max_time = 110.0
    
    best_centers = None
    best_score = -np.inf
    
    # Generate diverse initializations
    initializations = []
    
    # Grid-based initializations
    grid_configs = [(7, 8), (8, 7), (5, 10), (10, 5), (6, 9), (9, 6)]
    for rows, cols in grid_configs:
        initializations.append(_make_grid_init(rows, cols, 0.003, rng))
    
    # Hexagonal packing
    initializations.append(_make_hex_init(0.19, 0.003, rng))
    initializations.append(_make_hex_init(0.18, 0.003, rng))
    
    # Random initialization
    initializations.append(rng.uniform(-0.45, 0.45, (N, 2)))
    
    # Run SA from each initialization
    n_inits = len(initializations)
    for init_idx, init_centers in enumerate(initializations):
        remaining = max_time - (time.time() - start_time)
        if remaining < 8.0:
            break
        
        # Allocate time budget
        time_budget = min(remaining, max(10.0, remaining / (n_inits - init_idx)))
        
        centers_local, score_local = _sa_run(init_centers, rng, time_budget)
        
        if score_local > best_score:
            best_score = score_local
            best_centers = centers_local
    
    # Final verification
    if best_centers is None:
        best_centers = _make_grid_init(7, 8, 0.0, None)
    
    return best_centers
# EVOLVE-BLOCK-END
