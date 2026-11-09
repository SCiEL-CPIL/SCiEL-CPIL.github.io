"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(positions):
    """Compute the minimum radius given positions in [-0.5, 0.5]^2."""
    n = len(positions)
    if n == 0:
        return 0.0
    # Distance to boundary
    min_boundary = np.min(np.minimum(
        0.5 - np.abs(positions[:, 0]),
        0.5 - np.abs(positions[:, 1])
    ))
    if min_boundary <= 0:
        return 0.0
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(min_boundary, min_pair / 2.0)

def _soft_min_radius(positions, t=500.0):
    """
    Smooth surrogate for the minimum radius using soft-min (log-sum-exp).
    Returns a smooth approximation of min(min_boundary, min_pair/2).
    """
    n = len(positions)
    # Boundary distances
    boundary_dists = np.minimum(
        0.5 - np.abs(positions[:, 0]),
        0.5 - np.abs(positions[:, 1])
    )
    # Pairwise distances / 2
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    pair_radii = np.sqrt(dist_sq) / 2.0
    
    # Collect all candidate radii
    all_radii = np.concatenate([boundary_dists, pair_radii[np.triu_indices(n, k=1)]])
    
    # Soft min using log-sum-exp trick: -logsumexp(-x/t) * t
    # For numerical stability
    max_val = np.max(all_radii)
    exp_vals = np.exp(-(all_radii - max_val) * t)
    logsumexp_val = np.log(np.sum(exp_vals)) + (max_val - max_val) * t  # simplify
    # Actually: soft_min = -t * log(sum(exp(-r_i / t)))
    # Let's use: -t * logsumexp(-all_radii / t)
    neg_scaled = -all_radii / t
    max_neg = np.max(neg_scaled)
    log_sum = np.log(np.sum(np.exp(neg_scaled - max_neg))) + max_neg
    soft_min_val = -t * log_sum
    
    return soft_min_val

def _generate_hexagonal_grid(n_circles):
    """Generate a hexagonal grid of n_circles points centered in [-0.5, 0.5]^2."""
    points = []
    max_rows = int(np.ceil(np.sqrt(n_circles * 2 / np.sqrt(3)))) + 2
    d = 1.0
    for row in range(-max_rows, max_rows + 1):
        offset = (row % 2) * 0.5 * d
        for col in range(-max_rows, max_rows + 1):
            x = col * d + offset
            y = row * np.sqrt(3) / 2 * d
            points.append([x, y])
    
    points = np.array(points)
    dists = np.sqrt(np.sum(points**2, axis=1))
    idx = np.argsort(dists)[:n_circles]
    points = points[idx]
    
    max_coord = np.max(np.abs(points))
    if max_coord > 0.5:
        points = points * (0.5 / max_coord)
    
    return points

def _generate_square_grid(n_circles):
    """Generate a square grid of n_circles points centered in [-0.5, 0.5]^2."""
    m = int(np.ceil(np.sqrt(n_circles)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n_circles].copy()

def _local_push_optimize(positions, max_iters=1000, tol=1e-10):
    """Iteratively push apart the closest pairs to increase minimum distance."""
    positions = positions.copy()
    n = len(positions)
    
    for it in range(max_iters):
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dist, np.inf)
        
        min_idx = np.argmin(dist)
        i, j = np.unravel_index(min_idx, (n, n))
        if i >= j:
            i, j = j, i
        
        min_dist = dist[i, j]
        if min_dist < tol:
            break
        
        direction = positions[i] - positions[j]
        norm = np.linalg.norm(direction)
        if norm < 1e-15:
            direction = np.array([1.0, 0.0])
            norm = 1.0
        else:
            direction = direction / norm
        
        push = 0.01 * (1.0 - min_dist / 1.0)
        push = max(push, 0.001)
        
        positions[i] += direction * push
        positions[j] -= direction * push
        
        positions = np.clip(positions, -0.5, 0.5)
    
    return positions

def _soft_min_optimize(positions, bounds, max_iters=500, t_start=100.0, t_end=50.0):
    """
    Optimize using soft-min surrogate with annealed temperature.
    Uses L-BFGS-B with smooth objective.
    """
    positions = positions.copy().ravel()
    n = len(positions) // 2
    
    # Anneal temperature
    for iter_phase in range(3):
        t_val = t_start * (t_end / t_start) ** (iter_phase / 2.0)
        
        def _soft_obj(pos_flat):
            pos = pos_flat.reshape(-1, 2)
            # Penalty for out of bounds
            penalty = np.sum(np.maximum(0, pos - 0.5)**2) + np.sum(np.maximum(0, -0.5 - pos)**2)
            if penalty > 1e-6:
                return 1e6 + penalty
            return -_soft_min_radius(pos, t=t_val)
        
        try:
            res = minimize(
                _soft_obj,
                positions,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iters, 'maxfun': max_iters * 10, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            positions = res.x
        except Exception:
            pass
    
    return positions.reshape(-1, 2)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = 0.0
    best_positions = None
    
    # Generate multiple initializations
    hex_pos = _generate_hexagonal_grid(N)
    curr_score = _score(hex_pos)
    if curr_score > best_score:
        best_score = curr_score
        best_positions = hex_pos.copy()
    
    sq_pos = _generate_square_grid(N)
    sq_score = _score(sq_pos)
    if sq_score > best_score:
        best_score = sq_score
        best_positions = sq_pos.copy()
    
    # Define bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Phase 1: Local push from best initial
    if time.time() < deadline - 40:
        try:
            pushed = _local_push_optimize(best_positions, max_iters=500)
            push_score = _score(pushed)
            if push_score > best_score:
                best_score = push_score
                best_positions = pushed.copy()
        except Exception:
            pass
    
    # Phase 2: Soft-min optimization with annealing
    if time.time() < deadline - 30:
        try:
            soft_opt = _soft_min_optimize(best_positions, bounds, max_iters=300, t_start=200.0, t_end=50.0)
            soft_score = _score(soft_opt)
            if soft_score > best_score:
                best_score = soft_score
                best_positions = soft_opt.copy()
        except Exception:
            pass
    
    # Phase 3: Local push refinement
    if time.time() < deadline - 20:
        try:
            pushed2 = _local_push_optimize(best_positions, max_iters=400)
            push_score2 = _score(pushed2)
            if push_score2 > best_score:
                best_score = push_score2
                best_positions = pushed2.copy()
        except Exception:
            pass
    
    # Phase 4: Final soft-min polish with lower temperature
    if time.time() < deadline - 10:
        try:
            soft_polish = _soft_min_optimize(best_positions, bounds, max_iters=200, t_start=100.0, t_end=30.0)
            polish_score = _score(soft_polish)
            if polish_score > best_score:
                best_score = polish_score
                best_positions = soft_polish.copy()
        except Exception:
            pass
    
    # Phase 5: Final local push
    if time.time() < deadline - 5:
        try:
            final_push = _local_push_optimize(best_positions, max_iters=300)
            final_score = _score(final_push)
            if final_score > best_score:
                best_score = final_score
                best_positions = final_push.copy()
        except Exception:
            pass
    
    # Ensure valid output
    if best_positions is None:
        best_positions = _generate_hexagonal_grid(N)
    
    return best_positions

# EVOLVE-BLOCK-END
