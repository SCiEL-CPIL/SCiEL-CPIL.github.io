"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

def _compute_radius(centers):
    """Compute the packing radius for a set of centers."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    min_pair_dist = np.sqrt(np.min(dists_sq[iu]))
    
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _smooth_objective(x, t=0.02):
    """
    Objective function to be minimized.
    We want to maximize the minimum clearance.
    Clearance is defined as min(min_pair_dist/2, min_wall_dist).
    We use a soft-min approximation: -t * log(sum(exp(-c/t)))
    Minimizing this is equivalent to maximizing the soft-min of clearances.
    """
    centers = x.reshape(N, 3)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    pair_dists = np.sqrt(np.maximum(dists_sq[iu], 1e-12))
    pair_clearances = pair_dists / 2.0
    
    # Wall clearances
    wall_dists = 0.5 - np.abs(centers)
    wall_clearances = np.min(wall_dists, axis=1)
    
    # Combine all clearances
    all_clearances = np.concatenate([pair_clearances, wall_clearances])
    
    # Soft-min: -t * log(sum(exp(-c/t)))
    # To avoid overflow, subtract max
    max_c = np.max(all_clearances)
    log_sum_exp = np.log(np.sum(np.exp(-(all_clearances - max_c) / t)))
    soft_min = -(max_c + t * log_sum_exp)
    
    return -soft_min # Minimize negative soft-min => Maximize soft-min

def _smooth_gradient(x, t=0.02):
    """
    Gradient of the smooth objective.
    """
    centers = x.reshape(N, 3)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    pair_dists = np.sqrt(np.maximum(dists_sq[iu], 1e-12))
    pair_clearances = pair_dists / 2.0
    
    wall_dists = 0.5 - np.abs(centers)
    wall_clearances = np.min(wall_dists, axis=1)
    
    all_clearances = np.concatenate([pair_clearances, wall_clearances])
    
    # Weights for soft-min
    max_c = np.max(all_clearances)
    weights = np.exp(-(all_clearances - max_c) / t)
    weights /= np.sum(weights)
    
    grad = np.zeros_like(x)
    
    # Pair contributions
    # d/dc_i (|x_i - x_j|/2) = (x_i - x_j) / (2 * |x_i - x_j|)
    # d/dx_i (-t * log(sum exp(-c/t))) = sum_k w_k * (t/t) * d(-c_k)/dx_i = sum_k w_k * d(-c_k)/dx_i
    # Wait, objective is -soft_min = t * log(sum exp(-c/t)).
    # d/dx (t * log S) = t * (1/S) * sum exp(-c_k/t) * (-1/t) * dc_k/dx = -1/S * sum exp(-c_k/t) * dc_k/dx = - sum w_k * dc_k/dx
    # So gradient of objective is - sum w_k * dc_k/dx.
    # We want to minimize objective, so we move in direction -grad_obj = sum w_k * dc_k/dx.
    # Let's just compute grad_obj = - sum w_k * dc_k/dx.
    
    n_pairs = len(iu[0])
    pair_weights = weights[:n_pairs]
    pair_diffs = diff[iu[0], iu[1], :]
    
    for k in range(n_pairs):
        w = pair_weights[k]
        d = pair_diffs[k]
        dist = pair_dists[k]
        if dist < 1e-8:
            continue
        # dc_k/dx_i = d / (2 * dist)
        # dc_k/dx_j = -d / (2 * dist)
        # grad_obj += -w * (d / (2*dist)) for x_i
        # grad_obj += -w * (-d / (2*dist)) for x_j
        grad[iu[0][k] * 3 : iu[0][k] * 3 + 3] -= w * d / (2 * dist)
        grad[iu[1][k] * 3 : iu[1][k] * 3 + 3] += w * d / (2 * dist)
    
    # Wall contributions
    wall_weights = weights[n_pairs:]
    for i in range(N):
        w = wall_weights[i]
        # c_i = 0.5 - min_k |x_ik|
        # dc_i/dx_ik = -sign(x_ik) for the min k
        # To be smooth, we can approximate or just use the min index.
        # Using the min index is non-smooth but okay for L-BFGS-B if we handle it.
        # Better: use a soft-min for wall clearances too?
        # For now, use the min index.
        min_idx = np.argmin(0.5 - np.abs(centers[i]))
        # dc_i/dx_i[min_idx] = -sign(centers[i][min_idx])
        # grad_obj += -w * (-sign(x)) = w * sign(x)
        grad[i * 3 : i * 3 + 3] += w * np.sign(centers[i])
        # Note: this is only for the min coordinate. Other coordinates have 0 gradient.
        # This is a subgradient.
        
    return grad

def _optimize_lbfgsb(centers, t=0.02, max_iter=500):
    """
    Run L-BFGS-B optimization on the smooth objective.
    """
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    res = minimize(
        _smooth_objective,
        x0,
        jac=_smooth_gradient,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    return res.x.reshape(N, 3)

def _seed_grid(rng):
    """Generate a grid-based seed with perturbation."""
    # 4x4x4 grid = 64 points. We need 50.
    ticks = np.linspace(-0.4, 0.4, 4)
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = rng.permutation(len(grid))[:N]
    centers = grid[idx].copy()
    centers += rng.normal(0, 0.01, centers.shape)
    return np.clip(centers, -0.49, 0.49)

def _seed_fcc_like(rng):
    """Generate an FCC-like lattice seed."""
    points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                points.append([i, j, k])
                if i < 2:
                    points.append([i + 0.5, j, k])
                    points.append([i + 0.5, j + 0.5, k])
                    points.append([i + 0.5, j, k + 0.5])
                if j < 2:
                    points.append([i, j + 0.5, k])
                    points.append([i, j + 0.5, k + 0.5])
                if k < 2:
                    points.append([i, j, k + 0.5])
                    points.append([i + 0.5, j, k + 0.5])
    
    points = np.array(points)
    points = points / 2.5 - 0.5
    mask = np.all(points >= -0.49, axis=1) & np.all(points <= 0.49, axis=1)
    points = points[mask]
    
    if len(points) >= N:
        selected = [rng.integers(len(points))]
        for _ in range(N - 1):
            dists = np.min(np.linalg.norm(points[:, np.newaxis, :] - points[selected][np.newaxis, :, :], axis=2), axis=1)
            selected.append(np.argmax(dists))
        return points[selected]
    else:
        extra = rng.uniform(-0.45, 0.45, (N - len(points), 3))
        return np.vstack([points, extra])

def _seed_random(rng):
    """Generate a random seed."""
    return rng.uniform(-0.45, 0.45, (N, 3))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_score = 0.0
    best_centers = None
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts
    n_restarts = 10
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # Choose seed
        if restart == 0:
            centers = _seed_fcc_like(rng)
        elif restart < 5:
            centers = _seed_grid(rng)
        else:
            centers = _seed_random(rng)
        
        centers = np.clip(centers, -0.49, 0.49)
        
        # Optimize with L-BFGS-B
        # Use a small t for sharper minimum approximation
        t = 0.01
        try:
            optimized_centers = _optimize_lbfgsb(centers, t=t, max_iter=200)
            score = _compute_radius(optimized_centers)
            
            if score > best_score:
                best_score = score
                best_centers = optimized_centers.copy()
        except Exception as e:
            pass
        
        # If time permits, do a second pass with smaller t
        if time.time() < deadline - 10 and best_centers is not None:
            t2 = 0.005
            try:
                refined_centers = _optimize_lbfgsb(best_centers, t=t2, max_iter=100)
                score2 = _compute_radius(refined_centers)
                if score2 > best_score:
                    best_score = score2
                    best_centers = refined_centers.copy()
            except Exception as e:
                pass
    
    if best_centers is None:
        # Fallback
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
