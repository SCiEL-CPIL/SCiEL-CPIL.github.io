"""Fifty equal spheres in a unit cube: maximize common radius via smooth optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
# Deadline: 150s wall-clock, keep safety margin
DEADLINE = 140.0


def _score(centers):
    """Compute the radius for a set of centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
        return 0.0
    # Check boundaries
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    # Pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    # Distance to walls
    wall_dists = np.minimum(
        np.min(0.5 - np.abs(centers), axis=1),
        0.5
    )
    min_wall = np.min(wall_dists)
    return min(min_dist / 2.0, min_wall)


def _smooth_objective(x, beta=50.0):
    """
    Smooth surrogate: minimize -softmin(pairs, walls) using log-sum-exp.
    x is flattened (N*3,) array of centers.
    """
    c = x.reshape(N, 3)
    
    # Pairwise squared distances (avoid sqrt for speed, use squared)
    # But we need actual distances for the radius... let's use a different approach.
    # Actually, let's compute the negative of a soft-min of all constraints.
    
    # For pairs: d_ij = ||c_i - c_j||
    # For walls: w_ik = 0.5 - |c_ik|
    
    # We want to maximize min(all d_ij / 2, all w_ik)
    # Equivalently minimize -min(...)
    # Use log-sum-exp for soft-min: softmin(a) = -logsumexp(-a/beta) * beta
    
    # Compute all pairwise distances
    diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d_sq = np.sum(diffs * diffs, axis=2)
    # Set diagonal to large value so they don't affect min
    np.fill_diagonal(d_sq, 1e10)
    d = np.sqrt(d_sq)
    # Only upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    pair_dists = d[iu]
    
    # Wall distances
    wall_dists = 0.5 - np.abs(c)  # shape (N, 3)
    wall_flat = wall_dists.flatten()
    
    # All constraints in units of radius
    # Pair constraints contribute d/2, wall constraints contribute w
    all_vals = np.concatenate([pair_dists / 2.0, wall_flat])
    
    # Soft-min via log-sum-exp: softmin = -beta * log(mean(exp(-val/beta)))
    # = -beta * logsumexp(-val/beta) + beta * log(len)
    neg_scaled = -all_vals / beta
    max_neg = np.max(neg_scaled)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_scaled - max_neg)))
    soft_min = -beta * (log_sum - np.log(len(all_vals)))
    
    # We want to maximize soft_min, so minimize -soft_min
    return -soft_min


def _smooth_grad(x, beta=50.0):
    """
    Gradient of the smooth objective.
    """
    c = x.reshape(N, 3)
    
    diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d_sq = np.sum(diffs * diffs, axis=2)
    np.fill_diagonal(d_sq, 1e10)
    d = np.sqrt(d_sq)
    iu = np.triu_indices(N, k=1)
    pair_dists = d[iu]
    
    wall_dists = 0.5 - np.abs(c)
    wall_flat = wall_dists.flatten()
    
    all_vals = np.concatenate([pair_dists / 2.0, wall_flat])
    
    neg_scaled = -all_vals / beta
    max_neg = np.max(neg_scaled)
    exp_vals = np.exp(neg_scaled - max_neg)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # shape (num_constraints,)
    
    # Gradient of -soft_min with respect to each constraint value v_k:
    # d(-soft_min)/dv_k = -beta * (-weights_k) = beta * weights_k
    # But we need d(-soft_min)/dx = sum_k (d(-soft_min)/dv_k) * (dv_k/dx)
    
    # Actually: soft_min = -beta * logsumexp(-v/beta) + beta*log(n)
    # d(soft_min)/dv_k = -beta * (1/beta) * exp(-v_k/beta)/sum(exp(-v/beta)) = -weights_k
    # d(-soft_min)/dv_k = weights_k
    
    # So grad = sum_k weights_k * (dv_k/dx)
    
    grad = np.zeros_like(x)
    
    # Pair contributions
    # v_k = d_ij / 2 for pair (i,j)
    # d(v_k)/d(c_i) = (c_i - c_j) / (2 * d_ij)
    # d(v_k)/d(c_j) = (c_j - c_i) / (2 * d_ij)
    
    num_pairs = len(pair_dists)
    pair_weights = weights[:num_pairs]
    
    for idx in range(num_pairs):
        i, j = iu[0][idx], iu[1][idx]
        diff = c[i] - c[j]
        dist = pair_dists[idx]
        if dist < 1e-10:
            continue
        w = pair_weights[idx]
        # dv/dc_i = diff / (2*dist), dv/dc_j = -diff / (2*dist)
        grad[i * 3:(i+1) * 3] += w * diff / (2.0 * dist)
        grad[j * 3:(j+1) * 3] += w * (-diff) / (2.0 * dist)
    
    # Wall contributions
    # v = 0.5 - |c_ik|
    # d(v)/d(c_ik) = -sign(c_ik)
    wall_weights = weights[num_pairs:]
    wall_grad = -np.sign(c) * wall_weights.reshape(N, 3)
    grad += wall_grad.flatten()
    
    return grad


def _vectorized_smooth_grad(x, beta=50.0):
    """
    Vectorized gradient computation for the smooth objective.
    """
    c = x.reshape(N, 3)
    
    diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d_sq = np.sum(diffs * diffs, axis=2)
    np.fill_diagonal(d_sq, 1e10)
    d = np.sqrt(d_sq)
    iu = np.triu_indices(N, k=1)
    pair_dists = d[iu]
    
    wall_dists = 0.5 - np.abs(c)
    wall_flat = wall_dists.flatten()
    
    all_vals = np.concatenate([pair_dists / 2.0, wall_flat])
    
    neg_scaled = -all_vals / beta
    max_neg = np.max(neg_scaled)
    exp_vals = np.exp(neg_scaled - max_neg)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    grad = np.zeros_like(x)
    
    # Vectorized pair gradient
    num_pairs = len(pair_dists)
    pair_weights = weights[:num_pairs]
    
    # For each pair (i,j), contribution to grad[i] += w * (c_i - c_j)/(2*d)
    # and grad[j] += w * (c_j - c_i)/(2*d)
    
    i_idx = iu[0]
    j_idx = iu[1]
    
    # diff[i,j] = c[i] - c[j], shape (num_pairs, 3)
    pair_diffs = c[i_idx] - c[j_idx]  # (num_pairs, 3)
    pair_dist_vec = pair_dists[:, np.newaxis]  # (num_pairs, 1)
    
    # Weighted contribution
    contrib = pair_weights[:, np.newaxis] * pair_diffs / (2.0 * pair_dist_vec)
    
    # Scatter to grad
    for idx in range(num_pairs):
        grad[i_idx[idx] * 3:(i_idx[idx]+1) * 3] += contrib[idx]
        grad[j_idx[idx] * 3:(j_idx[idx]+1) * 3] -= contrib[idx]
    
    # Wall gradient
    wall_weights = weights[num_pairs:]
    wall_grad = -np.sign(c) * wall_weights.reshape(N, 3)
    grad += wall_grad.flatten()
    
    return grad


def _initial_grid():
    """Create an initial grid placement."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _optimize_from_initial(x0, beta_schedule, max_iter=500, time_limit=None):
    """
    Run L-BFGS-B optimization from initial point x0 with given beta schedule.
    Returns best centers found.
    """
    start_time = time.time()
    best_x = x0.copy()
    best_score = _score(x0.reshape(N, 3))
    
    n_betas = len(beta_schedule)
    
    for bi, beta in enumerate(beta_schedule):
        if time_limit is not None and time.time() > time_limit:
            break
        
        def obj(x):
            return _smooth_objective(x, beta)
        
        def grad(x):
            return _vectorized_smooth_grad(x, beta)
        
        # Bounds: centers in [-0.5, 0.5]
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(
                obj, x0, jac=grad, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success or res.fun < obj(x0):
                x0 = res.x
                s = _score(x0.reshape(N, 3))
                if s > best_score:
                    best_score = s
                    best_x = x0.copy()
        except Exception:
            pass
    
    return best_x, best_score


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with grid
    x0 = _initial_grid().flatten()
    best_x = x0.copy()
    best_score = _score(x0.reshape(N, 3))
    
    # Beta schedule: start coarse, get finer
    # Extended schedule up to beta=1500
    beta_schedule = [10.0, 30.0, 100.0, 300.0, 1000.0, 1500.0]
    
    # First optimization pass
    x_opt, score = _optimize_from_initial(x0, beta_schedule, max_iter=300, time_limit=deadline - 5)
    if score > best_score:
        best_score = score
        best_x = x_opt
    
    # Perturbation restarts: 2 additional
    rng = np.random.default_rng(42)
    for restart in range(2):
        if time.time() > deadline - 10:
            break
        
        # Perturb from best solution
        noise_scale = 0.02 * (restart + 1)
        x_pert = best_x + rng.standard_normal(N * 3) * noise_scale
        # Clip to bounds
        x_pert = np.clip(x_pert, -0.5, 0.5)
        
        # Use a shorter schedule for restarts
        short_schedule = [50.0, 200.0, 500.0, 1000.0]
        x_opt, score = _optimize_from_initial(x_pert, short_schedule, max_iter=200, time_limit=deadline - 5)
        if score > best_score:
            best_score = score
            best_x = x_opt
    
    centers = best_x.reshape(N, 3)
    
    # Final polish: try a few more iterations with high beta on the best solution
    if time.time() < deadline - 3:
        for beta in [2000.0, 3000.0]:
            if time.time() > deadline - 2:
                break
            def obj(x):
                return _smooth_objective(x, beta)
            def grad(x):
                return _vectorized_smooth_grad(x, beta)
            bounds = [(-0.5, 0.5)] * (N * 3)
            try:
                res = minimize(
                    obj, best_x, jac=grad, method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res.fun < obj(best_x):
                    s = _score(res.x.reshape(N, 3))
                    if s > best_score:
                        best_score = s
                        best_x = res.x
            except Exception:
                pass
    
    centers = best_x.reshape(N, 3)
    
    # Ensure validity: clip to cube
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers
