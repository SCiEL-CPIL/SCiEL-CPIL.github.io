"""Improved packer: hex-lattice seed + soft-min L-BFGS-B optimization with time guard."""
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 300
SEED = 12345
T_START = time.time()
DEADLINE = 145.0  # safety margin before the 150 s wall-clock limit


def _score(points):
    """Return the feasible common radius (0.0 if invalid)."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # distance to container boundary
    norm = np.linalg.norm(points, axis=1)
    r_bound = 1.0 - norm
    if np.min(r_bound) < 0.0:
        return 0.0
    # minimum pairwise half-distance using KD-tree for efficiency
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    r_pair = 0.5 * np.min(dists[:, 1])
    r = min(r_bound.min(), r_pair)
    return float(r) if r > 0.0 else 0.0


def _min_dist(points):
    """Efficiently compute the minimum pairwise distance."""
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    return float(np.min(dists[:, 1]))


def _lattice_seed(spacing, rng):
    """Generate a hexagonal lattice and pick N points inside the disk."""
    # Add slight random jitter to break symmetry
    while True:
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                # Add small noise
                x += rng.normal(0, spacing * 0.05)
                y += rng.normal(0, spacing * 0.05)
                rows.append((x, y))
        pts = np.array(rows)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        keep = rad <= 1.0 - spacing / 2.0
        pts = pts[keep]
        if len(pts) >= N:
            # prefer inner points (smaller radius) to maximise margin
            order = np.argsort(np.hypot(pts[:, 0], pts[:, 1]), kind="stable")
            return pts[order[:N]].copy()
        spacing *= 0.97


def _soft_min_obj(x, beta):
    """
    Objective function for L-BFGS-B.
    We want to maximize the minimum distance.
    L-BFGS-B minimizes, so we minimize the soft-min of the negative distances?
    No, standard trick: minimize log-sum-exp of negative distances.
    Let d_k be the set of all distances (pairwise and boundary).
    We want to maximize min(d_k).
    This is equivalent to minimizing -min(d_k).
    A smooth approximation of -min(d_k) is (1/beta) * log(sum(exp(-beta * d_k))).
    """
    pts = x.reshape(N, 2)
    
    # Check bounds
    norms = np.linalg.norm(pts, axis=1)
    if np.any(norms > 1.0):
        # Large penalty if outside
        return 1e10 + np.sum(norms[norms > 1.0])

    # Boundary terms: 1 - ||x_i||
    boundary_terms = 1.0 - norms
    
    # Pairwise terms
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Extract upper triangle to avoid double counting and self-distance
    iu = np.triu_indices(N, 1)
    pair_dists = dists[iu]
    
    # Combine all terms
    # We use log-sum-exp for numerical stability
    # log(sum(exp(-beta * d))) = -beta * max(d) + log(sum(exp(-beta * (d - max(d)))))
    
    # To save memory and time, we compute the log-sum-exp directly on the stacked array
    # But creating the full array every time is expensive.
    # Let's compute the two parts separately and combine them in log-space.
    
    # Part 1: Pairs
    max_pair = np.max(pair_dists)
    exp_pairs = np.exp(-beta * (pair_dists - max_pair))
    log_sum_pairs = np.log(np.sum(exp_pairs)) - beta * max_pair
    
    # Part 2: Boundaries
    max_bound = np.max(boundary_terms)
    exp_bounds = np.exp(-beta * (boundary_terms - max_bound))
    log_sum_bounds = np.log(np.sum(exp_bounds)) - beta * max_bound
    
    # Combine: log(exp(log_sum_pairs) + exp(log_sum_bounds))
    # = log_sum_pairs + log(1 + exp(log_sum_bounds - log_sum_pairs))
    # Assuming log_sum_pairs is usually larger (more terms), we shift by it
    # Actually, just take the max of the two log_sums to be safe
    max_log_sum = max(log_sum_pairs, log_sum_bounds)
    combined_log_sum = max_log_sum + np.log(np.exp(log_sum_pairs - max_log_sum) + np.exp(log_sum_bounds - max_log_sum))
    
    # Objective is (1/beta) * combined_log_sum
    return combined_log_sum / beta


def _grad_soft_min_obj(x, beta):
    """
    Gradient of the soft-min objective.
    Obj = (1/beta) * log(sum(exp(-beta * d_k)))
    Grad = (1/beta) * (1 / sum(exp(-beta * d_k))) * sum(exp(-beta * d_k) * (-beta) * grad(d_k))
         = - sum( w_k * grad(d_k) )
    where w_k = exp(-beta * d_k) / sum(exp(-beta * d_m))
    """
    pts = x.reshape(N, 2)
    norms = np.linalg.norm(pts, axis=1)
    
    # Boundary gradients
    # d(1 - ||x_i||)/dx_i = -x_i / ||x_i||
    # Handle zero norm
    safe_norms = np.where(norms < 1e-12, 1.0, norms)
    boundary_grads = -pts / safe_norms[:, np.newaxis]
    boundary_terms = 1.0 - norms
    
    # Pairwise gradients
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero
    dists[np.eye(N, dtype=bool)] = 1.0
    
    iu = np.triu_indices(N, 1)
    pair_dists = dists[iu]
    
    # Calculate weights
    # We need w_k = exp(-beta * d_k) / Z
    # Z = sum(exp(-beta * d_pairs)) + sum(exp(-beta * d_bounds))
    
    # Shift for stability
    max_pair = np.max(pair_dists)
    exp_pairs = np.exp(-beta * (pair_dists - max_pair))
    sum_exp_pairs = np.sum(exp_pairs)
    
    max_bound = np.max(boundary_terms)
    exp_bounds = np.exp(-beta * (boundary_terms - max_bound))
    sum_exp_bounds = np.sum(exp_bounds)
    
    # Combine Z in log space to avoid underflow/overflow if beta is large
    # But we need the normalized weights w_k.
    # Let's compute the unnormalized weights and then normalize.
    # w_pair_ij = exp(-beta * d_ij) / Z
    # w_bound_i = exp(-beta * b_i) / Z
    
    # To avoid underflow, we can compute weights relative to the max term.
    # Let M = max(max_pair, max_bound)
    M = max(max_pair, max_bound)
    
    # Unnormalized weights shifted by M
    u_pairs = np.exp(-beta * (pair_dists - M))
    u_bounds = np.exp(-beta * (boundary_terms - M))
    
    Z_unnorm = np.sum(u_pairs) + np.sum(u_bounds)
    
    if Z_unnorm == 0:
        return np.zeros_like(x)
    
    w_pairs = u_pairs / Z_unnorm
    w_bounds = u_bounds / Z_unnorm
    
    grad = np.zeros_like(pts)
    
    # Boundary contributions
    # grad[i] += w_bounds[i] * (-x_i/||x_i||)
    grad += w_bounds[:, np.newaxis] * boundary_grads
    
    # Pairwise contributions
    # For pair (i, j):
    # grad[i] += w_pair * (x_i - x_j) / d_ij
    # grad[j] += w_pair * (x_j - x_i) / d_ij
    
    diff_pairs = diff[iu] # Shape (M, 2)
    unit_pairs = diff_pairs / pair_dists[:, np.newaxis]
    
    np.add.at(grad, iu[0], w_pairs[:, np.newaxis] * unit_pairs)
    np.add.at(grad, iu[1], -w_pairs[:, np.newaxis] * unit_pairs)
    
    # The objective is (1/beta) * log_sum_exp.
    # The derivative of (1/beta) * log(sum(exp(-beta d))) is 
    # (1/beta) * (1/S) * sum(exp(-beta d_k) * (-beta) * grad(d_k))
    # = - sum( (exp(-beta d_k)/S) * grad(d_k) )
    # = - sum( w_k * grad(d_k) )
    
    # So we need to negate the accumulated gradient
    return -grad.flatten()


def _optimize(points, time_budget):
    """Run L-BFGS-B optimization starting from points with a time budget."""
    start = time.time()
    x0 = points.flatten()
    best_x = x0.copy()
    best_r = _score(points)
    
    # Adaptive beta schedule
    # Start with low beta (smooth) and increase (sharper)
    # Higher beta makes the soft-min closer to the hard min
    betas = [50, 100, 200, 500, 1000, 2000]
    
    for beta in betas:
        if time.time() - start > time_budget:
            break
            
        # Define objective and gradient for this beta
        def obj(x, b=beta):
            return _soft_min_obj(x, b)
            
        def grad(x, b=beta):
            return _grad_soft_min_obj(x, b)
            
        try:
            # Use bounds to keep points roughly in the unit square
            bounds = [(-1.0, 1.0)] * (2 * N)
            res = minimize(obj, x0, jac=grad, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 50, 'maxfun': 50, 'ftol': 1e-12, 'gtol': 1e-8})
            
            new_pts = res.x.reshape(N, 2)
            r = _score(new_pts)
            
            if r > best_r:
                best_r = r
                best_x = res.x.copy()
                x0 = res.x.copy()
            else:
                # If no improvement, break early for this beta?
                # Or continue to next beta?
                # Usually if a step fails to improve, higher beta might struggle more.
                # But sometimes it needs to get over a barrier.
                # Let's stick to the best found so far.
                pass
        except Exception:
            pass
            
    return best_x.reshape(N, 2), best_r


def construct():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # Initial hex-lattice seed
    # Estimate spacing for 300 points in unit circle
    # Area ~ pi * r^2. N * pi * (r/2)^2 ~ pi * 1^2 -> r ~ 2/sqrt(N)
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    
    # Generate initial seed
    pts = _lattice_seed(spacing, rng)
    
    best_pts = pts.copy()
    best_r = _score(best_pts)
    
    # Multi-start optimization
    # We have ~140s budget.
    # Each optimization run takes ~5-15s depending on convergence.
    # We can afford ~10-20 restarts.
    
    num_restarts = 15
    
    for i in range(num_restarts):
        if time.time() - t0 > DEADLINE - 10:
            break
            
        # Perturb the current best or generate a new seed
        if i == 0:
            start_pts = best_pts
        else:
            # Mix of best and random perturbation
            if rng.random() < 0.5:
                start_pts = best_pts + rng.normal(0, 0.05, (N, 2))
            else:
                start_pts = _lattice_seed(spacing, rng)
                
        # Clip to unit circle
        norms = np.linalg.norm(start_pts, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            start_pts[mask] = start_pts[mask] / norms[mask, None] * (1.0 - 1e-6)
            
        # Determine time budget for this restart
        remaining = DEADLINE - 5 - (time.time() - t0)
        if remaining < 5:
            break
            
        # Allocate time: more time for later restarts? Or equal?
        # Let's use a fraction of the remaining time, ensuring at least 5s per run
        time_budget = max(5.0, remaining / (num_restarts - i))
        
        # Optimize
        opt_pts, opt_r = _optimize(start_pts, time_budget)
        
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts
            
        # Check time
        if time.time() - t0 > DEADLINE - 5:
            break
            
    # Final polish with high beta on the best solution found
    if time.time() - t0 < DEADLINE - 5:
        # Use a very high beta for final precision
        # We can't easily call _optimize with a custom beta list, 
        # but we can do a quick manual L-BFGS-B run with high beta
        try:
            x0 = best_pts.flatten()
            bounds = [(-1.0, 1.0)] * (2 * N)
            beta_final = 5000
            res = minimize(lambda x: _soft_min_obj(x, beta_final), x0, 
                           jac=lambda x: _grad_soft_min_obj(x, beta_final), 
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 20, 'maxfun': 20})
            new_pts = res.x.reshape(N, 2)
            r = _score(new_pts)
            if r > best_r:
                best_pts = new_pts
                best_r = r
        except Exception:
            pass

    # Final check
    final_r = _score(best_pts)
    if final_r <= 0.0:
        # Fallback
        best_pts = _lattice_seed(spacing, rng)
        
    return best_pts
