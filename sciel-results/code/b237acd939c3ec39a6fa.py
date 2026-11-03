"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_init(n, seed=0):
    """Generate n points from a hexagonal lattice inside the unit disk."""
    rng = np.random.RandomState(seed)
    # Estimate spacing for n points in unit disk
    # Area per point ~ pi/n, hex cell area ~ s^2 * sqrt(3)/2
    s = np.sqrt(2 * np.pi / (n * np.sqrt(3)))
    
    for attempt in range(30):
        # Try slightly varying spacings
        spacing = s * (0.95 + rng.random() * 0.05)
        k = int(2.0 / spacing) + 5
        xs, ys = np.meshgrid(
            np.arange(-k, k + 1) * spacing,
            np.arange(-k, k + 1) * spacing * np.sqrt(3.0) / 2.0
        )
        # Offset odd rows
        for j in range(2 * k + 1):
            if j % 2 == 1:
                xs[j, :] += spacing / 2.0
        pts = np.column_stack([xs.ravel(), ys.ravel()])
        r = np.hypot(pts[:, 0], pts[:, 1])
        mask = r <= 1.0 - spacing / 2.0 - 0.002
        pts = pts[mask]
        if len(pts) >= n:
            idx = np.argsort(r[mask])[:n]
            return pts[idx].copy()
        spacing *= 0.98
    
    # Fallback: random points
    angles = rng.uniform(0, 2 * np.pi, n)
    radii = rng.uniform(0, 0.85, n)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _compute_radius(pts):
    """Compute the maximum radius for given points."""
    n = len(pts)
    if n < 2:
        return min(1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1])), 0.5)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    boundary_dists = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min_objective(x, n, beta=100.0):
    """
    Soft-min objective using log-sum-exp for smooth minimization.
    We want to maximize min(all distances/2, all boundary distances).
    Equivalent to minimizing -soft_min of all these values.
    """
    pts = x.reshape(n, 2)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Get upper triangle
    upper = np.triu_indices(n, k=1)
    pair_dists = dists[upper] / 2.0
    
    # Boundary distances
    boundary_dists = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    
    # Combine all values
    all_vals = np.concatenate([pair_dists, boundary_dists])
    
    # Soft min using log-sum-exp
    min_val = np.min(all_vals)
    if min_val < 0:
        # If any value is negative, penalize heavily
        neg_penalty = np.sum(all_vals[all_vals < 0] ** 2)
        return -min_val + neg_penalty
    
    # log-sum-exp for stability
    shifted = all_vals - min_val
    log_sum_exp = np.log(np.sum(np.exp(shifted))) + min_val
    soft_min = min_val - (1.0 / beta) * log_sum_exp
    
    # We want to maximize soft_min, so return negative
    return -soft_min


def _soft_min_gradient(x, n, beta=100.0):
    """
    Gradient of the soft-min objective with respect to x.
    """
    pts = x.reshape(n, 2)
    grad = np.zeros_like(x)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    upper = np.triu_indices(n, k=1)
    pair_dists = dists[upper] / 2.0
    
    # Boundary distances
    norms = np.hypot(pts[:, 0], pts[:, 1])
    boundary_dists = 1.0 - norms
    
    # Combine all values
    all_vals = np.concatenate([pair_dists, boundary_dists])
    n_pairs = len(pair_dists)
    n_boundary = len(boundary_dists)
    
    min_val = np.min(all_vals)
    
    if min_val < 0:
        # Gradient of penalty term
        for i in range(n_pairs):
            if pair_dists[i] < 0:
                idx_i, idx_j = upper[0][i], upper[1][i]
                d = diff[idx_i, idx_j]
                dist = dists[idx_i, idx_j]
                if dist > 1e-10:
                    grad[idx_i * 2:idx_i * 2 + 2] += 2 * pair_dists[i] * d / dist
                    grad[idx_j * 2:idx_j * 2 + 2] -= 2 * pair_dists[i] * d / dist
        for i in range(n_boundary):
            if boundary_dists[i] < 0:
                if norms[i] > 1e-10:
                    grad[i * 2:i * 2 + 2] += 2 * boundary_dists[i] * pts[i] / norms[i]
        return grad
    
    # log-sum-exp weights
    shifted = all_vals - min_val
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    
    # Gradient of soft_min = min_val - (1/beta) * log_sum_exp
    # d(soft_min)/dx = d(min_val)/dx - (1/beta) * sum(weights * d(all_vals)/dx)
    # But min_val is not differentiable, so we approximate with the weighted average
    
    # For pairwise distances
    for i in range(n_pairs):
        w = weights[i] / beta
        idx_i, idx_j = upper[0][i], upper[1][i]
        d = diff[idx_i, idx_j]
        dist = dists[idx_i, idx_j]
        if dist > 1e-10:
            # d(pair_dist)/dx_i = d/dx_i (||x_i - x_j||/2) = (x_i - x_j)/(2*dist) = d/(2*dist)
            # d(pair_dist)/dx_j = -d/(2*dist)
            grad[idx_i * 2:idx_i * 2 + 2] -= w * d / dist
            grad[idx_j * 2:idx_j * 2 + 2] += w * d / dist
    
    # For boundary distances
    for i in range(n_boundary):
        w = weights[n_pairs + i] / beta
        if norms[i] > 1e-10:
            # d(boundary_dist)/dx_i = -d(1-|x_i|)/dx_i = -x_i/|x_i|
            grad[i * 2:i * 2 + 2] += w * pts[i] / norms[i]
    
    return grad


def _anneal(pts, time_budget, rng):
    """Run simulated annealing to improve the packing."""
    n = len(pts)
    best_pts = pts.copy()
    best_score = _compute_radius(pts)
    current_pts = pts.copy()
    current_score = best_score
    
    # Initial temperature
    T = 0.005
    cooling = 0.9995
    
    while time.time() < time_budget:
        # Randomly perturb one or two points
        if rng.random() < 0.3:
            idx = rng.randint(n)
            delta = rng.normal(0, T, size=2)
            new_pts = current_pts.copy()
            new_pts[idx] += delta
        elif rng.random() < 0.6:
            idx1, idx2 = rng.choice(n, 2, replace=False)
            delta1 = rng.normal(0, T, size=2)
            delta2 = rng.normal(0, T, size=2)
            new_pts = current_pts.copy()
            new_pts[idx1] += delta1
            new_pts[idx2] += delta2
        else:
            # Move a small cluster
            n_move = rng.randint(3, 6)
            indices = rng.choice(n, n_move, replace=False)
            deltas = rng.normal(0, T * 0.7, size=(n_move, 2))
            new_pts = current_pts.copy()
            new_pts[indices] += deltas
        
        # Check boundary
        norms = np.linalg.norm(new_pts, axis=1)
        if np.any(norms > 1.0):
            new_score = 0.0
        else:
            new_score = _compute_radius(new_pts)
        
        # Accept or reject
        if new_score > current_score or rng.random() < np.exp((new_score - current_score) / max(T, 1e-10)):
            current_pts = new_pts
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        
        T *= cooling
    
    return best_pts, best_score


def construct():
    """
    Construct 300 equal circle centers inside the unit disk to maximize the minimum
    common radius. Uses hexagonal lattice initialization with L-BFGS-B optimization
    and simulated annealing refinement.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    target_r = 0.0531542991469
    
    # Generate initial hexagonal configuration
    best_pts = _hex_init(N, seed=42)
    best_score = _compute_radius(best_pts)
    
    # Run L-BFGS-B optimization with soft-min objective
    if time.time() < deadline - 40:
        try:
            result = minimize(
                _soft_min_objective,
                best_pts.flatten(),
                jac=_soft_min_gradient,
                args=(N, 100.0),
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            pts_opt = result.x.reshape(N, 2)
            # Clamp to unit disk
            norms = np.linalg.norm(pts_opt, axis=1)
            for i in range(N):
                if norms[i] > 1.0:
                    pts_opt[i] = pts_opt[i] / norms[i] * 0.999
            score_opt = _compute_radius(pts_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_pts = pts_opt
        except Exception:
            pass
    
    # Run simulated annealing refinement
    if time.time() < deadline - 15:
        rng = np.random.RandomState(12345)
        time_budget = deadline - 5
        best_pts, best_score = _anneal(best_pts, time_budget, rng)
    
    # Final check: ensure all points are valid
    norms = np.linalg.norm(best_pts, axis=1)
    for i in range(N):
        if norms[i] > 1.0:
            best_pts[i] = best_pts[i] / norms[i] * 0.999
    
    return best_pts
# EVOLVE-BLOCK-END
