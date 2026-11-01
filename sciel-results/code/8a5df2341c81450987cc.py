"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _pairwise_min_dist(pts):
    """Compute minimum pairwise distance using vectorized NumPy."""
    n = len(pts)
    # Compute all pairwise squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    sq_dists = np.sum(diff * diff, axis=2)
    # Set diagonal to inf
    np.fill_diagonal(sq_dists, np.inf)
    min_sq = np.min(sq_dists)
    return np.sqrt(min_sq)

def _compute_score(pts):
    """Compute the packing score (max radius)."""
    if len(pts) != N:
        return -1.0
    if not np.all(np.isfinite(pts)):
        return -1.0
    dists = np.linalg.norm(pts, axis=1)
    if np.any(dists > 1.0):
        return -1.0
    min_dist = _pairwise_min_dist(pts)
    r = min(min_dist / 2.0, 1.0 - np.max(dists))
    return r

def _soft_min(values, beta):
    """Compute soft minimum using log-sum-exp trick."""
    values = np.asarray(values)
    m = np.min(values)
    if beta < 1e-10:
        return m
    exp_vals = np.exp(-beta * (values - m))
    s = np.sum(exp_vals)
    return m - np.log(s) / beta

def _soft_min_grad(values, beta):
    """Gradient of soft-min with respect to values."""
    values = np.asarray(values)
    m = np.min(values)
    if beta < 1e-10:
        grad = np.zeros_like(values)
        grad[np.argmin(values)] = 1.0
        return grad
    exp_vals = np.exp(-beta * (values - m))
    s = np.sum(exp_vals)
    weights = exp_vals / s
    return -weights

def _objective_and_grad(params, fixed_points, opt_indices, beta, deadline):
    """Objective and gradient for the soft-min packing problem."""
    if time.time() > deadline:
        return 1e10, None
    
    pts = fixed_points.copy()
    pts[opt_indices] = params.reshape(-1, 2)
    
    # Check inside container
    dists = np.linalg.norm(pts, axis=1)
    if np.any(dists > 1.0):
        return 1e10, None
    
    n = len(pts)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    sq_dists = np.sum(diff * diff, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Pairwise distances
    dists_pair = np.sqrt(sq_dists)
    
    # Container distances: 1 - |c_i|
    container_dists = 1.0 - dists
    
    # We want to maximize min of all pairwise dists/2 and container_dists
    # So minimize -soft_min of (pairwise_dists/2, container_dists)
    
    # Collect all constraints
    # Upper triangle of pairwise distances
    iu = np.triu_indices(n, k=1)
    pair_dists = dists_pair[iu] / 2.0  # radius constraint from pairs
    
    all_constraints = np.concatenate([pair_dists, container_dists])
    
    # Soft min
    sm = _soft_min(all_constraints, beta)
    obj = -sm
    
    # Gradient with respect to points
    if beta < 1e-10:
        # Subgradient: pick the minimum constraint
        min_idx = np.argmin(all_constraints)
        grad_pts = np.zeros_like(pts)
        if min_idx < len(pair_dists):
            # It's a pair constraint
            # Map back to indices
            n_pairs = len(pair_dists)
            pair_i = iu[0][min_idx]
            pair_j = iu[1][min_idx]
            d = pts[pair_i] - pts[pair_j]
            norm_d = np.linalg.norm(d)
            if norm_d > 1e-12:
                grad_pts[pair_i] += d / (2.0 * norm_d)
                grad_pts[pair_j] -= d / (2.0 * norm_d)
        else:
            # Container constraint
            idx = min_idx - len(pair_dists)
            norm_c = np.linalg.norm(pts[idx])
            if norm_c > 1e-12:
                grad_pts[idx] += pts[idx] / norm_c
        
        grad_params = grad_pts[opt_indices].flatten()
        return obj, grad_params
    
    # Smooth gradient
    weights = _soft_min_grad(all_constraints, beta)
    
    grad_pts = np.zeros_like(pts)
    n_pairs = len(pair_dists)
    
    # Pair constraints
    for k in range(n_pairs):
        w = weights[k]
        if abs(w) < 1e-12:
            continue
        pair_i = iu[0][k]
        pair_j = iu[1][k]
        d = pts[pair_i] - pts[pair_j]
        norm_d = np.linalg.norm(d)
        if norm_d > 1e-12:
            # d/d(pts_i) of (|pts_i - pts_j|/2) = (pts_i - pts_j) / (2*|pts_i - pts_j|)
            # d/d(pts_j) of (|pts_i - pts_j|/2) = -(pts_i - pts_j) / (2*|pts_i - pts_j|)
            grad_pts[pair_i] += w * d / (2.0 * norm_d)
            grad_pts[pair_j] -= w * d / (2.0 * norm_d)
    
    # Container constraints
    for k in range(len(container_dists)):
        w = weights[n_pairs + k]
        if abs(w) < 1e-12:
            continue
        norm_c = np.linalg.norm(pts[k])
        if norm_c > 1e-12:
            # d/d(pts_k) of (1 - |pts_k|) = -pts_k / |pts_k|
            grad_pts[k] -= w * pts[k] / norm_c
    
    grad_params = grad_pts[opt_indices].flatten()
    return obj, grad_params

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # First, get a good initial packing using hexagonal lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            points = points[np.argsort(radius, kind="stable")[:N]]
            break
        spacing *= 0.98
    
    initial_score = _compute_score(points)
    if initial_score <= 0:
        return points
    
    best_points = points.copy()
    best_score = initial_score
    
    # Try to improve using local optimization with soft-min
    # Optimize all points for better results
    n_opt = N
    opt_indices = np.arange(n_opt)
    
    # Adaptive beta schedule: start with low beta (smooth), increase to high beta (sharp)
    beta_schedule = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    for beta in beta_schedule:
        if time.time() > deadline - 15:
            break
        
        def objective(params):
            obj, _ = _objective_and_grad(params, points, opt_indices, beta, deadline)
            return obj
        
        def jac(params):
            _, grad = _objective_and_grad(params, points, opt_indices, beta, deadline)
            if grad is None:
                return np.zeros_like(params)
            return grad
        
        x0 = points[opt_indices].flatten()
        
        # Bounds to keep points inside container
        bounds = [(-0.999, 0.999)] * (2 * n_opt)
        
        try:
            result = minimize(objective, x0, jac=jac, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 30, 'maxfun': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if result.fun < 1e9:
                new_points = points.copy()
                new_points[opt_indices] = result.x.reshape(-1, 2)
                new_score = _compute_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
                    points = new_points.copy()
        except Exception:
            pass
    
    points = best_points
    
    # If we have time, try a final high-precision polish pass
    if time.time() < deadline - 20:
        # Use very high beta for final polish
        beta_final = 500.0
        n_opt = N
        opt_indices = np.arange(n_opt)
        
        def objective(params):
            obj, _ = _objective_and_grad(params, points, opt_indices, beta_final, deadline)
            return obj
        
        def jac(params):
            _, grad = _objective_and_grad(params, points, opt_indices, beta_final, deadline)
            if grad is None:
                return np.zeros_like(params)
            return grad
        
        x0 = points[opt_indices].flatten()
        bounds = [(-0.9999, 0.9999)] * (2 * n_opt)
        
        try:
            result = minimize(objective, x0, jac=jac, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 50, 'maxfun': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.fun < 1e9:
                new_points = points.copy()
                new_points[opt_indices] = result.x.reshape(-1, 2)
                new_score = _compute_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
        except Exception:
            pass
    
    points = best_points
    
    # Final validation
    final_score = _compute_score(points)
    if final_score <= 0:
        return points
    
    return points
# EVOLVE-BLOCK-END
