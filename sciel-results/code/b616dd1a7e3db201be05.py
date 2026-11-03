"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _compute_score(pts):
    """Compute the packing radius: min of half-min-distance and min(1-|c_i|)."""
    n = len(pts)
    if n < 2:
        return 0.0
    # Pairwise distances
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    # Distance to boundary
    r = np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(1.0 - r)
    return min(min_dist / 2.0, min_boundary)

def _soft_min(values, beta):
    """
    Smooth approximation to min(values) using log-sum-exp.
    """
    m = np.min(values)
    if np.max(values) - m < 1e-12:
        return m
    exps = np.exp(-beta * (values - m))
    log_sum = np.log(np.sum(exps))
    return m - log_sum / beta

def _soft_min_grad(values, beta):
    """
    Gradient of soft_min with respect to values.
    """
    m = np.min(values)
    if np.max(values) - m < 1e-12:
        return np.ones_like(values) / len(values)
    exps = np.exp(-beta * (values - m))
    sum_exps = np.sum(exps)
    return -exps / sum_exps

def _generate_hex_init(n):
    """Generate hexagonal lattice points clipped to unit disk, select n closest to center."""
    spacing = 0.10
    k = int(1.1 / spacing) + 2
    points = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            points.append([x, y])
    points = np.array(points)
    r = np.hypot(points[:, 0], points[:, 1])
    mask = r <= 1.0 - spacing / 2.0
    points = points[mask]
    
    if len(points) < n:
        pass

    # Sort by distance to center and take the first n
    r = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(r, kind="stable")[:n]
    return points[idx]

def _compute_soft_min_objective_and_grad(pts, beta):
    """
    Compute the soft-min objective (to maximize) and its gradient.
    Objective = soft_min([dist_ij/2 for all i<j, (1-|c_i|) for all i])
    """
    n = len(pts)
    # Pairwise distances
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    
    # All pairwise distances (upper triangle)
    iu, ju = np.triu_indices(n, k=1)
    pair_dists = dist[iu, ju]
    
    # Boundary gaps
    r = np.hypot(pts[:, 0], pts[:, 1])
    boundary_gaps = 1.0 - r
    
    # Combine all constraints: pair_dists/2 and boundary_gaps
    all_values = np.concatenate([pair_dists / 2.0, boundary_gaps])
    
    # Soft min value
    soft_min_val = _soft_min(all_values, beta)
    
    # Gradient of soft min w.r.t. all_values
    val_grad = _soft_min_grad(all_values, beta)
    
    # Now compute gradient w.r.t. pts
    grad_pts = np.zeros_like(pts)
    
    # Pairwise distances: d(dist_ij)/d(c_i) = (c_i - c_j) / dist_ij
    for idx in range(len(iu)):
        i, j = iu[idx], ju[idx]
        d = dist[i, j]
        if d < 1e-10:
            continue
        vec = (pts[i] - pts[j]) / d
        grad_pts[i] += val_grad[idx] * vec * 0.5
        grad_pts[j] -= val_grad[idx] * vec * 0.5
    
    # Boundary: d(boundary_gap_i)/d(c_i) = -c_i / |c_i|
    for i in range(n):
        ri = r[i]
        if ri < 1e-10:
            continue
        vec = -pts[i] / ri
        grad_pts[i] += val_grad[len(pair_dists) + i] * vec
    
    return soft_min_val, grad_pts

def _neg_soft_min_objective(pts_flat, beta):
    """Negative of soft-min objective for minimization."""
    pts = pts_flat.reshape(-1, 2)
    val, _ = _compute_soft_min_objective_and_grad(pts, beta)
    return -val

def _neg_soft_min_grad(pts_flat, beta):
    """Negative gradient of soft-min objective for minimization."""
    pts = pts_flat.reshape(-1, 2)
    _, grad = _compute_soft_min_objective_and_grad(pts, beta)
    return -grad.flatten()

def _optimize_with_lbfgsb(pts, time_limit, beta_start=100.0, beta_end=10000.0):
    """
    Optimize using scipy.optimize.minimize with L-BFGS-B.
    """
    start_time = time.time()
    n = len(pts)
    pts = pts.copy().astype(np.float64)
    
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    
    phases = [
        (beta_start, 0.35),
        (beta_start * 10, 0.30),
        (beta_end, 0.30),
    ]
    
    for beta, time_frac in phases:
        phase_end = start_time + time_limit * time_frac
        
        x0 = pts.flatten()
        
        def objective(x):
            return _neg_soft_min_objective(x, beta)
        
        def gradient(x):
            return _neg_soft_min_grad(x, beta)
        
        # Use L-BFGS-B with bounds to keep points inside unit circle
        bounds = [(-1.0, 1.0)] * (2 * n)
        
        try:
            result = minimize(
                objective,
                x0,
                jac=gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        except Exception:
            break
        
        if result.success or result.fun < objective(x0):
            new_pts = result.x.reshape(-1, 2)
            # Project inside unit circle
            r_new = np.hypot(new_pts[:, 0], new_pts[:, 1])
            mask_out = r_new >= 1.0
            if np.any(mask_out):
                new_pts[mask_out] *= (0.999 / r_new[mask_out])
            
            new_score = _compute_score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                pts = new_pts
        
        if time.time() > phase_end - 2.0:
            break
    
    return best_pts, best_score

def _polish(pts, time_limit):
    """
    Final high-precision polish using very sharp soft-min with L-BFGS-B.
    """
    start_time = time.time()
    n = len(pts)
    pts = pts.copy().astype(np.float64)
    
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    
    beta = 50000.0
    
    x0 = pts.flatten()
    bounds = [(-1.0, 1.0)] * (2 * n)
    
    def objective(x):
        return _neg_soft_min_objective(x, beta)
    
    def gradient(x):
        return _neg_soft_min_grad(x, beta)
    
    try:
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
        )
    except Exception:
        return best_pts, best_score
    
    if result.success or result.fun < objective(x0):
        new_pts = result.x.reshape(-1, 2)
        r_new = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask_out = r_new >= 1.0
        if np.any(mask_out):
            new_pts[mask_out] *= (0.999 / r_new[mask_out])
        
        new_score = _compute_score(new_pts)
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
    
    return best_pts, best_score

def construct():
    """
    Construct N=300 equal circles in unit disk.
    Uses hexagonal lattice init + L-BFGS-B soft-min optimization + polish.
    """
    start_time = time.time()
    total_time_limit = 115.0  # Safety margin before 120s kill
    
    # Generate initial hexagonal lattice
    pts = _generate_hex_init(N)
    
    # Main optimization with L-BFGS-B soft-min
    main_pts, main_score = _optimize_with_lbfgsb(pts, total_time_limit * 0.85)
    
    # Final polish with very sharp soft-min
    final_pts, final_score = _polish(main_pts, total_time_limit * 0.15)
    
    # Final validation: ensure all points are strictly inside
    r = np.hypot(final_pts[:, 0], final_pts[:, 1])
    for i in range(len(final_pts)):
        if r[i] >= 1.0:
            final_pts[i] *= 0.999 / r[i]
    
    return final_pts
# EVOLVE-BLOCK-END
