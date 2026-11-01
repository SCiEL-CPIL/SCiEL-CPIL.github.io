import numpy as np
from scipy.optimize import minimize
import time

N = 300

def _compute_score(points):
    """Compute the packing score: min of all pairwise distances/2 and boundary distances."""
    p = points
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dists = dists / 2.0
    valid_dists = min_dists[np.triu_indices(N, 1)]
    boundary_dists = 1.0 - np.linalg.norm(p, axis=1)
    return np.min(np.concatenate([valid_dists, boundary_dists]))

def _hex_lattice_init(N_target):
    """Generate hexagonal lattice initialization."""
    r_init = 0.05
    spacing = 2.0 * r_init
    R_est = spacing * np.sqrt(N_target * np.sqrt(3.0) / (2.0 * np.pi)) * 1.15
    
    k = int(np.ceil(R_est / spacing)) + 2
    points = []
    for j in range(-k, k + 1):
        y = j * spacing * np.sqrt(3.0) / 2.0
        if abs(y) > R_est:
            continue
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            if np.hypot(x, y) <= R_est:
                points.append((x, y))
    
    points = np.array(points)
    if len(points) == 0:
        return np.zeros((N_target, 2))
        
    if len(points) > N_target:
        radii = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(radii)
        points = points[indices[:N_target]]
    elif len(points) < N_target:
        while len(points) < N_target:
            angle = np.random.uniform(0, 2*np.pi)
            rad = np.random.uniform(0, 0.95)
            points = np.vstack([points, [rad*np.cos(angle), rad*np.sin(angle)]])
    
    # Ensure strictly inside
    max_rad = np.max(np.hypot(points[:, 0], points[:, 1]))
    if max_rad > 0.98:
        scale = 0.98 / max_rad
        points *= scale
        
    return points

def _soft_min_and_grad(x, beta):
    """Compute soft-min objective and gradient efficiently."""
    p = x.reshape(-1, 2)
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1.0)
    boundary_dists = 1.0 - np.linalg.norm(p, axis=1)
    
    valid_dists = dists / 2.0
    all_vals = np.concatenate([valid_dists[np.triu_indices(N, 1)], boundary_dists])
    
    m = np.min(all_vals)
    exp_vals = np.exp(-beta * (all_vals - m))
    S = np.sum(exp_vals)
    weights = exp_vals / S
    
    n_pairs = len(valid_dists[np.triu_indices(N, 1)])
    w_dists = weights[:n_pairs]
    w_bounds = weights[n_pairs:]
    
    grad = np.zeros_like(x)
    
    norms = np.linalg.norm(p, axis=1)
    norms[norms == 0] = 1.0
    grad_bounds = -p / norms[:, np.newaxis] * w_bounds[:, np.newaxis]
    grad += grad_bounds
    
    W_full = np.zeros((N, N))
    i_idx, j_idx = np.triu_indices(N, 1)
    W_full[i_idx, j_idx] = w_dists
    W_full[j_idx, i_idx] = w_dists
    
    inv_dists = 1.0 / (2.0 * dists)
    np.fill_diagonal(inv_dists, 0.0)
    
    coeff = W_full * inv_dists
    grad_dists = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    grad += grad_dists
    
    soft_min = m - (1.0/beta) * np.log(S)
    obj_val = -soft_min
    
    return obj_val, grad

def _local_optimize(x, beta, maxiter, ftol, gtol):
    """Run L-BFGS-B optimization with given parameters."""
    def objective(x):
        val, _ = _soft_min_and_grad(x, beta)
        return val
    
    def grad(x):
        _, g = _soft_min_and_grad(x, beta)
        return g
    
    bounds = [(-1.05, 1.05)] * (2 * N)
    
    res = minimize(objective, x, jac=grad, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'disp': False})
    
    return res.x, res.fun

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    # Initialize with hexagonal lattice
    points = _hex_lattice_init(N)
    best_points = points.copy()
    best_score = _compute_score(best_points)
    
    # Extended beta schedule with more intermediate values for smoother convergence
    beta_schedule = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
    
    current_x = points.flatten()
    
    for beta in beta_schedule:
        if time.time() > deadline - 15:  # Reserve time for final polish
            break
            
        # Use more iterations for lower beta, fewer for higher
        if beta < 100:
            maxiter = 300
            ftol = 1e-14
            gtol = 1e-10
        else:
            maxiter = 200
            ftol = 1e-14
            gtol = 1e-10
            
        try:
            new_x, fun_val = _local_optimize(current_x, beta, maxiter, ftol, gtol)
            
            # Validate and update best
            new_points = new_x.reshape(-1, 2)
            radii = np.hypot(new_points[:, 0], new_points[:, 1])
            if np.all(radii <= 1.0) and np.all(np.isfinite(new_points)):
                score = _compute_score(new_points)
                if score > best_score:
                    best_score = score
                    best_points = new_points.copy()
                    current_x = new_x
        except Exception:
            pass
    
    # Final high-precision polish pass with very high beta
    if time.time() < deadline - 5:
        for beta in [5000.0, 10000.0, 20000.0, 50000.0, 100000.0]:
            if time.time() > deadline - 3:
                break
                
            try:
                new_x, fun_val = _local_optimize(current_x, beta, 100, 1e-15, 1e-12)
                
                new_points = new_x.reshape(-1, 2)
                radii = np.hypot(new_points[:, 0], new_points[:, 1])
                if np.all(radii <= 1.0) and np.all(np.isfinite(new_points)):
                    score = _compute_score(new_points)
                    if score > best_score:
                        best_score = score
                        best_points = new_points.copy()
                        current_x = new_x
            except Exception:
                break
    
    # Final check: ensure points are valid (strictly inside circle)
    radii = np.hypot(best_points[:, 0], best_points[:, 1])
    if np.any(radii >= 1.0):
        max_r = np.max(radii)
        if max_r >= 1.0:
            scale = 0.9999 / max_r
            best_points *= scale
            best_score = _compute_score(best_points)
    
    return best_points
