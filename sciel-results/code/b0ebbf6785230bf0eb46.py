import numpy as np
import time
from scipy.optimize import minimize
import itertools
from functools import partial

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """
    Returns a flat array of squared pairwise distances for N points.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def fibonacci_sphere(n, radius=1.0):
    """Generate n points evenly distributed on a sphere using the Fibonacci lattice."""
    points = np.zeros((n, 3))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        z = 1.0 - 2.0 * i / (n - 1)
        r = np.sqrt(1.0 - z**2)
        theta = golden_angle * i
        points[i] = [r * np.cos(theta), r * np.sin(theta), z]
    return points * radius

def objective_log_ratio(x, tau=50.0):
    """
    Log-domain smooth approximation of log(max/min) of squared distances.
    This is better conditioned than the ratio directly.
    x is flattened array of shape (N*D,)
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    log_dists = np.log(sq_dists)
    
    # Soft max in log domain
    m_max = np.max(log_dists)
    exp_max_terms = np.exp(tau * (log_dists - m_max))
    soft_max_log = m_max + np.log(np.sum(exp_max_terms)) / tau
    
    # Soft min in log domain  
    m_min = np.min(log_dists)
    exp_min_terms = np.exp(-tau * (log_dists - m_min))
    soft_min_log = m_min - np.log(np.sum(exp_min_terms)) / tau
    
    # We want to minimize max/min, which is equivalent to minimizing log(max) - log(min)
    return soft_max_log - soft_min_log

def objective_epigraph(x, t, tau=100.0):
    """
    Epigraph formulation: minimize max_dist^2 + penalty for min_dist^2 < t
    x is flattened points, t is the auxiliary variable for min distance squared
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft max of squared distances
    m_max = np.max(sq_dists)
    exp_max_terms = np.exp(tau * (sq_dists - m_max))
    soft_max = m_max + np.log(np.sum(exp_max_terms)) / tau
    
    # Penalty if any distance squared is below t
    # Use smooth approximation: sum of max(0, t - d_i^2) with smoothing
    violations = np.maximum(t - sq_dists, 0.0)
    # Smooth the max(0, .) with a softplus-like function
    smooth_violations = np.log1p(np.exp(50.0 * violations)) / 50.0
    
    return soft_max + 10.0 * np.sum(smooth_violations)

def raw_score(points):
    """Compute the actual (non-smooth) score."""
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-10:
        return np.inf
    return np.max(sq_dists) / np.min(sq_dists)

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin for 120s limit
    
    best_score = np.inf
    best_points = None
    
    def try_update(pts):
        nonlocal best_score, best_points
        if time.time() - start_time > timeout:
            return
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()
    
    # Initial candidates
    # 1. Grid-based initialization (current best approach)
    points_grid = []
    for x in range(3):
        for y in range(3):
            for z in range(2):
                points_grid.append((x, y, z))
    to_remove = [(0,0,0), (0,0,1), (0,2,0), (0,2,1)]
    final_points_grid = [p for p in points_grid if p not in to_remove]
    init_grid = np.array(final_points_grid, dtype=float)
    init_grid = init_grid - np.mean(init_grid, axis=0)
    
    candidates = [init_grid.flatten()]
    
    # 2. Fibonacci sphere
    fib_pts = fibonacci_sphere(N, radius=2.0)
    candidates.append(fib_pts.flatten())
    
    # 3. Random starts with better spread
    rng = np.random.default_rng(42)
    num_random_starts = 12
    for i in range(num_random_starts):
        if time.time() - start_time > timeout:
            break
        # Mix of sphere and box distributions
        if i % 2 == 0:
            pts = rng.normal(size=(N, D))
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms * 2.0
        else:
            pts = rng.uniform(-2, 2, size=(N, D))
        candidates.append(pts.flatten())
    
    # 4. Perturbed versions
    for _ in range(4):
        if time.time() - start_time > timeout:
            break
        perturbed = init_grid + rng.normal(size=(N, D)) * 0.15
        candidates.append(perturbed.flatten())
    
    for _ in range(3):
        if time.time() - start_time > timeout:
            break
        perturbed_fib = fib_pts + rng.normal(size=(N, D)) * 0.15
        candidates.append(perturbed_fib.flatten())
    
    # Optimization using log-ratio formulation (better conditioned)
    tau_schedule = [20.0, 100.0, 500.0, 2000.0]
    maxiter_schedule = [300, 400, 500, 600]
    
    for i, init_x in enumerate(candidates):
        if time.time() - start_time > timeout:
            break
            
        x_curr = init_x.copy()
        
        try:
            # Multi-stage optimization with increasing tau
            for idx, tau in enumerate(tau_schedule):
                if time.time() - start_time > timeout:
                    break
                
                obj = partial(objective_log_ratio, tau=tau)
                maxiter = maxiter_schedule[idx]
                
                res = minimize(
                    obj, 
                    x_curr, 
                    method='L-BFGS-B', 
                    options={'maxiter': maxiter, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 50}
                )
                
                if res.x is not None and res.fun < np.inf:
                    x_curr = res.x
            
            # Final polish with high tau
            if time.time() - start_time < timeout - 8.0:
                try:
                    obj_final = partial(objective_log_ratio, tau=5000.0)
                    res_polish = minimize(
                        obj_final,
                        x_curr,
                        method='L-BFGS-B',
                        options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-12}
                    )
                    if res_polish.x is not None and res_polish.fun < np.inf:
                        x_curr = res_polish.x
                except Exception:
                    pass
            
            # Try SLSQP polish if time permits
            if time.time() - start_time < timeout - 5.0:
                try:
                    obj_slsqp = partial(objective_log_ratio, tau=10000.0)
                    res_slsqp = minimize(
                        obj_slsqp,
                        x_curr,
                        method='SLSQP',
                        options={'maxiter': 200, 'ftol': 1e-14}
                    )
                    if res_slsqp.x is not None and res_slsqp.fun < np.inf:
                        x_curr = res_slsqp.x
                except Exception:
                    pass
            
            pts = x_curr.reshape(N, D)
            try_update(pts)
            
        except Exception:
            pass
    
    # Additional pass: try epigraph formulation on best found so far
    if best_points is not None and time.time() - start_time < timeout - 15.0:
        try:
            x_curr = best_points.flatten().copy()
            # Initialize t as the current min distance squared
            sq_dists_curr = get_pairwise_sq_dists(best_points)
            t_init = np.min(sq_dists_curr) * 0.95
            
            # Optimize (x, t) jointly
            def joint_obj(z):
                x_part = z[:N*D]
                t_part = z[N*D]
                return objective_epigraph(x_part, t_part, tau=200.0)
            
            z_init = np.concatenate([x_curr, [t_init]])
            
            for tau_ep in [50.0, 200.0, 500.0]:
                if time.time() - start_time > timeout - 5.0:
                    break
                obj_j = partial(objective_epigraph, tau=tau_ep)
                
                def joint_obj_tau(z):
                    x_part = z[:N*D]
                    t_part = z[N*D]
                    return objective_epigraph(x_part, t_part, tau=tau_ep)
                
                res_joint = minimize(
                    joint_obj_tau,
                    z_init,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-11}
                )
                
                if res_joint.x is not None and res_joint.fun < np.inf:
                    z_init = res_joint.x
            
            pts_joint = z_init[:N*D].reshape(N, D)
            try_update(pts_joint)
            
        except Exception:
            pass
    
    # If optimization didn't find anything, fall back to grid
    if best_points is None:
        best_points = init_grid
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    sq_dists = get_pairwise_sq_dists(pts)
    score = (np.max(sq_dists) / np.min(sq_dists))
    print(f"Score: {score}")
