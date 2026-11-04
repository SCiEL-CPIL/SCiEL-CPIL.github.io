import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13
TIME_LIMIT = 145.0  # Safety margin

def _decode_params(params):
    """
    Decode polar parameters to Cartesian coordinates.
    params: array of shape (2*N,)
      - First N values: angles (radians)
      - Last N values: log-radii (so radii = exp(log_r) > 0)
    Returns: array of shape (N, 2)
    """
    angles = params[:N]
    log_radii = params[N:]
    radii = np.exp(log_radii)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])

def _all_triangle_areas(pts):
    """Compute areas of all C(n,3) triangles. Returns array of shape (C(n,3),)."""
    n = len(pts)
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Precompute indices for i < j < k
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    xi, xj, xk = x[i_flat], x[j_flat], x[k_flat]
    yi, yj, yk = y[i_flat], y[j_flat], y[k_flat]
    
    cross = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area_fast(pts):
    """Compute convex hull area efficiently."""
    n = len(pts)
    if n < 3:
        return 0.0
    
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _true_score(pts):
    """Compute min triangle area / convex hull area."""
    try:
        hull_area = _hull_area_fast(pts)
        if hull_area < 1e-12:
            return 0.0
        min_area = np.min(_all_triangle_areas(pts))
        return min_area / hull_area
    except Exception:
        return 0.0

def _objective_polar(params, beta):
    """
    Objective for SLSQP in polar coordinates.
    Minimizes: -soft_min(areas) / hull_area
    Using log-sum-exp for numerical stability.
    """
    pts = _decode_params(params)
    
    # Check for degenerate configurations
    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return 1e10
    
    areas = _all_triangle_areas(pts)
    if np.any(areas < 0) or np.all(areas < 1e-15):
        return 1e10
    
    hull_area = _hull_area_fast(pts)
    if hull_area < 1e-9:
        return 1e10
    
    # Log-sum-exp for stability
    max_a = np.max(areas)
    shifted = areas - max_a
    log_sum_exp = -beta * max_a + np.log(np.sum(np.exp(-beta * shifted)))
    
    # soft_min = -log_sum_exp / beta
    # We want to minimize -soft_min/hull_area = log_sum_exp / (beta * hull_area)
    return log_sum_exp / (beta * hull_area)

def _grad_objective_polar(params, beta):
    """
    Numerical gradient for SLSQP in polar coordinates.
    Uses central differences.
    """
    grad = np.zeros_like(params)
    h = 1e-7
    f0 = _objective_polar(params, beta)
    
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += h
        params_minus = params.copy()
        params_minus[i] -= h
        grad[i] = (_objective_polar(params_plus, beta) - _objective_polar(params_minus, beta)) / (2 * h)
    
    return grad

def _local_polish_cartesian(pts, time_budget, start_time):
    """
    High-precision local search using Nelder-Mead on the true score in Cartesian coordinates.
    """
    deadline = start_time + time_budget
    
    def neg_score(x):
        return -_true_score(x.reshape(N, 2))
    
    best_x = pts.flatten()
    best_score = -neg_score(best_x)
    
    # Run Nelder-Mead
    try:
        res = minimize(neg_score, best_x, method='Nelder-Mead',
                       options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 800, 'maxfev': 1500})
        
        if time.time() < deadline:
            if res.fun < -best_score:
                best_x = res.x
                best_score = -res.fun
    except Exception:
        pass
    
    # Final coordinate descent polish
    step_sizes = [1e-4, 1e-5, 1e-6]
    for step in step_sizes:
        if time.time() > deadline - 1:
            break
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(N):
                for d in range(2):
                    for delta in [step, -step]:
                        trial_x = best_x.copy()
                        trial_x[i*2 + d] += delta
                        trial_pts = trial_x.reshape(N, 2)
                        s = _true_score(trial_pts)
                        if s > best_score + 1e-12:
                            best_x = trial_x
                            best_score = s
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    return best_x.reshape(N, 2), best_score

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Initial guess: Regular 13-gon in polar coordinates
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii = np.ones(N)  # Unit circle
    log_radii = np.zeros(N)
    
    # Initial params: [angles, log_radii]
    params0 = np.concatenate([angles, log_radii])
    
    best_pts = _decode_params(params0)
    best_score = _true_score(best_pts)
    
    # SLSQP optimization with increasing beta (soft-min annealing)
    betas = [5, 20, 50, 100, 200, 500]
    
    current_params = params0.copy()
    
    for beta in betas:
        if time.time() > deadline - 40:
            break
        
        try:
            res = minimize(
                _objective_polar,
                current_params,
                args=(beta,),
                jac=_grad_objective_polar,
                method='SLSQP',
                options={'maxiter': 300, 'ftol': 1e-12, 'disp': False}
            )
            
            if time.time() < deadline:
                trial_pts = _decode_params(res.x)
                trial_score = _true_score(trial_pts)
                
                if trial_score > best_score:
                    best_score = trial_score
                    best_pts = trial_pts
                
                # Update current_params for next iteration
                current_params = res.x.copy()
                
        except Exception:
            pass
    
    # Final local polish on the best result found
    remaining_time = deadline - time.time() - 5
    if remaining_time > 5:
        polished_pts, polished_score = _local_polish_cartesian(best_pts, remaining_time, start_time)
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished_pts
    
    return best_pts
