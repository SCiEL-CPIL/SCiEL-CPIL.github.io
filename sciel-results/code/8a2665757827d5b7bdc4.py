"""Optimized packing of 50 equal circles in unit circle using smooth optimization."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _smooth_min(values, beta):
    """Smooth approximation of min using log-sum-exp."""
    # -log(sum(exp(-beta * x))) / beta approximates min(x) for large beta
    m = np.min(values)
    shifted = values - m
    log_sum = m + np.log(np.sum(np.exp(-beta * shifted)))
    return -log_sum / beta

def _compute_clearances(pts):
    """Compute all clearance values: half pairwise distances and boundary distances."""
    n = len(pts)
    # Pairwise distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    half_dists = dists / 2.0
    
    # Boundary clearances
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    boundary = 1.0 - norms
    
    # Combine all clearances
    return np.concatenate([half_dists.ravel(), boundary])

def _negative_smooth_min(pts, beta):
    """Objective: negative of smooth min clearance (to minimize)."""
    clearances = _compute_clearances(pts)
    return -_smooth_min(clearances, beta)

def _objective_and_grad(x, beta):
    """Objective and gradient for L-BFGS-B."""
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    # Compute clearances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1) + 1e-30)
    np.fill_diagonal(dists, np.inf)
    half_dists = dists / 2.0
    
    norms = np.sqrt(np.sum(pts ** 2, axis=1) + 1e-30)
    boundary = 1.0 - norms
    
    all_clear = np.concatenate([half_dists.ravel(), boundary])
    
    # Smooth min: f = -log(sum(exp(-beta * x))) / beta
    m = np.min(all_clear)
    shifted = all_clear - m
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    
    if sum_exp <= 0:
        return 0.0, np.zeros_like(x)
    
    f = -(m + np.log(sum_exp)) / beta
    
    # Gradient of f w.r.t. x
    # df/d(all_clear) = exp(-beta * (all_clear - m)) / sum_exp
    weights = exp_vals / sum_exp
    
    # df/d(pts) = sum over clearances of (df/dclearance) * (dclearance/dpts)
    grad_pts = np.zeros_like(pts)
    
    # Pairwise distance gradients
    # d(half_dist_ij)/d(pts_i) = (pts_i - pts_j) / (2 * dist_ij)
    # d(half_dist_ij)/d(pts_j) = (pts_j - pts_i) / (2 * dist_ij)
    w_pair = weights[:n*n].reshape(n, n)
    np.fill_diagonal(w_pair, 0.0)
    
    # For each pair (i,j), contribution to grad_pts[i] += w_pair[i,j] * (pts[i]-pts[j]) / (2*dist[i,j])
    # and grad_pts[j] += w_pair[i,j] * (pts[j]-pts[i]) / (2*dist[i,j])
    
    # Vectorized: grad_pts[i] = sum_j w_pair[i,j] * (pts[i] - pts[j]) / (2 * dists[i,j])
    # = sum_j w_pair[i,j] / (2*dists[i,j]) * pts[i] - sum_j w_pair[i,j] / (2*dists[i,j]) * pts[j]
    
    coeff = w_pair / (2.0 * dists)
    np.fill_diagonal(coeff, 0.0)
    
    # grad_pts[i] = (sum_j coeff[i,j]) * pts[i] - sum_j coeff[i,j] * pts[j]
    row_sum = np.sum(coeff, axis=1)
    grad_pts += row_sum[:, np.newaxis] * pts
    grad_pts -= coeff @ pts
    
    # Boundary gradients
    # d(1 - |pts_i|)/d(pts_i) = -pts_i / |pts_i|
    w_boundary = weights[n*n:]
    grad_pts -= w_boundary[:, np.newaxis] * pts / (norms[:, np.newaxis] + 1e-30)
    
    return f, grad_pts.ravel()

def _initial_hexagonal():
    """Generate initial points using hexagonal lattice."""
    spacing = 0.25
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            # Sort by radius and take first N
            idx = np.argsort(np.hypot(points[:, 0], points[:, 1]), kind='stable')
            return points[idx[:N]]
        spacing *= 0.98

def _initial_random(seed):
    """Generate random initial points inside unit circle."""
    rng = np.random.RandomState(seed)
    pts = []
    while len(pts) < N:
        angle = rng.uniform(0, 2 * np.pi)
        r = np.sqrt(rng.uniform(0, 1))
        pts.append([r * np.cos(angle), r * np.sin(angle)])
    return np.array(pts)

def _evaluate(pts):
    """Compute the actual packing radius."""
    n = len(pts)
    if n != N:
        return 0.0
    # Check all points are finite
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check all points are inside unit circle
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    if np.any(norms > 1.0):
        return 0.0
    
    # Pairwise distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Boundary
    min_boundary = 1.0 - np.max(norms)
    
    return min(min_pair, min_boundary)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin
    
    best_pts = None
    best_score = 0.0
    
    # Start with hexagonal lattice
    hex_pts = _initial_hexagonal()
    best_pts = hex_pts.copy()
    best_score = _evaluate(hex_pts)
    
    # Try multiple optimization runs
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        if attempt == 0:
            x0 = hex_pts.copy()
        else:
            x0 = _initial_random(attempt * 7 + 1)
        
        # Start with moderate beta, increase over iterations
        beta = 50.0
        
        def obj(x):
            val, _ = _objective_and_grad(x, beta)
            return val
        
        def grad(x):
            _, g = _objective_and_grad(x, beta)
            return g
        
        try:
            result = minimize(
                obj, x0.ravel(), method='L-BFGS-B', jac=grad,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            if result.success or result.fun < obj(x0.ravel()):
                new_pts = result.x.reshape(-1, 2)
                new_score = _evaluate(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
        except Exception:
            pass
        
        # If we have time, do a polish pass with higher beta
        if time.time() < deadline - 15.0 and best_pts is not None:
            beta_polish = 200.0
            x0_polish = best_pts.ravel()
            
            def obj_polish(x):
                val, _ = _objective_and_grad(x, beta_polish)
                return val
            
            def grad_polish(x):
                _, g = _objective_and_grad(x, beta_polish)
                return g
            
            try:
                result2 = minimize(
                    obj_polish, x0_polish, method='L-BFGS-B', jac=grad_polish,
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                )
                if result2.success or result2.fun < obj_polish(x0_polish):
                    new_pts = result2.x.reshape(-1, 2)
                    new_score = _evaluate(new_pts)
                    if new_score > best_score:
                        best_score = new_score
                        best_pts = new_pts.copy()
            except Exception:
                pass
    
    return best_pts
# EVOLVE-BLOCK-END
