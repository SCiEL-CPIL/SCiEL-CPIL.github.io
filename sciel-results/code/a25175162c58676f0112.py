"""
Packing 100 equal spheres in a unit cube using a structural optimization approach.

Strategy:
1. Initialize with a well-structured grid
2. Use a repulsive energy formulation with smooth min approximation
3. Multi-start local optimization with L-BFGS-B
4. Final high-precision polish with increasing beta values
5. Deadline guard to ensure completion within 150s
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 100

def _score(points):
    """Return the maximal radius for the given set of centres."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Inside the cube [-0.5, 0.5]^3
    margin = 0.5 - np.abs(pts).max(axis=1)
    r_margin = margin.min()
    if r_margin <= 0:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    r_pair = min_dist / 2.0
    return min(r_pair, r_margin)

def _soft_min(a, b, beta=50.0):
    """Smooth approximation of min(a, b) using log-sum-exp."""
    m = np.minimum(a, b)
    return m - np.logaddexp(-beta * (a - m), -beta * (b - m)) / beta

def _neg_soft_radius(x, beta=50.0):
    """Smooth objective: -soft_min(r_pair, r_margin)."""
    pts = x.reshape(N, 3)
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return 1e10
    margin = 0.5 - np.abs(pts).max(axis=1)
    r_margin = margin.min()
    if r_margin <= 0:
        return 1e10
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = dists.min() / 2.0
    return -_soft_min(r_pair, r_margin, beta)

def _grad_neg_soft_radius(x, beta=50.0):
    """Analytical gradient of the negative soft radius objective."""
    pts = x.reshape(N, 3)
    margin = 0.5 - np.abs(pts).max(axis=1)
    r_margin = margin.min()
    if r_margin <= 0:
        return np.zeros_like(x)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = dists.min() / 2.0
    
    # Gradient of soft_min
    m = np.minimum(r_pair, r_margin)
    w_pair = np.exp(-beta * (r_pair - m)) / (np.exp(-beta * (r_pair - m)) + np.exp(-beta * (r_margin - m)))
    w_margin = 1.0 - w_pair
    
    # Gradient of r_pair with respect to points
    # r_pair = min_{i<j} |c_i - c_j| / 2
    # Find the minimum distance pair
    idx = np.unravel_index(np.argmin(dists), dists.shape)
    i, j = idx
    
    # Gradient of distance between i and j
    d_ij = pts[i] - pts[j]
    dist_ij = np.linalg.norm(d_ij)
    if dist_ij > 1e-12:
        grad_dist = d_ij / dist_ij
    else:
        grad_dist = np.zeros(3)
    
    # Gradient of r_pair
    grad_r_pair = np.zeros_like(x)
    grad_r_pair[i*3:(i+1)*3] = grad_dist / 2.0
    grad_r_pair[j*3:(j+1)*3] = -grad_dist / 2.0
    
    # Gradient of r_margin
    # r_margin = min_i min_k (0.5 - |c_ik|)
    # Find the point and coordinate that give the minimum margin
    min_idx = np.argmin(margin)
    min_pt = pts[min_idx]
    min_coord = np.argmax(np.abs(min_pt))
    
    grad_r_margin = np.zeros_like(x)
    sign = 1.0 if min_pt[min_coord] >= 0 else -1.0
    grad_r_margin[min_idx*3 + min_coord] = -sign
    
    # Combined gradient
    grad = -w_pair * grad_r_pair - w_margin * grad_r_margin
    return grad

def _grid_init(nx=5, ny=5, nz=4):
    """Generate a grid-based initialisation."""
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.375, 0.375, nz)
    points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return points

def _optimize_smooth(points, beta=50.0, maxiter=100):
    """Optimize using smooth soft-min surrogate with analytical gradients."""
    x0 = points.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    def neg_radius(x):
        return _neg_soft_radius(x, beta)
    
    def grad_neg_radius(x):
        return _grad_neg_soft_radius(x, beta)
    
    try:
        result = minimize(neg_radius, x0, jac=grad_neg_radius, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-7})
        if result.fun < neg_radius(x0):
            return result.x.reshape(N, 3)
        else:
            return points
    except Exception:
        return points

def _local_polish(points, maxiter=200):
    """Final polish with tighter tolerances and higher beta for sharper min."""
    for beta in [100.0, 200.0, 500.0]:
        x0 = points.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        def neg_radius(x):
            return _neg_soft_radius(x, beta)
        
        def grad_neg_radius(x):
            return _grad_neg_soft_radius(x, beta)
        
        try:
            result = minimize(neg_radius, x0, jac=grad_neg_radius, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
            if result.fun < neg_radius(x0):
                points = result.x.reshape(N, 3)
        except Exception:
            pass
    return points

def _hill_climbing(points, n_iter=500, step_decay=0.95):
    """Hill climbing with random perturbations."""
    current_points = points.copy()
    current_score = _score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    step_size = 0.05
    for i in range(n_iter):
        idx = random.randint(0, N - 1)
        delta = np.random.normal(0, step_size, 3)
        new_points = current_points.copy()
        new_points[idx] += delta
        new_points[idx] = np.clip(new_points[idx], -0.5, 0.5)
        
        new_score = _score(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_points = current_points.copy()
                best_score = current_score
        else:
            if random.random() < 0.1:
                current_points = new_points
                current_score = new_score
        
        step_size *= step_decay
    
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial solution from grid
    points = _grid_init()
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Phase 1: Local optimisation with smooth soft-min and analytical gradients
    if time.time() < deadline - 30:
        opt_points = _optimize_smooth(points, beta=50.0, maxiter=100)
        opt_score = _score(opt_points)
        if opt_score > best_score:
            best_points = opt_points
            best_score = opt_score
    
    # Phase 2: Multi-start with random perturbations and local optimisation
    random.seed(42)
    
    while time.time() < deadline - 20:
        if random.random() < 0.5:
            new_start = best_points + np.random.normal(0, 0.03, (N, 3))
        else:
            new_start = _grid_init() + np.random.normal(0, 0.02, (N, 3))
        
        new_start = np.clip(new_start, -0.5, 0.5)
        
        opt_points = _optimize_smooth(new_start, beta=50.0, maxiter=60)
        opt_score = _score(opt_points)
        
        if opt_score > best_score:
            best_points = opt_points
            best_score = opt_score
        
        # Occasional hill climbing
        if random.random() < 0.2 and time.time() < deadline - 25:
            hc_points = _hill_climbing(best_points, n_iter=100)
            hc_score = _score(hc_points)
            if hc_score > best_score:
                best_points = hc_points
                best_score = hc_score
    
    # Phase 3: High-precision polish with increasing beta
    if time.time() < deadline - 5:
        polished = _local_polish(best_points, maxiter=200)
        polished_score = _score(polished)
        if polished_score > best_score:
            best_points = polished
            best_score = polished_score
    
    return best_points

# EVOLVE-BLOCK-END
