"""
Solver for packing 50 equal spheres in a unit cube.
Strategy: Multi-start with Repulsion, Simulated Annealing, Smooth Polish, and Deterministic Local Search.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 50
DEADLINE_MARGIN = 5.0

def _compute_radius(c):
    """
    Compute the maximum radius for given centers c (N,3) in [-0.5, 0.5]^3.
    """
    # Pairwise distances
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Wall distances
    wall = 0.5 - np.abs(c)
    min_wall = np.min(wall)
    
    return min(min_pair / 2.0, min_wall)

def _pair_dist_matrix(c):
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(d, np.inf)
    return d

def _generate_fcc_init():
    """Generate FCC-like initialization."""
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                points.append([i, j, k])
                points.append([i+0.5, j+0.5, k])
                points.append([i+0.5, j, k+0.5])
                points.append([i, j+0.5, k+0.5])
    points = np.array(points)
    points -= np.mean(points, axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    pts = points[idx[:N]].copy()
    max_coord = np.max(np.abs(pts))
    if max_coord > 0:
        pts *= 0.45 / max_coord
    return pts

def _generate_grid_init():
    """Generate grid initialization."""
    n = 4
    grid = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)], dtype=float)
    grid = (grid / (n - 1) - 0.5) * 0.85
    rng = np.random.default_rng(123)
    grid += rng.uniform(-0.02, 0.02, size=grid.shape)
    return grid[:N].copy()

def _generate_random_init(seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 3))

def _repulsion_optimize(c, deadline, steps=2000):
    """
    Gradient descent on repulsion energy to remove overlaps and spread spheres.
    """
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best_r = _compute_radius(c)
    best_c = c.copy()
    
    for it in range(steps):
        if time.time() > deadline:
            break
        
        d = _pair_dist_matrix(c)
        # Force is proportional to 1/d^2
        inv_d2 = 1.0 / np.maximum(d, 1e-6)**2
        diff = c[:, None, :] - c[None, :, :]
        F = np.sum(diff * inv_d2[:, :, None], axis=1)
        
        # Wall force: push away from walls if close
        wall_force = np.zeros_like(c)
        for k in range(3):
            sgn = np.sign(c[:, k])
            sgn[sgn == 0] = 1.0
            # Only apply if close to wall
            dist_wall = 0.5 - np.abs(c[:, k])
            mask = dist_wall < 0.2
            wall_force[:, k] = sgn * np.where(mask, (0.5 - np.abs(c[:, k]))**2, 0)
            
        # Adaptive learning rate
        lr = 0.005 * (1.0 - it/steps)
        
        c_new = c + lr * (F + 0.5 * wall_force)
        c_new = np.clip(c_new, -0.49, 0.49)
        
        r_new = _compute_radius(c_new)
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
        # Always move if it improves radius, or if it reduces overlap (heuristic)
        # For now, just accept if radius improves or is same
        if r_new >= _compute_radius(c):
            c = c_new
        else:
            # Revert if it got worse
            pass
            
    return best_c, best_r

def _softmin_radius_and_grad(c, t):
    """
    Compute soft-min radius and its gradient.
    """
    d = _pair_dist_matrix(c)
    pair_vals = d / 2.0
    wall_vals = 0.5 - np.abs(c)
    
    vals = np.concatenate([pair_vals.ravel(), wall_vals.ravel()])
    vals = vals[np.isfinite(vals)]
    
    if len(vals) == 0:
        return 0.0, np.zeros(N*3)
        
    m = np.min(vals)
    exps = np.exp(-(vals - m) / t)
    sum_exps = np.sum(exps)
    r = m - t * np.log(sum_exps)
    
    grad = np.zeros(N*3)
    weights = exps / sum_exps
    
    pair_weights = weights[:N*N].reshape(N, N)
    wall_weights = weights[N*N:].reshape(N, 3)
    
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diff*diff, axis=-1))
    np.fill_diagonal(dists, 1.0)
    unit_diff = diff / dists[:, :, None]
    
    grad_pair = np.sum(pair_weights[:, :, None] * unit_diff, axis=1)
    
    signs = np.sign(c)
    signs[signs == 0] = 1.0
    grad_wall = -np.sum(wall_weights * signs, axis=1)
    
    grad = (grad_pair + grad_wall).flatten()
    
    return r, grad

def _smooth_obj(x, t):
    c = x.reshape(N, 3)
    r, _ = _softmin_radius_and_grad(c, t)
    pen = np.sum(np.maximum(0.0, np.abs(c) - 0.499)**2)
    return -r + 100.0 * pen

def _smooth_grad(x, t):
    c = x.reshape(N, 3)
    _, grad_r = _softmin_radius_and_grad(c, t)
    # Gradient of penalty
    pen_grad = np.zeros_like(x)
    mask = np.abs(c) > 0.499
    pen_grad[mask] = 200.0 * np.sign(c[mask])
    return -grad_r + pen_grad

def _smooth_polish(c, deadline, fine=False):
    """Polish using L-BFGS-B on smooth objective with analytical gradient."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best_r = _compute_radius(c)
    best_c = c.copy()
    x0 = c.flatten()
    bounds = [(-0.499, 0.499)] * (N * 3)
    
    if fine:
        temps = (0.005, 0.002, 0.001, 0.0005)
        maxiter = 2000
    else:
        temps = (0.05, 0.02, 0.01, 0.005, 0.002)
        maxiter = 1000
        
    for t in temps:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _smooth_obj, x0, method='L-BFGS-B', bounds=bounds,
                jac=_smooth_grad, args=(t,), options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            c_new = np.clip(res.x.reshape(N, 3), -0.499, 0.499)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
            x0 = c_new.flatten()
        except Exception:
            pass
    return best_c, best_r

def _local_search(c, deadline, iters=2000):
    """Active-set local search on binding constraints."""
    c = c.copy()
    c = np.clip(c, -0.499, 0.499)
    best_r = _compute_radius(c)
    best_c = c.copy()
    rng = np.random.default_rng(42)
    
    for it in range(iters):
        if time.time() > deadline:
            break
            
        d = _pair_dist_matrix(c)
        min_pair = np.min(d)
        wall = 0.5 - np.abs(c)
        min_wall = np.min(wall)
        
        r = min(min_pair / 2.0, min_wall)
        
        tol = 1e-5
        active_pairs = np.argwhere(d < min_pair + tol)
        active_pairs = [(i, j) for i, j in active_pairs if i < j]
        
        active_walls = np.argwhere(wall < min_wall + tol)
        
        if not active_pairs and not active_walls:
             break
             
        base_step = 0.002 * (best_r / 0.135)
        base_step = np.clip(base_step, 0.0005, 0.005)
        
        for i, j in active_pairs:
            diff = c[i] - c[j]
            norm = np.linalg.norm(diff)
            if norm > 1e-12:
                dir = diff / norm
                step = base_step * rng.uniform(0.5, 1.5)
                c[i] += step * dir
                c[j] -= step * dir
                
        for i, k in active_walls:
            step = base_step * rng.uniform(0.5, 1.5)
            if c[i, k] > 0:
                c[i, k] -= step
            else:
                c[i, k] += step
                
        c = np.clip(c, -0.499, 0.499)
        r_new = _compute_radius(c)
        if r_new > best_r:
            best_r = r_new
            best_c = c.copy()
            
    return best_c, best_r

def _metropolis_mc(c, deadline, steps=5000, temp0=0.01, temp_min=0.001):
    """Metropolis Monte Carlo with group perturbations."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best_r = _compute_radius(c)
    best_c = c.copy()
    rng = np.random.default_rng(77)
    
    for it in range(steps):
        if time.time() > deadline:
            break
            
        frac = it / max(steps - 1, 1)
        temp = temp0 * (temp_min / temp0) ** frac
        
        # Perturb 1 or 2 random spheres
        n_perturb = rng.choice([1, 2], p=[0.7, 0.3])
        indices = rng.choice(N, size=n_perturb, replace=False)
        
        c_new = c.copy()
        for i in indices:
            delta = rng.normal(0, 0.01, size=3)
            c_new[i] += delta
        c_new = np.clip(c_new, -0.49, 0.49)
        
        r_new = _compute_radius(c_new)
        r_cur = _compute_radius(c)
        
        if r_new >= r_cur:
            c = c_new
        else:
            prob = np.exp((r_new - r_cur) / max(temp, 1e-10))
            if rng.random() < prob:
                c = c_new
                
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
            
    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin
    
    best_pts = None
    best_r = 0.0
    
    # Try multiple initializations
    inits = [
        _generate_fcc_init(),
        _generate_grid_init(),
        _generate_random_init(1),
        _generate_random_init(2)
    ]
    
    # Allocate time per initialization
    # We will run a full pipeline for each if time permits, or iterate on the best.
    # Strategy: Run pipeline on each init, keep best.
    
    for init in inits:
        if time.time() > deadline:
            break
            
        c = init.copy()
        
        # 1. Repulsion
        if time.time() < deadline:
            c, r = _repulsion_optimize(c, deadline, steps=1000)
            if r > best_r:
                best_r = r
                best_pts = c.copy()
                
        # 2. Metropolis (to escape local minima)
        if time.time() < deadline:
            c, r = _metropolis_mc(c, deadline, steps=2000)
            if r > best_r:
                best_r = r
                best_pts = c.copy()
                
        # 3. Smooth Polish
        if time.time() < deadline:
            c, r = _smooth_polish(c, deadline, fine=False)
            if r > best_r:
                best_r = r
                best_pts = c.copy()
                
        # 4. Local Search
        if time.time() < deadline:
            c, r = _local_search(c, deadline, iters=1000)
            if r > best_r:
                best_r = r
                best_pts = c.copy()
                
        # 5. Fine Polish
        if time.time() < deadline:
            c, r = _smooth_polish(c, deadline, fine=True)
            if r > best_r:
                best_r = r
                best_pts = c.copy()
                
    # If time remains, do a final intensive polish on the best solution
    if best_pts is not None and time.time() < deadline:
        # One more round of local search + fine polish
        c, r = _local_search(best_pts, deadline, iters=2000)
        if r > best_r:
            best_r = r
            best_pts = c.copy()
            
        if time.time() < deadline:
            c, r = _smooth_polish(best_pts, deadline, fine=True)
            if r > best_r:
                best_r = r
                best_pts = c.copy()

    # Final validation
    if best_pts is None:
        return _generate_fcc_init()
        
    final_r = _compute_radius(best_pts)
    if final_r > best_r:
        best_r = final_r
        
    return best_pts
