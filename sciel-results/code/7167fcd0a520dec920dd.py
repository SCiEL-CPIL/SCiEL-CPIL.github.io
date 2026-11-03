"""
Smooth soft-min optimization for packing 100 equal spheres in a unit cube.

Uses a log-sum-exp approximation of the minimum distance (soft-min) to create
a differentiable objective function. This allows the use of L-BFGS-B to
maximize the minimal distance (radius). An annealing schedule on the
sharpness parameter beta is used to escape local minima.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
HALF = 0.5
DEADLINE_MARGIN = 5.0  # Safety margin before the 150s kill


def _exact_radius(pts):
    """Compute the exact maximum common radius for a set of centers."""
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, np.inf)
    min_pair_dist = np.sqrt(np.min(d2))
    
    # Distance to walls
    # pts are in [-0.5, 0.5]
    wall_dists = HALF - np.max(np.abs(pts), axis=1)
    min_wall_dist = np.min(wall_dists)
    
    # Radius is half the minimum of (pair dist, 2 * wall dist)
    # Wait, the constraint is |c_i - c_j| >= 2r and |c_ik| <= 0.5 - r.
    # So 2r <= |c_i - c_j| and 2r <= 1 - 2|c_ik| ? No.
    # Center is at c. Sphere radius r.
    # Distance to wall is 0.5 - |c| (if c>0) or 0.5 + c (if c<0). i.e., 0.5 - |c|.
    # Constraint: r <= 0.5 - |c_ik|.
    # So 2r <= 1 - 2|c_ik| is incorrect.
    # The radius r must satisfy r <= 0.5 - max_k |c_ik|.
    # And r <= 0.5 * min_{i<j} |c_i - c_j|.
    
    r_pair = 0.5 * min_pair_dist
    r_wall = min_wall_dist
    
    r = min(r_pair, r_wall)
    if not np.isfinite(r) or r < 0:
        return 0.0
    return float(r)


def _energy_and_grad(x, r):
    """
    Compute energy and gradient for a given radius r.
    Energy is sum of squared violations for pair overlaps and wall violations.
    Uses smooth approximation for max(0, v).
    """
    pts = x.reshape(N, 3)
    sigma = 1e-3 # Smoothing parameter
    
    # 1. Pairwise
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, np.inf)
    dists = np.sqrt(d2)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    # Violations: v = 2r - d
    v_pairs = 2.0 * r - pair_dists
    
    # Smooth max of v
    sqrt_term_pairs = np.sqrt(v_pairs**2 + sigma**2)
    f_pairs = 0.5 * (v_pairs + sqrt_term_pairs)
    fp_pairs = 0.5 * (1.0 + v_pairs / sqrt_term_pairs)
    
    # Gradient of v_pairs w.r.t x
    w_pairs = 2.0 * f_pairs * fp_pairs
    
    grad_x = np.zeros_like(pts)
    
    # Vectorized accumulation
    W_mat = np.zeros((N, N))
    W_mat[i_idx, j_idx] = w_pairs
    W_mat[j_idx, i_idx] = w_pairs
    
    dists_safe = np.where(dists < 1e-10, 1e-10, dists)
    u = diff / dists_safe[:, :, None]
    
    grad_pair_part = -np.einsum('ij,ijk->ik', W_mat, u)
    
    # 2. Wall
    abs_pts = np.abs(pts)
    max_coord_idx = np.argmax(abs_pts, axis=1)
    one_hot = np.zeros_like(pts)
    one_hot[np.arange(N), max_coord_idx] = 1.0
    
    max_abs = np.max(abs_pts, axis=1)
    v_walls = r - HALF + max_abs
    
    sqrt_term_walls = np.sqrt(v_walls**2 + sigma**2)
    f_walls = 0.5 * (v_walls + sqrt_term_walls)
    fp_walls = 0.5 * (1.0 + v_walls / sqrt_term_walls)
    
    w_walls = 2.0 * f_walls * fp_walls
    
    signs = np.sign(pts)
    grad_wall_part = w_walls[:, None] * signs * one_hot
    
    total_grad = grad_pair_part + grad_wall_part
    
    E_pairs = np.sum(f_pairs**2)
    E_walls = np.sum(f_walls**2)
    total_E = E_pairs + E_walls
    
    return total_E, total_grad.ravel()


def _optimize_radius(init_pts, deadline):
    """
    Iteratively increase radius r and minimize energy.
    """
    current_pts = init_pts.copy()
    best_pts = current_pts.copy()
    best_r = _exact_radius(best_pts)
    
    # Start from current radius + small epsilon
    r = best_r + 1e-4
    
    step = 0.0005
    max_iters = 300
    min_energy_threshold = 1e-6
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        def fun(x):
            e, g = _energy_and_grad(x, r)
            return e
            
        def jac(x):
            e, g = _energy_and_grad(x, r)
            return g
            
        bounds = [(-HALF, HALF)] * (N * 3)
        
        try:
            res = minimize(
                fun, 
                current_pts.ravel(), 
                jac=jac, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            
            cand_pts = res.x.reshape(N, 3)
            cand_pts = np.clip(cand_pts, -HALF, HALF)
            
            e, _ = _energy_and_grad(cand_pts.ravel(), r)
            
            if e < min_energy_threshold:
                current_pts = cand_pts
                exact_r = _exact_radius(current_pts)
                if exact_r > best_r:
                    best_r = exact_r
                    best_pts = current_pts.copy()
                
                r += step
            else:
                break
                
        except Exception:
            break
            
    return best_pts, best_r


def _init_grid():
    """Initialize with a grid pattern."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].astype(np.float64)


def _init_random(rng):
    """Initialize with random points."""
    return (rng.random((N, 3)) - 0.5) * 0.9


def construct():
    t0 = time.time()
    deadline = 150.0 - DEADLINE_MARGIN
    
    rng = np.random.default_rng(42)
    
    # Global best
    best_pts = _init_grid()
    best_r = _exact_radius(best_pts)
    
    # Initializations: Grid + Random
    inits = [best_pts]
    for _ in range(5):
        inits.append(_init_random(rng))
    
    for init in inits:
        if time.time() > deadline:
            break
        
        p, r = _optimize_radius(init, deadline)
        if r > best_r:
            best_r = r
            best_pts = p
    
    # Use spare time for more random restarts
    guard = 0
    while time.time() < deadline - 2.0 and guard < 10:
        guard += 1
        start = _init_random(rng)
        p, r = _optimize_radius(start, deadline)
        if r > best_r:
            best_r = r
            best_pts = p
    
    # Final safety check
    out = np.clip(np.asarray(best_pts, dtype=np.float64), -HALF, HALF)
    if not np.all(np.isfinite(out)):
        out = _init_grid()
        
    return out
