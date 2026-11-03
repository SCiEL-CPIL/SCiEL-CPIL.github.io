"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

def _compute_radius_fast(pts):
    """Compute the packing radius for given points using KDTree."""
    if len(pts) < 2:
        return 0.0
    tree = cKDTree(pts)
    # Query for nearest neighbor (k=2 to get self + neighbor)
    d, _ = tree.query(pts, k=2)
    min_pair_dist = np.min(d[:, 1])  # distance to nearest other point
    min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair_dist / 2.0, min_boundary)

def _init_hex(spacing):
    """Generate hexagonal lattice points within unit disk."""
    k = int(1.0 / spacing) + 3
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0 - spacing / 2.0:
                points.append((x, y))
    return np.array(points) if points else np.empty((0, 2))

def _logsumexp_neg_dists(dists, temp):
    """Compute -logsumexp(-dists/temp) * temp, which approximates min(dists)."""
    # dists is a flat array of positive distances
    # We want soft_min = -temp * log(sum(exp(-dists/temp)))
    # To avoid underflow, shift by min
    min_d = np.min(dists)
    shifted = dists - min_d
    log_sum = np.log(np.sum(np.exp(-shifted / temp)))
    return min_d - temp * log_sum

def _soft_min_objective(x, temp):
    """
    Objective function: negative soft-min packing radius.
    x is a flat array of shape (2*N,).
    """
    pts = x.reshape(N, 2)
    
    # Pairwise distances
    # Use a more efficient way to get all pairwise distances?
    # For N=300, 300x300 matrix is 90k elements, manageable.
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
    # Boundary distances
    radii = np.hypot(pts[:, 0], pts[:, 1])
    boundary_dists = 1.0 - radii
    
    # Soft-min of pairwise distances / 2
    # We want min(dists) / 2
    # Let's compute soft_min(dists) first
    flat_dists = dists[np.triu_indices(N, k=1)]
    soft_min_pair = _logsumexp_neg_dists(flat_dists, temp)
    soft_min_pair_half = soft_min_pair / 2.0
    
    # Soft-min of boundary distances
    soft_min_bound = _logsumexp_neg_dists(boundary_dists, temp)
    
    # Overall soft-min radius
    # r = min(soft_min_pair_half, soft_min_bound)
    # Use soft-min again
    vals = np.array([soft_min_pair_half, soft_min_bound])
    min_val = np.min(vals)
    log_sum = np.log(np.sum(np.exp(-(vals - min_val) / temp)))
    soft_r = min_val - temp * log_sum
    
    # Penalties for invalid configurations (outside boundary)
    # If any boundary_dist < 0, we should penalize heavily
    # The soft_min will naturally be small/negative if boundary_dist is negative,
    # but let's add a quadratic penalty to ensure points stay inside during optimization
    outside_penalty = np.sum(np.maximum(0, radii - 1.0) ** 2) * 10000
    
    return -(soft_r - outside_penalty)

def _soft_min_gradient(x, temp, eps=1e-7):
    """Numerical gradient of the soft-min objective."""
    grad = np.zeros_like(x)
    base_obj = _soft_min_objective(x, temp)
    for i in range(len(x)):
        x[i] += eps
        obj_plus = _soft_min_objective(x, temp)
        x[i] -= 2 * eps
        obj_minus = _soft_min_objective(x, temp)
        x[i] += eps
        grad[i] = (obj_plus - obj_minus) / (2 * eps)
    return grad

def _force_optimize(pts, deadline, temp_init=0.05):
    """
    Optimize points using L-BFGS-B with soft-min objective.
    Returns optimized points.
    """
    x0 = pts.flatten()
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Annealing schedule
    temps = [temp_init, 0.03, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    
    best_x = x0.copy()
    best_obj = _soft_min_objective(x0, temp_init)
    
    for temp in temps:
        if time.time() > deadline - 5.0:
            break
        
        # Define objective and gradient with current temp
        def obj(x):
            return _soft_min_objective(x, temp)
        
        def grad(x):
            return _soft_min_gradient(x, temp)
        
        try:
            result = minimize(
                obj,
                best_x,
                method='L-BFGS-B',
                jac=grad,
                bounds=bounds,
                options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_x = result.x.copy()
        except Exception:
            break
    
    return best_x.reshape(N, 2)

def _polish(pts, deadline):
    """
    Final high-precision polish pass.
    Use a very small temperature and more iterations.
    """
    x0 = pts.flatten()
    bounds = [(-1.0, 1.0)] * (2 * N)
    temp = 1e-4
    
    def obj(x):
        return _soft_min_objective(x, temp)
    
    def grad(x):
        return _soft_min_gradient(x, temp)
    
    best_x = x0.copy()
    best_obj = obj(x0)
    
    try:
        result = minimize(
            obj,
            best_x,
            method='L-BFGS-B',
            jac=grad,
            bounds=bounds,
            options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result.fun < best_obj:
            best_x = result.x
    except Exception:
        pass
    
    return best_x.reshape(N, 2)

def _random_perturb(pts, sigma):
    """Add random Gaussian noise to points."""
    noise = np.random.normal(0, sigma, pts.shape)
    new_pts = pts + noise
    # Clip to unit disk to maintain validity roughly
    radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
    mask = radii > 1.0
    if np.any(mask):
        new_pts[mask] = new_pts[mask] / radii[mask, np.newaxis] * 0.99
    return new_pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 1. Initialize with hexagonal lattice
    best_pts = None
    best_r = 0.0
    
    # Try a range of spacings
    spacings = np.linspace(0.09, 0.13, 20)
    for spacing in spacings:
        if time.time() > deadline - 10.0:
            break
        pts = _init_hex(spacing)
        if len(pts) < N:
            continue
        # Select N closest to center
        dists_center = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(dists_center, kind='stable')[:N]
        candidate = pts[idx]
        r = _compute_radius_fast(candidate)
        if r > best_r:
            best_r = r
            best_pts = candidate.copy()
    
    if best_pts is None:
        # Fallback: uniform random
        best_pts = np.random.uniform(-0.9, 0.9, (N, 2))
        # Clip to disk
        radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
        mask = radii > 0.9
        best_pts[mask] = best_pts[mask] / radii[mask, np.newaxis] * 0.9
        best_r = _compute_radius_fast(best_pts)

    # 2. Force optimization (soft-min)
    if time.time() < deadline - 20.0:
        optimized_pts = _force_optimize(best_pts, deadline)
        r_opt = _compute_radius_fast(optimized_pts)
        if r_opt > best_r:
            best_r = r_opt
            best_pts = optimized_pts

    # 3. Random restarts with local optimization
    # If we have time, try a few random perturbations and optimize
    remaining_time = deadline - time.time()
    if remaining_time > 30.0:
        num_restarts = 3
        for _ in range(num_restarts):
            if time.time() > deadline - 10.0:
                break
            # Perturb current best
            perturbed = _random_perturb(best_pts, 0.05)
            # Quick local optimization with higher temp
            quick_opt = _force_optimize(perturbed, deadline, temp_init=0.1)
            r_quick = _compute_radius_fast(quick_opt)
            if r_quick > best_r:
                best_r = r_quick
                best_pts = quick_opt

    # 4. Final polish
    if time.time() < deadline - 2.0:
        polished_pts = _polish(best_pts, deadline)
        r_pol = _compute_radius_fast(polished_pts)
        if r_pol > best_r:
            best_r = r_pol
            best_pts = polished_pts

    # Ensure shape and validity
    if best_pts is None:
        best_pts = np.zeros((N, 2))
    
    if best_pts.shape != (N, 2):
        if best_pts.size >= 2 * N:
            best_pts = best_pts[:2 * N].reshape(N, 2)
        else:
            temp = np.zeros((N, 2))
            temp[:best_pts.size // 2, :] = best_pts.reshape(-1, 2) if best_pts.size > 0 else 0
            best_pts = temp

    return best_pts
# EVOLVE-BLOCK-END
