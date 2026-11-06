"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _compute_radius(x):
    """Compute the maximum common radius for given centres."""
    x = x.reshape(-1, 3)
    d = cdist(x, x)
    np.fill_diagonal(d, np.inf)
    min_dist = np.min(d)
    boundary_dist = 0.5 - np.max(np.abs(x), axis=1)
    min_boundary = np.min(boundary_dist)
    return min(min_dist, min_boundary) / 2.0

def _smooth_objective_and_grad(x, alpha):
    """
    Smooth objective using log-sum-exp for the minimum distance problem.
    Returns (objective, gradient).
    """
    x = x.reshape(-1, 3)
    n = x.shape[0]
    
    # Compute pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(d, np.inf)
    
    # Boundary distances
    boundary = 0.5 - np.abs(x)
    boundary_min_idx = np.argmin(boundary, axis=1)
    boundary_min = boundary[np.arange(n), boundary_min_idx]
    
    # Collect all distances
    iu = np.triu_indices(n, k=1)
    pair_dists = d[iu]
    
    all_dists = np.concatenate([pair_dists, boundary_min])
    
    # Compute weights for soft-min
    neg_alpha_d = -alpha * all_dists
    neg_alpha_d -= np.max(neg_alpha_d)
    weights = np.exp(neg_alpha_d)
    weights /= np.sum(weights)
    
    # Objective value
    obj = -np.sum(weights * all_dists)
    
    # Gradient
    grad = np.zeros_like(x)
    
    # Pairwise contributions
    pair_i, pair_j = iu
    pair_w = weights[:len(pair_dists)]
    
    for k in range(len(pair_dists)):
        i, j = pair_i[k], pair_j[k]
        w = pair_w[k]
        if d[i, j] < 1e-10:
            continue
        diff_ij = x[i] - x[j]
        grad[i] += -w * diff_ij / d[i, j]
        grad[j] += -w * (-diff_ij) / d[i, j]
    
    # Boundary contributions
    boundary_w = weights[len(pair_dists):]
    for k in range(n):
        w = boundary_w[k]
        i = k
        axis = boundary_min_idx[k]
        sign_val = np.sign(x[i, axis])
        if sign_val == 0:
            sign_val = 1.0
        grad[i, axis] += -w * (-sign_val)
    
    return obj, grad

def _anneal_optimize(x0, alphas, deadline, maxiter_per_alpha=300):
    """
    Run annealed optimization with increasing alpha.
    """
    current_x = x0.copy()
    best_x = x0.copy()
    best_r = _compute_radius(x0)
    
    for alpha in alphas:
        if time.time() > deadline - 5.0:
            break
        
        try:
            result = minimize(
                lambda x: _smooth_objective_and_grad(x, alpha)[0],
                current_x,
                jac=lambda x: _smooth_objective_and_grad(x, alpha)[1],
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * N * 3,
                options={'maxiter': maxiter_per_alpha, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < 1e10:
                candidate = result.x.reshape(-1, 3)
                r_candidate = _compute_radius(candidate)
                if r_candidate > best_r:
                    best_r = r_candidate
                    best_x = candidate.copy()
                current_x = candidate.copy()
        except Exception:
            pass
    
    return best_x, best_r

def _local_search(x0, deadline, n_restarts=5):
    """
    Perform local search with random perturbations to escape local minima.
    """
    best_x = x0.copy()
    best_r = _compute_radius(x0)
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb the current best solution
        noise_scale = 0.02 * (1.0 - restart / n_restarts)
        perturbed = best_x + rng.normal(0, noise_scale, size=best_x.shape)
        perturbed = np.clip(perturbed, -0.45, 0.45)
        
        # Try to optimize from this perturbed state
        alphas = [50, 100, 200, 500]
        try:
            result = minimize(
                lambda x: _smooth_objective_and_grad(x, alphas[-1])[0],
                perturbed,
                jac=lambda x: _smooth_objective_and_grad(x, alphas[-1])[1],
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * N * 3,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < 1e10:
                candidate = result.x.reshape(-1, 3)
                r_candidate = _compute_radius(candidate)
                if r_candidate > best_r:
                    best_r = r_candidate
                    best_x = candidate.copy()
        except Exception:
            pass
    
    return best_x, best_r

def construct():
    """
    Find the optimal packing of 100 equal spheres in a unit cube.
    Uses smooth log-sum-exp objective with annealing, then local search.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s hard limit
    
    # Initialize with a simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    x0 = grid[:N].copy()
    
    # Phase 1: Annealed optimization from grid
    alphas = [10, 20, 50, 100, 200, 500, 1000, 2000]
    best_x, best_r = _anneal_optimize(x0, alphas, deadline, maxiter_per_alpha=400)
    
    # Phase 2: Local search with perturbations
    if time.time() < deadline - 15.0:
        best_x, best_r = _local_search(best_x, deadline, n_restarts=8)
    
    # Phase 3: Final polish with high precision
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                lambda x: _smooth_objective_and_grad(x, 5000)[0],
                best_x,
                jac=lambda x: _smooth_objective_and_grad(x, 5000)[1],
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * N * 3,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < 1e10:
                candidate = result.x.reshape(-1, 3)
                r_candidate = _compute_radius(candidate)
                if r_candidate > best_r:
                    best_r = r_candidate
                    best_x = candidate.copy()
        except Exception:
            pass
    
    # Ensure validity: clip to cube
    best_x = np.clip(best_x, -0.5, 0.5)
    
    # Final validation
    final_r = _compute_radius(best_x)
    if not np.isfinite(final_r) or final_r <= 0:
        return grid[:N]
    
    return best_x

# EVOLVE-BLOCK-END
