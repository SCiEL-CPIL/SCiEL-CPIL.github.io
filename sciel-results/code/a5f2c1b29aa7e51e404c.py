"""Improved packing: start from near-optimal grid + local refinement with soft-min objective."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 115.0  # safety margin before 120s limit


def _radius(c):
    """Compute the radius for a given set of centers."""
    if c.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check bounds
    if np.any(c < -0.5) or np.any(c > 0.5):
        return 0.0
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    # Distance to walls
    wall_dist = np.min(np.minimum(0.5 - c, c + 0.5), axis=1)
    min_wall = np.min(wall_dist)
    return min(min_dist, 2.0 * min_wall) / 2.0


def _soft_min(x, beta=100.0):
    """Soft minimum using log-sum-exp trick."""
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    # Use the fact that softmin(x) = -softmax(-x) weighted sum
    # For numerical stability, shift by min
    x_shifted = x - np.min(x)
    # softmin ≈ min - (1/beta) * log(sum(exp(-beta*(x-min))))
    # But for large beta, this approaches min
    # Alternative: use a simple smooth approximation
    # softmin(x) ≈ (1/beta) * log(sum(exp(-beta*x)))
    # But this can overflow, so shift
    x_neg = -beta * x
    x_neg -= np.max(x_neg)
    log_sum = np.log(np.sum(np.exp(x_neg)))
    return -(log_sum) / beta


def _objective(params):
    """Objective to minimize: negative of the radius (soft version for smoothness)."""
    c = params.reshape(N, 3)
    # Check bounds - add penalty if out of bounds
    if np.any(c < -0.5 + 1e-12) or np.any(c > 0.5 - 1e-12):
        return 10.0  # Large penalty
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    wall_dist = np.min(np.minimum(0.5 - c, c + 0.5), axis=1)
    
    # We want to maximize min(min_pairwise_dist, 2*min_wall_dist)
    # Equivalent to minimizing -min(...)
    # Use soft min for smoothness
    beta = 50.0
    # Combine all constraints: pairwise distances and wall distances (scaled by 2)
    # The radius is min(min_pairwise/2, min_wall)
    # So we want to maximize min(min_pairwise/2, min_wall)
    # = min over all pairs of (dist_ij/2) and over all i,k of (0.5 - |c_ik|)
    
    # Let's work with the "constraint values" that should be >= 2r
    # For pair (i,j): dist_ij >= 2r
    # For wall: 0.5 - |c_ik| >= r
    # So r <= min(dist_ij/2 for all pairs, 0.5 - |c_ik| for all i,k)
    
    # We want to maximize r, so minimize -r
    # r = min(min_pairwise_dist/2, min_wall_dist)
    
    # Use soft min on the set {dist_ij/2 for all pairs, wall_dist_i for all i}
    # But this is a huge set. Let's just use the min directly for the objective
    # and rely on L-BFGS-B's finite differences
    
    min_pair = np.min(dists) / 2.0
    min_wall = np.min(wall_dist)
    
    r = min(min_pair, min_wall)
    
    return -r


def _objective_smooth(params, beta=20.0):
    """Smooth objective using soft-min."""
    c = params.reshape(N, 3)
    if np.any(c < -0.5 + 1e-12) or np.any(c > 0.5 - 1e-12):
        return 10.0
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    wall_dist = np.min(np.minimum(0.5 - c, c + 0.5), axis=1)
    
    # Soft min of (dists/2, wall_dist)
    # Only consider off-diagonal dists
    mask = ~np.eye(N, dtype=bool)
    pair_vals = dists[mask] / 2.0
    wall_vals = wall_dist
    
    # Combine
    all_vals = np.concatenate([pair_vals, wall_vals])
    
    # Soft min
    x_neg = -beta * all_vals
    x_neg -= np.max(x_neg)
    log_sum = np.log(np.sum(np.exp(x_neg)))
    soft_min_val = -(log_sum) / beta
    
    return -soft_min_val


def _make_initial_grid():
    """Create initial grid configuration."""
    m = 5  # 5x5x4 = 100
    # 5x5x4 grid
    x = np.linspace(-0.5 + 0.1, 0.5 - 0.1, 5)
    y = np.linspace(-0.5 + 0.1, 0.5 - 0.1, 5)
    z = np.linspace(-0.5 + 0.125, 0.5 - 0.125, 4)
    
    centers = np.array([(xi, yi, zi) for zi in z for yi in y for xi in x])
    return centers


def construct():
    start_time = time.time()
    
    # Initial grid
    c0 = _make_initial_grid()
    params0 = c0.flatten()
    
    best_params = params0.copy()
    best_radius = _radius(c0)
    
    # Try local optimization with L-BFGS-B
    # Use the smooth objective for better gradient approximation
    beta_values = [10.0, 20.0, 50.0]
    
    for beta in beta_values:
        if time.time() - start_time > DEADLINE:
            break
        
        def obj(p):
            return _objective_smooth(p, beta=beta)
        
        try:
            res = minimize(
                obj,
                params0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxls': 20, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-8},
                bounds=[(-0.5, 0.5)] * (N * 3)
            )
            if res.success or res.fun < obj(best_params):
                c_res = res.x.reshape(N, 3)
                r_res = _radius(c_res)
                if r_res > best_radius:
                    best_radius = r_res
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # Final polish with exact objective
    if time.time() - start_time < DEADLINE:
        def obj_exact(p):
            return _objective(p)
        
        try:
            res = minimize(
                obj_exact,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxls': 10, 'maxfun': 500, 'ftol': 1e-14, 'gtol': 1e-10},
                bounds=[(-0.5, 0.5)] * (N * 3)
            )
            c_res = res.x.reshape(N, 3)
            r_res = _radius(c_res)
            if r_res > best_radius:
                best_radius = r_res
                best_params = res.x.copy()
        except Exception:
            pass
    
    return best_params.reshape(N, 3)
# EVOLVE-BLOCK-END
