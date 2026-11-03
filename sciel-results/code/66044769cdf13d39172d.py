"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
HALF = 0.5

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    n = len(centers)
    if n == 0:
        return 0.0
    
    # Minimum distance to boundary
    min_boundary = HALF - np.abs(centers).max()
    if min_boundary <= 0:
        return 0.0
    
    # Minimum pairwise distance / 2
    if n < 2:
        return min_boundary
    
    # Compute all pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    # Set diagonal to inf to ignore self-distances
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min() / 2.0
    
    return min(min_boundary, min_pair)

def _soft_min_obj_grad(x, beta):
    """
    Objective function: minimize -soft_min(effective_radii).
    effective_radii = [pair_dists/2, wall_dists]
    soft_min(z) = -1/beta * log(sum(exp(-beta * z)))
    We want to maximize soft_min, so we minimize -soft_min.
    
    Gradient of -soft_min(z) wrt z is - (exp(-beta*z) / sum(exp(-beta*z)))
    """
    c = x.reshape(N, 3)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Get upper triangle indices to avoid double counting in the sum
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Wall distances
    d_walls = HALF - np.abs(c)
    wall_dists = d_walls.flatten()
    
    # All effective radii
    # Note: pair dist d corresponds to radius d/2. Wall dist d corresponds to radius d.
    eff_radii = np.concatenate([pair_dists / 2.0, wall_dists])
    
    # Clip to avoid underflow/overflow in exp
    # If any radius is very small, exp(-beta * r) is huge.
    # We can shift by max to stabilize, but log-sum-exp is standard.
    # Let's use a stable log-sum-exp.
    m = np.max(eff_radii)
    if not np.isfinite(m):
        return 0.0, np.zeros_like(x)
        
    # z = -beta * eff_radii
    # log_sum_exp(z) = m_z + log(sum(exp(z - m_z)))
    # Here we want log(sum(exp(-beta * eff_radii)))
    # Let y = -beta * eff_radii
    y = -beta * eff_radii
    y_max = np.max(y)
    if not np.isfinite(y_max):
        return 0.0, np.zeros_like(x)
    
    sum_exp = np.sum(np.exp(y - y_max))
    log_sum_exp = y_max + np.log(sum_exp)
    
    # soft_min = -1/beta * log_sum_exp
    # obj = -soft_min = 1/beta * log_sum_exp
    obj = log_sum_exp / beta
    
    # Gradient of obj wrt eff_radii
    # d(obj)/d(eff_radii) = 1/beta * d(log_sum_exp)/d(eff_radii)
    # d(log_sum_exp)/d(eff_radii_k) = -beta * exp(-beta * eff_radii_k) / sum(exp(-beta * eff_radii))
    # So d(obj)/d(eff_radii_k) = - exp(-beta * eff_radii_k) / sum(exp(-beta * eff_radii))
    
    # We need gradient wrt x (centers).
    # Let w_k = exp(y_k - y_max) / sum_exp  (normalized weights)
    # Then d(obj)/d(eff_radii_k) = -w_k
    
    weights = np.exp(y - y_max) / sum_exp
    
    # Now backpropagate to x
    # eff_radii consists of pair_dists/2 and wall_dists
    # grad_x = sum_k (d_obj/d_eff_r_k) * (d_eff_r_k / d_x)
    
    grad_x = np.zeros_like(x)
    
    # 1. Wall contributions
    # eff_wall_j = HALF - |c_j|
    # d(eff_wall_j)/d(c_j) = -sign(c_j)
    # d_obj/d(c_j) += sum_{j in walls} (-w_wall_j) * (-sign(c_j))
    # Note: wall_dists are flattened from (N,3).
    wall_weights = weights[-3*N:]
    c_signs = np.sign(c)
    # If c is 0, sign is 0, gradient is 0. This is a non-differentiable point, but measure zero.
    grad_x_wall = (wall_weights.reshape(N, 3) * c_signs)
    grad_x += grad_x_wall.flatten()
    
    # 2. Pair contributions
    # eff_pair_ij = ||c_i - c_j|| / 2
    # d(eff_pair_ij)/d(c_i) = (c_i - c_j) / (2 * ||c_i - c_j||)
    # d(eff_pair_ij)/d(c_j) = (c_j - c_i) / (2 * ||c_i - c_j||)
    # d_obj/d(c_i) += sum_j (-w_pair_ij) * (c_i - c_j) / (2 * dist_ij)
    
    pair_weights = weights[:len(pair_dists)]
    
    # We need to accumulate gradients for each i and j.
    # Using a loop is too slow for N=100 (4950 pairs).
    # We can use matrix operations.
    
    # Reconstruct full matrix of pair weights (upper triangle only filled)
    W = np.zeros((N, N))
    W[iu] = pair_weights
    
    # dists is (N, N). We need (c_i - c_j) / (2 * dists_ij)
    # diff = c[:, None, :] - c[None, :, :]
    # unit_vecs = diff / dists[:, :, None]  (handle dists=0)
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    # grad_i = sum_j (-W_ij) * unit_vecs_ij / 2
    # grad_j = sum_i (-W_ji) * unit_vecs_ji / 2  (note W is symmetric, unit_vecs_ji = -unit_vecs_ij)
    # Actually, let's just compute grad_i and add to grad_x, and grad_j adds to grad_x.
    # Since W is symmetric and unit_vecs is antisymmetric, the total contribution is symmetric.
    
    # grad_x_pair[i] = sum_j (-W[i,j]) * unit_vecs[i,j] / 2
    grad_x_pair = -0.5 * np.sum(W[:, :, np.newaxis] * unit_vecs, axis=1)
    
    grad_x += grad_x_pair.flatten()
    
    return obj, grad_x

def _hard_obj_grad(x, target_r):
    """
    Objective function: minimize sum of squared violations.
    Violations: max(0, 2r - d_ij) for pairs, max(0, r - d_wall) for walls.
    """
    c = x.reshape(N, 3)
    
    # Pairwise Part
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1.0)
    
    active = dists < 2.0 * target_r
    active[np.diag_indices_from(active)] = False
    
    violations = np.where(active, 2.0 * target_r - dists, 0.0)
    
    energy_pair = 0.5 * np.sum(violations**2)
    
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    grad_pair = -np.sum(violations[:, :, np.newaxis] * unit_vecs, axis=1)
    
    # Wall Part
    wall_dists = 0.5 - np.abs(c)
    wall_violations = np.maximum(0.0, target_r - wall_dists)
    energy_wall = np.sum(wall_violations**2)
    
    signs = np.sign(c)
    grad_wall = 2.0 * wall_violations * signs
    
    total_energy = energy_pair + energy_wall
    total_grad = (grad_pair + grad_wall).flatten()
    
    return total_energy, total_grad

def _optimize(x0, obj_func, grad_func, max_iter, time_limit):
    """
    Run BFGS optimization with a time limit.
    obj_func and grad_func should return (value, grad).
    """
    t_start = time.time()
    timeout = [False]
    
    def objective(x):
        if time.time() - t_start > time_limit:
            timeout[0] = True
            raise TimeoutError()
        return obj_func(x)
    
    try:
        result = minimize(
            objective,
            x0,
            method='BFGS',
            jac=True,
            options={'maxiter': max_iter, 'gtol': 1e-8, 'disp': False}
        )
        return result.x, result.success
    except TimeoutError:
        # If timeout, we don't have the intermediate x easily.
        # scipy's minimize doesn't support callbacks to return intermediate x in all versions easily.
        # We will just return x0 if timeout occurs during the first few iterations,
        # or rely on the fact that BFGS is fast.
        # A better way is to check time in the outer loop and use a smaller max_iter.
        return x0, False
    except Exception:
        return x0, False

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # Safety margin under 150s
    
    # Initial Grid: 5x5x5 = 125 points, take 100
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    best_pts = grid[:N].copy()
    best_r = _compute_radius(best_pts)
    
    rng = np.random.RandomState(42)
    
    # Strategy:
    # 1. Soft-min optimization to find a good basin.
    # 2. Hard-constraint optimization to polish.
    # Repeat with different noise/seeds.
    
    # Beta schedule for soft-min: start low (smooth), increase (sharp)
    # We will do a few iterations of soft-min with increasing beta.
    
    current_pts = best_pts.copy()
    
    # Phase 1: Soft Min Annealing
    # Try a few different beta values
    betas = [10.0, 20.0, 50.0, 100.0, 200.0]
    
    for beta in betas:
        if time.time() - t0 > DEADLINE - 10.0:
            break
            
        # Add small noise to break symmetry
        noise = rng.normal(0, 0.01, size=(N, 3))
        init_pts = np.clip(current_pts + noise, -0.49, 0.49)
        
        remaining = DEADLINE - (time.time() - t0) - 5.0
        if remaining < 2:
            break
            
        # Optimize soft-min
        opt_pts, success = _optimize(
            init_pts.flatten(),
            lambda x: _soft_min_obj_grad(x, beta),
            None, # jac is provided in objective return
            max_iter=200,
            time_limit=remaining
        )
        
        if np.all(np.isfinite(opt_pts)):
            opt_pts = opt_pts.reshape(N, 3)
            opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
            r_opt = _compute_radius(opt_pts)
            
            if r_opt > best_r:
                best_r = r_opt
                best_pts = opt_pts.copy()
                current_pts = opt_pts.copy()
    
    # Phase 2: Hard Constraint Polish
    # Use the best found so far as initial guess for hard optimization.
    # We will try to push the radius slightly higher.
    
    # Estimate a target radius slightly above current best
    # We can just use the current best_r as a lower bound and let the optimizer
    # try to reduce violations for a slightly larger r, or simply minimize
    # the energy for the current r to ensure it's valid, then try r * 1.01.
    
    # Let's do a few polish steps.
    for i in range(10):
        if time.time() - t0 > DEADLINE - 5.0:
            break
            
        # Try to optimize for a target radius slightly higher than current best
        # If it fails (violations remain), the resulting points might still be better
        # if we re-evaluate the actual radius.
        
        # Actually, minimizing squared violations for a fixed r is a good way to
        # find a valid configuration with radius >= r.
        # If we set r = best_r * 1.001, and the energy goes to 0, we have a valid
        # config with radius >= best_r * 1.001.
        
        target_r = best_r * 1.005
        
        noise = rng.normal(0, 0.005, size=(N, 3))
        init_pts = np.clip(best_pts + noise, -0.49, 0.49)
        
        remaining = DEADLINE - (time.time() - t0) - 2.0
        if remaining < 1:
            break
            
        opt_pts, success = _optimize(
            init_pts.flatten(),
            lambda x: _hard_obj_grad(x, target_r),
            None,
            max_iter=500,
            time_limit=remaining
        )
        
        if np.all(np.isfinite(opt_pts)):
            opt_pts = opt_pts.reshape(N, 3)
            opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
            r_opt = _compute_radius(opt_pts)
            
            if r_opt > best_r:
                best_r = r_opt
                best_pts = opt_pts.copy()
    
    # Final Check and Return
    # Ensure the returned points are valid
    # If best_r is very small or points are invalid, fall back to grid
    final_r = _compute_radius(best_pts)
    if final_r < 0.1: # Grid gives 0.1
        best_pts = grid[:N].copy()
        
    return best_pts
# EVOLVE-BLOCK-END
