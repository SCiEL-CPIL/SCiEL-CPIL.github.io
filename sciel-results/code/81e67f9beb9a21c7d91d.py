"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import cKDTree

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import cKDTree

N = 50

def _compute_radius(centers):
    """Compute the true maximum radius for a given configuration."""
    n = len(centers)
    # Wall distances
    wall_dists = np.min(np.column_stack((
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    )), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_pair, min_wall)

def _soft_min_obj_and_grad(x, beta=100.0):
    """
    Soft-min objective and gradient.
    We want to maximize the minimum of all constraints (pair distances and wall distances).
    soft_min(v) = m - (1/beta) * log(sum(exp(-beta*(v_i - m))))
    We minimize -soft_min(v).
    
    Returns (objective_value, gradient)
    """
    centers = x.reshape(N, 2)
    
    # Wall distances
    wall_dists = np.column_stack((
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ))
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    all_vals = np.concatenate([pair_dists, wall_dists.ravel()])
    
    # Soft min computation
    min_val = np.min(all_vals)
    shifted = -beta * (all_vals - min_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    if sum_exp == 0:
        sm = min_val
        weights = np.ones_like(all_vals) / len(all_vals)
    else:
        sm = min_val - (1.0 / beta) * np.log(sum_exp)
        weights = exp_shifted / sum_exp
    
    val = -sm
    
    # Gradient computation
    # d(soft_min)/dx = -sum_i weights_i * d(v_i)/dx
    # So d(-soft_min)/dx = sum_i weights_i * d(v_i)/dx
    
    grad = np.zeros_like(centers)
    
    # Pair distance gradients
    # For each pair (i, j) with weight w:
    # d(d_ij)/dx_i = (x_i - x_j) / d_ij
    # d(d_ij)/dx_j = (x_j - x_i) / d_ij
    pair_weights = weights[:len(pair_dists)]
    
    for k in range(len(iu[0])):
        i, j = iu[0][k], iu[1][k]
        w = pair_weights[k]
        d_ij = pair_dists[k]
        if d_ij < 1e-15:
            continue
        vec = (centers[i] - centers[j]) / d_ij
        grad[i] += w * vec
        grad[j] -= w * vec
    
    # Wall distance gradients
    # w_x = 0.5 - |x|, dw/dx = -sign(x)
    # d(-soft_min)/dx includes sum of weights * dw/dx
    # So we need -sum(weights * dw/dx) for the final gradient of -soft_min
    # Wait, let me re-derive.
    
    # f(x) = -soft_min(v(x))
    # soft_min(v) = m - (1/beta)*log(sum(exp(-beta*(v_i - m))))
    # d(soft_min)/dx = -(1/beta) * [sum(exp(-beta*(v_i-m))) * (-beta) * d(v_i)/dx] / sum(exp(-beta*(v_i-m)))
    #                = sum(exp(-beta*(v_i-m))) * d(v_i)/dx / sum(exp(-beta*(v_i-m)))
    #                = sum(weights_i * d(v_i)/dx)
    
    # So d(soft_min)/dx = sum_i weights_i * d(v_i)/dx
    # d(f)/dx = -sum_i weights_i * d(v_i)/dx
    
    # For wall distances:
    # v_i = 0.5 - |x_i|
    # d(v_i)/dx_i = -sign(x_i)
    # So contribution to d(soft_min)/dx from wall i is weights[i] * (-sign(x_i))
    # And d(f)/dx gets -weights[i] * (-sign(x_i)) = weights[i] * sign(x_i)
    
    wall_weights = weights[len(pair_dists):]
    sign_x = np.sign(centers[:, 0])
    sign_y = np.sign(centers[:, 1])
    
    # Wall x distances are at indices N..2N-1 in all_vals
    # Wall y distances are at indices 2N..3N-1 in all_vals
    # Wait, wall_dists has shape (N, 2), so ravel gives [w_0x, w_0y, w_1x, w_1y, ...]
    # So wall_weights[0::2] are for x, wall_weights[1::2] are for y
    
    grad[:, 0] += wall_weights[0::2] * sign_x
    grad[:, 1] += wall_weights[1::2] * sign_y
    
    return val, grad.flatten()

def _violation_objective(x, r):
    """Minimize violation of radius r."""
    centers = x.reshape(N, 2)
    
    # Wall distances
    wall_dists = np.column_stack((
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ))
    wall_viol = np.maximum(0, r - wall_dists)
    wall_cost = np.sum(wall_viol**2)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    pair_viol = np.maximum(0, 2*r - pair_dists)
    pair_cost = np.sum(pair_viol**2)
    
    return pair_cost + wall_cost

def _violation_grad(x, r):
    """Gradient of the violation objective."""
    centers = x.reshape(N, 2)
    grad = np.zeros_like(centers)
    
    # Wall gradient
    wall_dists = np.column_stack((
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ))
    wall_viol = np.maximum(0, r - wall_dists)
    
    sign_x = np.sign(centers[:, 0])
    sign_y = np.sign(centers[:, 1])
    
    grad[:, 0] += 2 * wall_viol[:, 0] * sign_x
    grad[:, 1] += 2 * wall_viol[:, 1] * sign_y
    
    # Pair gradient
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1.0)
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    pair_viol = np.maximum(0, 2*r - pair_dists)
    
    for k in range(len(iu[0])):
        i, j = iu[0][k], iu[1][k]
        viol = pair_viol[k]
        if viol == 0:
            continue
        d_ij = pair_dists[k]
        vec = centers[i] - centers[j]
        grad_vec = vec / d_ij
        
        grad[i] -= 2 * viol * grad_vec
        grad[j] += 2 * viol * grad_vec
    
    return grad.flatten()

def _construct_initial_grid():
    """Create a 7x7 grid of 49 points plus one in the center."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    # Add one more point in the center
    grid = np.vstack([grid, [0.0, 0.0]])
    return grid[:N]

def _anneal(centers, time_budget):
    """Simulated annealing to improve the configuration."""
    start_time = time.time()
    current = centers.copy()
    current_radius = _compute_radius(current)
    best = current.copy()
    best_radius = current_radius
    
    # Initial temperature
    T = 0.001
    T_min = 1e-8
    cooling_rate = 0.999
    
    while time.time() - start_time < time_budget and T > T_min:
        # Generate random perturbation
        n_moves = 1
        move_indices = np.random.choice(N, size=n_moves, replace=False)
        
        for idx in move_indices:
            # Random displacement
            delta = np.random.normal(0, T, size=2)
            new_pos = current[idx] + delta
            # Keep within bounds
            new_pos = np.clip(new_pos, -0.5, 0.5)
            
            # Compute new radius
            old_pos = current[idx].copy()
            current[idx] = new_pos
            new_radius = _compute_radius(current)
            
            # Metropolis criterion
            if new_radius > current_radius or np.random.random() < np.exp((new_radius - current_radius) / T):
                current_radius = new_radius
                if new_radius > best_radius:
                    best_radius = new_radius
                    best = current.copy()
            else:
                current[idx] = old_pos
        
        T *= cooling_rate
    
    return best, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_centers = _construct_initial_grid()
    best_radius = _compute_radius(best_centers)
    
    # Phase 1: Binary search on r using violation minimization
    r_low = 0.0
    r_high = 0.08
    
    current_centers = best_centers.copy()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    for step in range(12):
        if time.time() > deadline - 30.0:
            break
            
        r_mid = (r_low + r_high) / 2.0
        x0 = current_centers.flatten()
        
        try:
            res = opt.minimize(
                _violation_objective,
                x0,
                jac=_violation_grad,
                args=(r_mid,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            x_res = res.x
            r_actual = _compute_radius(x_res.reshape(N, 2))
            
            if r_actual >= r_mid - 1e-7:
                # Feasible
                r_low = r_mid
                if r_actual > best_radius:
                    best_radius = r_actual
                    best_centers = x_res.reshape(N, 2).copy()
                current_centers = x_res.reshape(N, 2).copy()
            else:
                # Infeasible
                r_high = r_mid
                current_centers = x_res.reshape(N, 2).copy()
                
        except Exception:
            r_high = r_mid
    
    # Phase 2: Simulated annealing for global exploration
    if time.time() < deadline - 20.0:
        sa_time_budget = min(15.0, deadline - time.time() - 20.0)
        if sa_time_budget > 5.0:
            sa_centers, sa_radius = _anneal(current_centers, sa_time_budget)
            if sa_radius > best_radius:
                best_radius = sa_radius
                best_centers = sa_centers.copy()
    
    # Phase 3: Soft-min polish for high precision
    if time.time() < deadline - 5.0:
        polish_time_budget = min(15.0, deadline - time.time() - 5.0)
        if polish_time_budget > 3.0:
            # Use soft-min with analytic gradient
            x0 = best_centers.flatten()
            
            def _soft_min_wrapper(x):
                val, _ = _soft_min_obj_and_grad(x, beta=200.0)
                return val
            
            def _soft_min_grad_wrapper(x):
                _, grad = _soft_min_obj_and_grad(x, beta=200.0)
                return grad
            
            try:
                res = opt.minimize(
                    _soft_min_wrapper,
                    x0,
                    jac=_soft_min_grad_wrapper,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13}
                )
                x_res = res.x
                r_res = _compute_radius(x_res.reshape(N, 2))
                if r_res > best_radius:
                    best_radius = r_res
                    best_centers = x_res.reshape(N, 2).copy()
            except Exception:
                pass
            
            # Additional polish with higher beta
            if time.time() < deadline - 3.0:
                try:
                    res2 = opt.minimize(
                        _soft_min_wrapper,
                        best_centers.flatten(),
                        jac=_soft_min_grad_wrapper,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14}
                    )
                    x_res2 = res2.x
                    r_res2 = _compute_radius(x_res2.reshape(N, 2))
                    if r_res2 > best_radius:
                        best_radius = r_res2
                        best_centers = x_res2.reshape(N, 2).copy()
                except Exception:
                    pass
    
    # Ensure validity
    best_centers = np.clip(best_centers, -0.5, 0.5)
    if not np.all(np.isfinite(best_centers)):
        return _construct_initial_grid()
        
    return best_centers
# EVOLVE-BLOCK-END
