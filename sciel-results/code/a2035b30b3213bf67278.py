"""Optimized packing of 50 circles in a unit square using differentiable soft-min optimization."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _softmin(values, tau):
    """Soft-min using log-sum-exp: approximates min(values) as tau -> 0."""
    # Numerically stable: factor out max
    v = np.asarray(values, dtype=np.float64)
    vmax = np.max(v)
    # soft-min = vmax - tau * log(sum(exp(-(v - vmax)/tau)))
    return vmax - tau * np.log(np.sum(np.exp(-(v - vmax) / tau)))


def _score_of_centers(centers):
    """Compute the true packing radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    n = centers.shape[0]
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    # Distance to walls
    min_wall = np.min(0.5 - np.abs(centers))
    return min(0.5 * min_pair_dist, min_wall)


def _objective(params, tau, n):
    """
    Objective to minimize: -softmin of (pairwise distances/2, wall distances).
    params: flattened array of shape (2*n,)
    Returns scalar objective (negative soft-min radius).
    """
    centers = params.reshape(n, 2)
    # Wall distances: 0.5 - |c_ik| for each coordinate
    wall_dists = 0.5 - np.abs(centers)
    wall_flat = wall_dists.ravel()  # shape (2*n,)
    
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_dists = 0.5 * dists
    # Flatten upper triangle
    iu = np.triu_indices(n, k=1)
    pair_flat = pair_dists[iu]  # shape (n*(n-1)/2,)
    
    all_vals = np.concatenate([pair_flat, wall_flat])
    sm = _softmin(all_vals, tau)
    return -sm


def _objective_and_grad(params, tau, n):
    """Objective and analytic gradient."""
    centers = params.reshape(n, 2)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    wall_flat = wall_dists.ravel()
    
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_dists = 0.5 * dists
    iu = np.triu_indices(n, k=1)
    pair_flat = pair_dists[iu]
    
    all_vals = np.concatenate([pair_flat, wall_flat])
    sm = _softmin(all_vals, tau)
    obj = -sm
    
    # Gradient of softmin: d(softmin)/d(v_i) = -exp(-(v_i - vmax)/tau) / sum(exp(-(v_j - vmax)/tau))
    # So d(obj)/d(v_i) = -d(softmin)/d(v_i) = exp(-(v_i - vmax)/tau) / sum(exp(...))
    vmax = np.max(all_vals)
    exponents = np.exp(-(all_vals - vmax) / tau)
    weights = exponents / np.sum(exponents)
    
    # Gradient w.r.t. all_vals
    grad_all_vals = weights  # d(obj)/d(all_vals) = weights
    
    # Now chain rule back to centers
    grad_centers = np.zeros_like(centers)
    
    # Wall part: all_vals[n*(n-1)/2:] = wall_flat = 0.5 - |centers|
    n_pairs = n * (n - 1) // 2
    wall_weights = grad_all_vals[n_pairs:]
    # d(0.5 - |c|)/dc = -sign(c)
    wall_grad = -np.sign(centers).ravel() * wall_weights
    grad_centers += wall_grad.reshape(n, 2)
    
    # Pair part: pair_flat[k] = 0.5 * |c_i - c_j|
    pair_weights = grad_all_vals[:n_pairs]
    for k in range(n_pairs):
        i, j = iu[0][k], iu[1][k]
        d = centers[i] - centers[j]
        dist = np.sqrt(np.sum(d ** 2))
        if dist > 1e-12:
            # d(0.5*dist)/d(c_i) = 0.5 * d / dist
            unit = d / dist
            grad_centers[i] += 0.5 * pair_weights[k] * unit
            grad_centers[j] -= 0.5 * pair_weights[k] * unit
    
    return obj, grad_centers.ravel()


def _project_into_square(centers, margin=1e-6):
    """Clamp centers to stay inside the square with a small margin."""
    centers = centers.copy()
    centers[:, 0] = np.clip(centers[:, 0], -0.5 + margin, 0.5 - margin)
    centers[:, 1] = np.clip(centers[:, 1], -0.5 + margin, 0.5 - margin)
    return centers


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Initial grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    best_centers = centers.copy()
    best_score = _score_of_centers(best_centers)
    
    # Progressive soft-min optimization
    tau_schedule = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for tau in tau_schedule:
        if time.time() > deadline - 15.0:
            break
        
        current = best_centers.copy()
        params = current.ravel()
        
        # Use L-BFGS-B with analytic gradient
        max_iter = 500
        for it in range(max_iter):
            if time.time() > deadline - 15.0:
                break
            
            obj, grad = _objective_and_grad(params, tau, N)
            
            # Check for NaN/Inf
            if not np.all(np.isfinite(grad)) or not np.isfinite(obj):
                break
            
            # Gradient step with adaptive learning rate
            # Use a simple line search approach
            lr = 0.01
            new_params = params - lr * grad
            
            # Project into square
            new_centers = _project_into_square(new_params.reshape(N, 2))
            new_params = new_centers.ravel()
            
            new_obj, _ = _objective_and_grad(new_params, tau, N)
            
            if np.isfinite(new_obj) and new_obj < obj:
                params = new_params
            else:
                # Reduce learning rate
                lr *= 0.5
                if lr < 1e-8:
                    break
                new_params = params - lr * grad
                new_centers = _project_into_square(new_params.reshape(N, 2))
                new_params = new_centers.ravel()
        
        best_centers = _project_into_square(params.reshape(N, 2))
        current_score = _score_of_centers(best_centers)
        if current_score > best_score:
            best_score = current_score
    
    # Final polish: use scipy minimize with very tight tolerance
    if time.time() < deadline - 5.0:
        final_params = best_centers.ravel()
        
        def final_obj(p):
            return _objective(p, 0.0001, N)
        
        def final_grad(p):
            _, g = _objective_and_grad(p, 0.0001, N)
            return g
        
        res = minimize(
            final_obj, final_params, jac=final_grad,
            method='L-BFGS-B',
            options={'maxiter': 200, 'gtol': 1e-12, 'ftol': 1e-15}
        )
        if res.success and np.all(np.isfinite(res.x)):
            polished = _project_into_square(res.x.reshape(N, 2))
            polished_score = _score_of_centers(polished)
            if polished_score > best_score:
                best_centers = polished
                best_score = polished_score
    
    # Final validation and cleanup
    best_centers = _project_into_square(best_centers)
    # Ensure no coincident centers
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_centers[i] - best_centers[j])
            if d < 1e-10:
                best_centers[j] += 1e-8 * np.array([1.0, 0.0])
    
    return best_centers
# EVOLVE-BLOCK-END
