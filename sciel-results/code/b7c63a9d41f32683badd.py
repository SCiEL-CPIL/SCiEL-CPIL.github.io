"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _soft_min(values, beta):
    """Compute soft minimum using log-sum-exp for numerical stability."""
    # values: array of values to take soft-min of
    # beta: temperature parameter (larger = sharper minimum)
    min_val = np.min(values)
    shifted = values - min_val
    # exp(-beta * shifted) is safe since shifted >= 0
    exp_vals = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_vals))
    return min_val - log_sum / beta

def _compute_score(centers):
    """Compute the actual packing radius (minimum of pairwise distances/2 and wall distances)."""
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Wall distances
    wall_dists = np.minimum(0.5 - np.abs(centers), axis=1)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist, min_wall_dist)

def _objective_and_grad(centers, beta, wall_penalty):
    """Objective function and gradient for optimization."""
    # Reshape to (N, 2)
    x = centers.reshape(-1, 2)
    
    # Compute all pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1) + 1e-12)
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(x)
    
    # Combine all constraints: pairwise distances and wall distances
    # We want to maximize the minimum of (dists/2) and wall_dists
    # So we minimize -soft_min of all these values
    
    # For pairwise: use dists/2
    pair_vals = dists / 2.0
    # For walls: use wall_dists
    # We need to handle the fact that we have both types
    
    # Create a combined set of values for soft-min
    # Pairwise: N*(N-1)/2 values, Walls: 2*N values
    # This is large, so let's be smart about it
    
    # Actually, let's use a different approach: 
    # Objective = -soft_min(all_constraints) + penalty for violations
    
    # Get all finite pairwise distances
    pair_mask = np.isfinite(pair_vals)
    pair_vals_flat = pair_vals[pair_mask]
    
    # Wall distances (4 per circle: left, right, bottom, top)
    left_d = 0.5 + x[:, 0]   # distance to left wall
    right_d = 0.5 - x[:, 0]  # distance to right wall
    bottom_d = 0.5 + x[:, 1] # distance to bottom wall
    top_d = 0.5 - x[:, 1]    # distance to top wall
    
    wall_vals = np.concatenate([left_d, right_d, bottom_d, top_d])
    
    # Combine all constraint values
    all_vals = np.concatenate([pair_vals_flat, wall_vals])
    
    # Soft minimum
    soft_min_val = _soft_min(all_vals, beta)
    
    # Penalty for any constraint violation (negative values)
    violations = np.maximum(0, -all_vals)
    penalty = wall_penalty * np.sum(violations**2)
    
    obj = -soft_min_val + penalty
    
    # Gradient
    # Gradient of -soft_min: 
    # d/dx_i [-soft_min] = sum_j (exp(-beta*(v_j - min_val)) / sum_k exp(-beta*(v_k - min_val))) * beta * dv_j/dx_i
    # where v_j are the constraint values
    
    min_val = np.min(all_vals)
    shifted = all_vals - min_val
    exp_vals = np.exp(-beta * shifted)
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(x)
    
    # Gradient contributions from pairwise distances
    # For each pair (i,j), the constraint value is dist(i,j)/2
    # d(dist(i,j)/2)/dx_i = (x_i - x_j) / dist(i,j)
    # d(dist(i,j)/2)/dx_j = (x_j - x_i) / dist(i,j)
    
    # We need to map weights back to pairwise and wall constraints
    n_pairs = len(pair_vals_flat)
    pair_weights = weights[:n_pairs]
    wall_weights = weights[n_pairs:]
    
    # Pairwise gradient
    # Reconstruct which pairs correspond to which indices
    pair_indices = np.argwhere(pair_mask)
    for idx in range(len(pair_indices)):
        i, j = pair_indices[idx]
        if i == j:
            continue
        w = pair_weights[idx] * beta
        if w < 1e-15:
            continue
        diff_ij = x[i] - x[j]
        dist_ij = np.sqrt(np.sum(diff_ij**2) + 1e-12)
        grad[i] += w * diff_ij / dist_ij
        grad[j] -= w * diff_ij / dist_ij
    
    # Wall gradients
    # left_d = 0.5 + x[:,0], d/dx[:,0] = 1
    # right_d = 0.5 - x[:,0], d/dx[:,0] = -1
    # bottom_d = 0.5 + x[:,1], d/dx[:,1] = 1
    # top_d = 0.5 - x[:,1], d/dx[:,1] = -1
    
    n = len(x)
    left_w = wall_weights[0:n]
    right_w = wall_weights[n:2*n]
    bottom_w = wall_weights[2*n:3*n]
    top_w = wall_weights[3*n:4*n]
    
    grad[:, 0] += beta * (left_w - right_w)
    grad[:, 1] += beta * (bottom_w - top_w)
    
    # Penalty gradient
    # penalty = wall_penalty * sum(max(0, -all_vals)^2)
    # d/dx of penalty: for violated constraints, 2 * wall_penalty * (-all_vals) * d(-all_vals)/dx
    # = -2 * wall_penalty * (-all_vals) * d(all_vals)/dx for violated ones
    
    # Pairwise penalty gradient
    for idx in range(len(pair_indices)):
        i, j = pair_indices[idx]
        if i == j:
            continue
        v = pair_vals_flat[idx]
        if v < 0:
            w = -2 * wall_penalty * (-v) * (-1)  # d(-v)/dx = -dv/dx
            # Actually: penalty term is wall_penalty * max(0, -v)^2
            # d/dx [wall_penalty * (-v)^2] for v<0 = wall_penalty * 2*(-v)*(-dv/dx) = -2*wall_penalty*v*dv/dx
            # dv/dx_i = (x_i - x_j)/dist, so:
            diff_ij = x[i] - x[j]
            dist_ij = np.sqrt(np.sum(diff_ij**2) + 1e-12)
            grad[i] += -2 * wall_penalty * v * diff_ij / dist_ij
            grad[j] -= -2 * wall_penalty * v * diff_ij / dist_ij
    
    # Wall penalty gradient
    # For left_d = 0.5 + x[:,0]: if left_d < 0, penalty contributes -2*wall_penalty*left_d*1 to grad[:,0]
    # etc.
    for idx in range(n):
        if left_d[idx] < 0:
            grad[idx, 0] += -2 * wall_penalty * left_d[idx]
        if right_d[idx] < 0:
            grad[idx, 0] += -2 * wall_penalty * right_d[idx] * (-1)
        if bottom_d[idx] < 0:
            grad[idx, 1] += -2 * wall_penalty * bottom_d[idx]
        if top_d[idx] < 0:
            grad[idx, 1] += -2 * wall_penalty * top_d[idx] * (-1)
    
    return obj, grad.ravel()

def _hexagonal_init(n_circles, scale=1.0, jitter=0.0):
    """Initialize circles in a hexagonal-like pattern."""
    centers = []
    dx = 1.0 / np.sqrt(n_circles)
    dy = dx
    dx_hex = dx * np.sqrt(3) / 2
    
    row = 0
    y = -0.5 + dy/2
    while len(centers) < n_circles:
        offset = (dx_hex / 2) if (row % 2 == 1) else 0.0
        x = -0.5 + dx_hex/2 + offset
        col = 0
        while x < 0.5 and len(centers) < n_circles:
            centers.append([x + jitter * (np.random.rand() - 0.5), 
                           y + jitter * (np.random.rand() - 0.5)])
            x += dx_hex
            col += 1
        y += dy
        row += 1
    
    return np.array(centers[:n_circles])

def construct():
    """Construct optimal packing of 200 circles in unit square."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_centers = None
    best_score = 0.0
    
    # Try multiple initializations and optimization runs
    n_restarts = 10
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        # Different initializations
        if restart == 0:
            # Hexagonal with small jitter
            np.random.seed(42 + restart)
            centers = _hexagonal_init(N, jitter=0.02)
        elif restart == 1:
            # Grid with jitter
            m = int(np.ceil(np.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            centers = grid[:N].copy()
            centers += np.random.normal(0, 0.01, centers.shape)
        elif restart == 2:
            # Hexagonal with larger jitter
            np.random.seed(100 + restart)
            centers = _hexagonal_init(N, jitter=0.05)
        elif restart == 3:
            # Random uniform
            np.random.seed(200 + restart)
            centers = np.random.uniform(-0.45, 0.45, (N, 2))
        else:
            # Perturbed best so far
            if best_centers is not None:
                np.random.seed(300 + restart)
                centers = best_centers + np.random.normal(0, 0.02, (N, 2))
                centers = np.clip(centers, -0.49, 0.49)
            else:
                np.random.seed(300 + restart)
                centers = _hexagonal_init(N, jitter=0.03)
        
        # Clamp to valid region
        centers = np.clip(centers, -0.49, 0.49)
        
        # Check initial score
        initial_score = _compute_score(centers)
        if initial_score > best_score:
            best_score = initial_score
            best_centers = centers.copy()
        
        # Run optimization with annealing beta
        # Start with low beta (smooth), increase to sharp
        beta_schedule = [5, 10, 20, 50, 100, 200, 500]
        wall_penalty = 100.0
        
        x = centers.flatten()
        
        for beta in beta_schedule:
            if time.time() > deadline - 10.0:
                break
            
            # Run optimization
            try:
                result = minimize(
                    _objective_and_grad,
                    x,
                    args=(beta, wall_penalty),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                x = result.x
            except Exception:
                break
            
            # Evaluate
            current_centers = x.reshape(-1, 2)
            current_score = _compute_score(current_centers)
            
            if current_score > best_score:
                best_score = current_score
                best_centers = current_centers.copy()
        
        # Final polish with high beta
        if time.time() < deadline - 5.0:
            try:
                result = minimize(
                    _objective_and_grad,
                    x,
                    args=(1000, 1000.0),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                x = result.x
                current_centers = x.reshape(-1, 2)
                current_score = _compute_score(current_centers)
                
                if current_score > best_score:
                    best_score = current_score
                    best_centers = current_centers.copy()
            except Exception:
                pass
    
    # Final validation and return
    if best_centers is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_centers = np.array([(x, y) for y in ticks for x in ticks])[:N]
    
    # Ensure valid: clip to square and check
    best_centers = np.clip(best_centers, -0.5 + 1e-10, 0.5 - 1e-10)
    
    # Final score check
    final_score = _compute_score(best_centers)
    
    return best_centers
# EVOLVE-BLOCK-END
