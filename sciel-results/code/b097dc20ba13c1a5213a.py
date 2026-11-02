"""Optimized packing of 50 equal circles in a unit square using smooth optimization."""
N = 50
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START
def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    # Ensure centers are within bounds (clip slightly inside for safety)
    centers = np.clip(centers, -0.5, 0.5)
    
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Minimum pairwise distance
    min_pair_dist = np.min(dists)
    
    # Minimum distance to boundary
    # For each center, distance to each of the 4 walls
    x = centers[:, 0]
    y = centers[:, 1]
    dist_to_left = x + 0.5
    dist_to_right = 0.5 - x
    dist_to_bottom = y + 0.5
    dist_to_top = 0.5 - y
    
    # Minimum distance to any boundary for each circle
    min_boundary_dist = np.min(np.stack([dist_to_left, dist_to_right, dist_to_bottom, dist_to_top]), axis=0)
    min_boundary = np.min(min_boundary_dist)
    
    # The radius is half of the minimum of these two
    r = 0.5 * min(min_pair_dist, min_boundary)
    
    return r, centers

def _soft_min_loss(centers, tau=0.01):
    """
    Smooth surrogate for the negative radius objective.
    We want to maximize r = 0.5 * min(min_pair_dist, min_boundary_dist).
    Equivalently, minimize -r.
    
    Using a soft-min approximation:
    min(a, b) ≈ -tau * log(exp(-a/tau) + exp(-b/tau))
    
    For multiple constraints, we can use:
    min_i d_i ≈ -tau * log(sum_i exp(-d_i/tau))
    
    The objective is:
    r = 0.5 * min(min_pair_dist, min_boundary_dist)
    
    We'll combine all constraints (pairwise and boundary) into a single set of distances
    and use soft-min over all of them.
    """
    centers = np.clip(centers, -0.5, 0.5)
    
    # Compute all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Get all pairwise distances (upper triangle)
    i_upper = np.triu_indices(N, k=1)
    pair_dists = dists[i_upper]
    
    # Compute boundary distances for each circle
    x = centers[:, 0]
    y = centers[:, 1]
    boundary_dists = np.concatenate([
        x + 0.5,       # left
        0.5 - x,       # right
        y + 0.5,       # bottom
        0.5 - y        # top
    ])
    
    # Combine all distances
    all_dists = np.concatenate([pair_dists, boundary_dists])
    
    # Soft-min approximation
    # min(all_dists) ≈ -tau * log(sum(exp(-all_dists/tau)))
    # For numerical stability, use log-sum-exp
    max_val = np.max(all_dists)
    log_sum_exp = max_val + np.log(np.sum(np.exp(-(all_dists - max_val) / tau)))
    soft_min_val = -tau * log_sum_exp
    
    # The radius is 0.5 * soft_min_val
    # We want to maximize this, so minimize -0.5 * soft_min_val
    return -0.5 * soft_min_val

def _soft_min_loss_grad(centers, tau=0.01):
    """
    Gradient of the soft-min loss.
    """
    centers = np.clip(centers, -0.5, 0.5)
    
    # Compute all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Get all pairwise distances (upper triangle)
    i_upper = np.triu_indices(N, k=1)
    pair_dists = dists[i_upper]
    
    # Compute boundary distances for each circle
    x = centers[:, 0]
    y = centers[:, 1]
    boundary_dists = np.concatenate([
        x + 0.5,       # left
        0.5 - x,       # right
        y + 0.5,       # bottom
        0.5 - y        # top
    ])
    
    # Combine all distances
    all_dists = np.concatenate([pair_dists, boundary_dists])
    
    # Soft-min: f = -tau * log(sum(exp(-d_i/tau)))
    # df/dd_i = -exp(-d_i/tau) / sum(exp(-d_j/tau))
    
    # For numerical stability
    max_val = np.max(all_dists)
    exp_terms = np.exp(-(all_dists - max_val) / tau)
    sum_exp = np.sum(exp_terms)
    
    # Weights for each distance
    weights = exp_terms / sum_exp
    
    # Gradient with respect to centers
    grad = np.zeros_like(centers)
    
    # Pairwise distances contribution
    for k in range(len(i_upper[0])):
        i, j = i_upper[0][k], i_upper[1][k]
        w = weights[k] / tau
        # d(pair_dist)/d(centers[i]) = -(centers[i] - centers[j]) / pair_dist
        # d(pair_dist)/d(centers[j]) = (centers[i] - centers[j]) / pair_dist
        d = diff[i, j]
        dist = pair_dists[k]
        if dist > 1e-10:
            grad[i] -= w * d / dist
            grad[j] += w * d / dist
    
    # Boundary distances contribution
    # Left: d_i = x_i + 0.5, d/dx_i = 1, d/dy_i = 0
    for i in range(N):
        w = weights[N*(N-1)//2 + i] / tau
        grad[i, 0] -= w  # d(x_i + 0.5)/dx_i = 1, but we have -w * d(dist)/d(center)
        # Actually: d(-tau*log(...))/d(centers) = -sum_k w_k * d(dist_k)/d(centers)
        # For left boundary: dist = x_i + 0.5, d(dist)/dx_i = 1
        # So contribution to grad[i,0] is -w * 1
        # Wait, let me reconsider.
        # f = -0.5 * soft_min_val
        # soft_min_val = -tau * log(sum(exp(-d_k/tau)))
        # f = 0.5 * tau * log(sum(exp(-d_k/tau)))
        # df/dd_k = 0.5 * tau * (1/sum) * (-1/tau) * exp(-d_k/tau) = -0.5 * exp(-d_k/tau)/sum = -0.5 * w_k
        # df/d(centers) = sum_k df/dd_k * dd_k/d(centers)
        # For left boundary: dd_k/dx_i = 1, so contribution to grad[i,0] = -0.5 * w * 1
        # I had the factor wrong. Let me redo.
    
    # Let me redo the gradient calculation more carefully.
    # f = -0.5 * soft_min_val where soft_min_val = -tau * log(sum(exp(-d_k/tau)))
    # So f = 0.5 * tau * log(sum(exp(-d_k/tau)))
    # Let S = sum(exp(-d_k/tau))
    # f = 0.5 * tau * log(S)
    # df/dd_k = 0.5 * tau * (1/S) * d(S)/dd_k = 0.5 * tau * (1/S) * (-1/tau) * exp(-d_k/tau)
    #         = -0.5 * exp(-d_k/tau) / S = -0.5 * w_k
    # where w_k = exp(-d_k/tau) / S
    # 
    # df/d(centers) = sum_k df/dd_k * dd_k/d(centers)
    
    grad = np.zeros_like(centers)
    
    # Pairwise distances
    for k in range(len(i_upper[0])):
        i, j = i_upper[0][k], i_upper[1][k]
        w = 0.5 * weights[k]
        d = diff[i, j]
        dist = pair_dists[k]
        if dist > 1e-10:
            # dd_k/d(centers[i]) = -(centers[i] - centers[j]) / dist = -d/dist
            # dd_k/d(centers[j]) = (centers[i] - centers[j]) / dist = d/dist
            grad[i] += w * (-d / dist)
            grad[j] += w * (d / dist)
    
    # Boundary distances
    base_idx = N*(N-1)//2
    for i in range(N):
        w = 0.5 * weights[base_idx + i]  # left
        grad[i, 0] += w * 1  # d(x_i + 0.5)/dx_i = 1
        
        w = 0.5 * weights[base_idx + N + i]  # right
        grad[i, 0] += w * (-1)  # d(0.5 - x_i)/dx_i = -1
        
        w = 0.5 * weights[base_idx + 2*N + i]  # bottom
        grad[i, 1] += w * 1  # d(y_i + 0.5)/dy_i = 1
        
        w = 0.5 * weights[base_idx + 3*N + i]  # top
        grad[i, 1] += w * (-1)  # d(0.5 - y_i)/dy_i = -1
    
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Initial configuration: hexagonal-like packing
    # Use a grid with slight perturbation to break symmetry
    np.random.seed(42)
    
    # Start with a roughly hexagonal arrangement
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small random perturbation
    centers += np.random.randn(N, 2) * 0.01
    
    # Clip to valid range
    centers = np.clip(centers, -0.49, 0.49)
    
    best_centers = centers.copy()
    best_radius, _ = _compute_radius(best_centers)
    
    # Progressive optimization with decreasing tau
    tau_values = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for tau in tau_values:
        if time.time() > deadline:
            break
        
        # Optimize with current tau
        for restart in range(3):
            if time.time() > deadline:
                break
            
            # Add small perturbation for restarts > 0
            if restart > 0:
                current_centers = best_centers + np.random.randn(N, 2) * 0.005
                current_centers = np.clip(current_centers, -0.49, 0.49)
            else:
                current_centers = best_centers.copy()
            
            # Define objective and gradient
            def obj(x):
                return _soft_min_loss(x, tau)
            
            def grad(x):
                return _soft_min_loss_grad(x, tau)
            
            # Use L-BFGS-B with bounds
            bounds = [(-0.5, 0.5)] * (2 * N)
            
            result = minimize(
                obj,
                current_centers,
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < obj(current_centers):
                new_centers = np.clip(result.x, -0.5, 0.5)
                new_radius, _ = _compute_radius(new_centers)
                
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = new_centers.copy()
    
    # Final polish with very small tau
    if time.time() < deadline - 10:
        tau = 0.0005
        current_centers = best_centers.copy()
        
        def obj(x):
            return _soft_min_loss(x, tau)
        
        def grad(x):
            return _soft_min_loss_grad(x, tau)
        
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        result = minimize(
            obj,
            current_centers,
            jac=grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if result.success or result.fun < obj(current_centers):
            new_centers = np.clip(result.x, -0.5, 0.5)
            new_radius, _ = _compute_radius(new_centers)
            
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers.copy()
    
    # Ensure the result is valid
    final_centers = np.clip(best_centers, -0.5, 0.5)
    final_radius, _ = _compute_radius(final_centers)
    
    return final_centers
# EVOLVE-BLOCK-END
