import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _softmin(vals, beta):
    # Stable log-sum-exp for softmin
    # softmin(v) = -1/beta * log(sum(exp(-beta * v)))
    # To avoid overflow, subtract max(-beta * v)
    neg_beta_v = -beta * vals
    max_neg_beta_v = np.max(neg_beta_v)
    # If max is -inf, handle separately, but vals are finite
    sum_exp = np.sum(np.exp(neg_beta_v - max_neg_beta_v))
    return -(1.0 / beta) * (np.log(sum_exp) + max_neg_beta_v)

def _compute_radius(centers):
    # Compute the true non-smooth radius
    n = centers.shape[0]
    min_dist = np.inf
    
    # Pairwise distances
    # To save memory/time, we can do this in chunks or just full matrix for N=50
    # 50x50 matrix is small
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Boundary distances
    # dist to wall = 0.5 - |coord|
    # min over all coords
    min_boundary_dist = np.min(0.5 - np.abs(centers))
    
    return min(min_pair_dist, min_boundary_dist)

def _repulsion_step(centers, step_size):
    """
    Perform a simple repulsion step to spread out spheres.
    This helps escape bad local minima from grid initialization.
    """
    n = centers.shape[0]
    forces = np.zeros_like(centers)
    
    # Pairwise repulsion
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0) # Avoid division by zero, we'll mask later
    
    # Force is proportional to 1/d^2 in direction of separation
    # We use a softening to avoid singularities
    safe_dists = np.maximum(dists, 1e-6)
    # inv_d2 = 1.0 / (safe_dists**2)
    # We want to push them apart.
    # F_ij = (c_i - c_j) / dist^3 * strength
    # Let's just use a simple inverse distance force for stability
    inv_d = 1.0 / safe_dists
    inv_d[np.eye(n, dtype=bool)] = 0
    
    # Force on i from j is (c_i - c_j) * inv_d * inv_d
    # Vectorized: 
    # F_i = sum_j (c_i - c_j) / dist_ij^2
    # This is equivalent to:
    # Sum_j c_i * inv_d_ij^2 - Sum_j c_j * inv_d_ij^2
    # Note: inv_d is symmetric.
    # Let's just compute the matrix product.
    # W[i,j] = inv_d[i,j]^2
    W = inv_d * inv_d
    sum_W = np.sum(W, axis=1)
    W_c = W.dot(centers)
    forces += (centers * sum_W[:, np.newaxis] - W_c)
    
    # Boundary repulsion (keep them inside)
    # If close to boundary, push inwards
    # dist_to_wall = 0.5 - |c|
    # If dist < threshold, push
    # Simple linear push:
    margin = 0.1
    dists_to_wall = 0.5 - np.abs(centers)
    # If dist < margin, force = (margin - dist) * sign(c)
    # sign(c) points outwards, so we want -sign(c) to push inwards?
    # No, if c > 0, wall is at 0.5. dist = 0.5 - c.
    # If c is close to 0.5, dist is small.
    # We want to decrease c. So force is negative.
    # -sign(c) is -1 if c>0. Correct.
    # If c < 0, wall is at -0.5. dist = 0.5 - |c| = 0.5 + c.
    # If c is close to -0.5, dist is small.
    # We want to increase c. So force is positive.
    # -sign(c) is -(-1) = 1. Correct.
    
    mask = dists_to_wall < margin
    forces[mask] -= (margin - dists_to_wall[mask]) * np.sign(centers[mask]) * 10.0 # Stronger boundary force
    
    # Update
    new_centers = centers + step_size * forces
    # Clip to box to prevent explosion
    new_centers = np.clip(new_centers, -0.49, 0.49)
    return new_centers

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_radius = 0.0
    best_centers = None
    
    # Initial best from seed (grid)
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    initial_centers = grid[:N].copy()
    
    # Improve initial centers with a few repulsion steps
    # This is cheap and helps a lot
    for i in range(50):
        step = 0.01 * (1.0 - i/50.0)
        initial_centers = _repulsion_step(initial_centers, step)
    
    best_radius = _compute_radius(initial_centers)
    best_centers = initial_centers.copy()
    
    # Objective and Gradient functions
    def objective_and_grad(x, beta):
        centers = x.reshape(N, 3)
        
        # 1. Pairwise terms
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Avoid division by zero
        dists = np.where(dists < 1e-10, 1e-10, dists)
        
        v_pair = dists / 2.0
        # Mask out diagonal
        mask = ~np.eye(N, dtype=bool)
        
        # Boundary contributions
        v_bound = 0.5 - np.abs(centers)
        
        # Weights
        exp_pair = np.exp(-beta * v_pair)
        np.fill_diagonal(exp_pair, 0)
        
        S_pair = 0.5 * np.sum(exp_pair)
        
        # Gradient contribution from pairs
        W = exp_pair / (2.0 * dists)
        np.fill_diagonal(W, 0)
        
        sum_W = np.sum(W, axis=1)
        W_c = W.dot(centers)
        
        G_pair = 0.5 * (centers * sum_W[:, np.newaxis] - W_c)
        
        # Boundary part
        exp_bound = np.exp(-beta * v_bound)
        S_bound = np.sum(exp_bound)
        
        signs = np.sign(centers)
        G_bound = -exp_bound * signs
        
        # Total S and G
        S_total = S_pair + S_bound
        G_total = G_pair + G_bound
        
        # Gradient of L = -softmin
        grad = G_total / S_total
        
        # Objective value
        obj = (1.0 / beta) * np.log(S_total)
        
        return obj, grad.ravel()

    # Optimization loop
    num_restarts = 15
    # More gradual beta increase, and a final high-beta polish
    betas = [5, 10, 20, 50, 100, 200, 500]
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        if r == 0:
            x0 = best_centers.ravel()
        else:
            # Random initialization within the cube
            # Add some noise to previous best or pure random
            if r % 2 == 0:
                x0 = best_centers.ravel() + np.random.normal(0, 0.02, N*3)
                x0 = np.clip(x0, -0.49, 0.49)
            else:
                x0 = np.random.uniform(-0.45, 0.45, N*3)
                
        # Annealing over betas
        current_x = x0
        for i, beta in enumerate(betas):
            if time.time() > deadline:
                break
                
            # Use tighter tolerances for higher betas
            if beta > 100:
                ftol = 1e-14
                gtol = 1e-10
                maxiter = 300
            else:
                ftol = 1e-12
                gtol = 1e-8
                maxiter = 200
                
            try:
                res = minimize(
                    objective_and_grad,
                    current_x,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True, # We return (f, grad)
                    bounds=[(-0.5, 0.5)] * (N * 3),
                    options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
                )
                current_x = res.x
            except Exception:
                pass
                
        # Evaluate true radius
        centers = current_x.reshape(N, 3)
        # Ensure inside cube
        centers = np.clip(centers, -0.5, 0.5)
        
        r_curr = _compute_radius(centers)
        
        if r_curr > best_radius:
            best_radius = r_curr
            best_centers = centers.copy()
            
    # Final Polish: Use the best centers found and run a few more steps with very high beta
    # This is the "high-precision polish pass" suggested in the directive
    if time.time() < deadline - 10.0:
        try:
            x_polish = best_centers.ravel()
            # Run a short optimization with very high beta
            res_polish = minimize(
                objective_and_grad,
                x_polish,
                args=(1000,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            centers_polish = res_polish.x.reshape(N, 3)
            centers_polish = np.clip(centers_polish, -0.5, 0.5)
            r_polish = _compute_radius(centers_polish)
            
            if r_polish > best_radius:
                best_radius = r_polish
                best_centers = centers_polish.copy()
        except Exception:
            pass
            
    return best_centers
# EVOLVE-BLOCK-END
