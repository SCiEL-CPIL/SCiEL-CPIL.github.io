"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
def _generate_grid(nx, ny, nz):
    """Generate a grid of points in [-0.5, 0.5]^3."""
    xs = (np.arange(nx) + 0.5) / nx - 0.5
    ys = (np.arange(ny) + 0.5) / ny - 0.5
    zs = (np.arange(nz) + 0.5) / nz - 0.5
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return grid

def _compute_radius(centers):
    """Compute the maximum valid radius for the given centers."""
    # Check bounds
    # Centers must be in [-0.5, 0.5]
    # Distance to wall: 0.5 - |coord|
    # Min distance to wall
    d_walls = np.min(0.5 - np.abs(centers), axis=1)
    min_wall_dist = np.min(d_walls)
    
    if min_wall_dist < 0:
        return 0.0
        
    # Min distance between pairs
    # Use cdist or manual calculation
    # For N=100, pairwise distance matrix is 100x100, very fast.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    r = min(min_wall_dist, min_pair_dist / 2.0)
    return r

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Initial guess: 5x5x4 grid
    # 5*5*4 = 100
    centers = _generate_grid(5, 5, 4).flatten()
    
    # Objective function: minimize -softmin(distances)
    # softmin(x) = -T * log(sum(exp(-x/T)))
    # We want to maximize min(distances), so minimize -softmin(distances)
    # -softmin(distances) = T * log(sum(exp(-distances/T)))
    
    # Precompute indices for pairs and walls to speed up gradient calculation
    # Pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_indices = np.stack([i_idx, j_idx], axis=1)
    
    # Walls
    # For each center i and each dim k, distance is 0.5 - |c_{ik}|
    # We treat each wall constraint as a separate 'distance' term in the softmin.
    # There are N*3 wall terms.
    
    T = 0.05  # Temperature for softmin
    
    def objective_and_grad(flat_centers):
        if time.time() > deadline:
            return 0.0, np.zeros_like(flat_centers)
            
        c = flat_centers.reshape(N, 3)
        
        # Calculate pair distances squared (we use squared dist in exp to avoid sqrt, 
        # but the objective is on distances, not squared distances. 
        # Actually, maximizing min distance is equivalent to maximizing min squared distance.
        # Let's use squared distances for numerical stability and speed.
        # dist_sq = sum((ci - cj)^2)
        # Wall dist sq = (0.5 - |c_ik|)^2
        
        # Pair distances
        ci = c[i_idx]
        cj = c[j_idx]
        diff = ci - cj
        dist_sq_pairs = np.sum(diff**2, axis=1)
        
        # Wall distances
        # c is (N, 3)
        # dist to wall in dim k for point i is 0.5 - |c[i,k]|
        # We create a vector of all wall distances squared
        # Shape: (N*3,)
        abs_c = np.abs(c)
        wall_dists = 0.5 - abs_c
        # Flatten to (N*3,)
        wall_dists_flat = wall_dists.flatten()
        dist_sq_walls = wall_dists_flat**2
        
        # Combine all distances squared
        all_dists_sq = np.concatenate([dist_sq_pairs, dist_sq_walls])
        
        # Softmin of dist_sq
        # f = T * log(sum(exp(-dist_sq / T)))
        # To avoid overflow/underflow in exp, subtract max
        # Since we are minimizing f, and exp(-x) is decreasing, 
        # the term with smallest x (min dist) dominates the sum.
        # Let's just compute it carefully.
        
        # Note: If any dist_sq is negative (inside wall), we should penalize heavily.
        # The grid init keeps them inside. L-BFGS-B bounds will help.
        
        # Clip small negative values to 0 to avoid log issues if any
        all_dists_sq = np.maximum(all_dists_sq, 1e-12)
        
        # Compute sum exp(-x/T)
        # x/T can be large negative if x is small? No, x is positive.
        # If x is small, -x/T is close to 0.
        # If x is large, -x/T is large negative, exp is small.
        
        scaled = -all_dists_sq / T
        max_scaled = np.max(scaled)
        sum_exp = np.sum(np.exp(scaled - max_scaled))
        
        val = T * (np.log(sum_exp) + max_scaled)
        
        # Gradient
        # d/dx_k [ T * log(sum exp(-x_i/T)) ]
        # = T * (1/sum) * sum( d/dx_k exp(-x_i/T) )
        # = T * (1/sum) * sum( exp(-x_i/T) * (-1/T) * delta_ik )
        # = - (exp(-x_k/T) / sum)
        
        # We need gradient with respect to flat_centers.
        # Let's compute weights w_i = exp(-dist_sq_i / T) / sum_exp
        # w_i = exp(scaled_i - max_scaled) / sum_exp
        
        weights = np.exp(scaled - max_scaled) / sum_exp
        
        # Gradient for pairs
        # dist_sq_pairs = sum(diff^2)
        # d(dist_sq)/d(ci) = 2 * (ci - cj)
        # d(dist_sq)/d(cj) = 2 * (cj - ci)
        
        grad_flat = np.zeros_like(flat_centers)
        
        # Pair gradients
        # For each pair p, weight w_p
        # d(val)/d(c_i) += w_p * (-1) * d(dist_sq_p)/d(c_i)
        # Wait, val = T log sum exp. 
        # d(val)/d(x_k) = - w_k where w_k = exp(-x_k/T)/sum.
        # So d(val)/d(c_i) = sum_{pairs involving i} - w_p * d(dist_sq_p)/d(c_i)
        
        # Vectorized:
        # grad_pair_i = - weights_pairs * 2 * (ci - cj)
        # grad_pair_j = - weights_pairs * 2 * (cj - ci)
        
        # We need to accumulate these into the gradient array.
        # Using np.add.at for efficiency with repeated indices
        
        # grad_i_contrib = -weights * 2 * diff
        # grad_j_contrib = -weights * 2 * (-diff) = weights * 2 * diff
        
        grad_i = -weights * 2 * diff
        grad_j = weights * 2 * diff
        
        np.add.at(grad_flat, i_idx * 3, grad_i)
        np.add.at(grad_flat, j_idx * 3, grad_j)
        
        # Wall gradients
        # dist_sq_wall = (0.5 - |c_ik|)^2
        # d(dist_sq)/d(c_ik) = 2 * (0.5 - |c_ik|) * d(0.5 - |c_ik|)/d(c_ik)
        # d(0.5 - |c_ik|)/d(c_ik) = -sign(c_ik)
        # So d(dist_sq)/d(c_ik) = -2 * (0.5 - |c_ik|) * sign(c_ik)
        
        # We have wall_dists_flat corresponding to c.flatten()
        # wall_dists = 0.5 - abs_c
        # signs = np.sign(c)
        # d_dist_sq_dc = -2 * wall_dists * signs
        
        signs = np.sign(c)
        # Handle 0: sign(0) is 0, which is fine as derivative of |x| at 0 is undefined, 
        # but in practice for optimization, 0 is a good subgradient.
        d_dist_sq_dc_walls = -2 * wall_dists * signs
        
        # Gradient contribution: - weights_walls * d_dist_sq_dc_walls
        grad_walls = -weights[N*(N-1)//2:] * d_dist_sq_dc_walls
        
        # Add to grad_flat
        # wall_dists_flat is c.flatten() order
        grad_flat += grad_walls
        
        return val, grad_flat.flatten()
    
    # Bounds: centers must be in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Run optimization
    best_result = None
    best_val = np.inf
    
    # Try a few different T values or just one good one
    # T=0.05 is a good start
    try:
        res = minimize(
            objective_and_grad, 
            centers, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12, 'maxfun': 1000}
        )
        if res.success or res.fun < best_val:
            best_val = res.fun
            best_result = res.x
    except Exception as e:
        pass
        
    # If time allows, try a finer T or another run
    if time.time() < deadline - 10:
        # Refine with smaller T
        T = 0.01
        # Re-define objective with new T? 
        # The closure captures T. We need to re-define or make T a parameter.
        # For simplicity, let's just do one more run if we have time, 
        # but we need to update the T in the closure.
        # Let's create a factory function.
        
        def make_obj(T_val):
            def obj(flat_centers):
                if time.time() > deadline:
                    return 0.0, np.zeros_like(flat_centers)
                c = flat_centers.reshape(N, 3)
                ci = c[i_idx]
                cj = c[j_idx]
                diff = ci - cj
                dist_sq_pairs = np.sum(diff**2, axis=1)
                abs_c = np.abs(c)
                wall_dists = 0.5 - abs_c
                wall_dists_flat = wall_dists.flatten()
                dist_sq_walls = wall_dists_flat**2
                all_dists_sq = np.concatenate([dist_sq_pairs, dist_sq_walls])
                all_dists_sq = np.maximum(all_dists_sq, 1e-12)
                scaled = -all_dists_sq / T_val
                max_scaled = np.max(scaled)
                sum_exp = np.sum(np.exp(scaled - max_scaled))
                val = T_val * (np.log(sum_exp) + max_scaled)
                weights = np.exp(scaled - max_scaled) / sum_exp
                grad_flat = np.zeros_like(flat_centers)
                grad_i = -weights * 2 * diff
                grad_j = weights * 2 * diff
                np.add.at(grad_flat, i_idx * 3, grad_i)
                np.add.at(grad_flat, j_idx * 3, grad_j)
                signs = np.sign(c)
                d_dist_sq_dc_walls = -2 * wall_dists * signs
                grad_walls = -weights[N*(N-1)//2:] * d_dist_sq_dc_walls
                grad_flat += grad_walls
                return val, grad_flat.flatten()
            return obj
        
        obj_func = make_obj(T)
        try:
            res2 = minimize(
                obj_func, 
                best_result if best_result is not None else centers, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12, 'maxfun': 1000}
            )
            if res2.success or res2.fun < best_val:
                best_val = res2.fun
                best_result = res2.x
        except Exception as e:
            pass

    if best_result is None:
        best_result = centers
        
    # Final check and return
    final_centers = best_result.reshape(N, 3)
    
    # Ensure validity: clip to bounds if slightly out
    final_centers = np.clip(final_centers, -0.5, 0.5)
    
    return final_centers

# EVOLVE-BLOCK-END
