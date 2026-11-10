import numpy as np
import scipy.optimize as opt
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin

    # Initial configuration: 5x5x5 grid, take first 100
    # 5x5x5 = 125 points.
    # To make it more symmetric/balanced, we can remove 25 points.
    # Removing corners (8) and edge midpoints (12) and face centers (5) = 25.
    # Or just take a 5x5x4 block? No, 5x5x4 = 100.
    # A 5x5x4 grid is a valid starting point.
    m_x, m_y, m_z = 5, 5, 4
    ticks_x = (np.arange(m_x) + 0.5) / m_x - 0.5
    ticks_y = (np.arange(m_y) + 0.5) / m_y - 0.5
    ticks_z = (np.arange(m_z) + 0.5) / m_z - 0.5
    
    # Generate grid
    grid = np.array([(x, y, z) for z in ticks_z for y in ticks_y for x in ticks_x])
    # This gives 100 points.
    
    # Add small jitter to break symmetry
    np.random.seed(42)
    c = grid + np.random.normal(0, 0.01, size=(N, 3))
    
    # Pre-compute pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    pairs = np.column_stack((i_idx, j_idx))
    
    def objective_and_grad(c_flat):
        c = c_flat.reshape(N, 3)
        
        # Calculate distances between all pairs
        # d_ij = |c_i - c_j|
        # We want min(d_ij / 2)
        # and min(0.5 - |c_ik|) for each sphere i and dimension k
        
        # 1. Inter-sphere distances
        diff = c[pairs[:, 0]] - c[pairs[:, 1]]
        dists = np.linalg.norm(diff, axis=1)
        half_dists = dists / 2.0
        
        # 2. Wall distances
        # For each sphere i, distance to walls in x, y, z
        # d_x = 0.5 - |c_ix|
        # d_y = 0.5 - |c_iy|
        # d_z = 0.5 - |c_iz|
        wall_dists = 0.5 - np.abs(c)
        # Flatten wall_dists to 1D array of length N*3
        wall_dists_flat = wall_dists.flatten()
        
        # Combine all distances
        all_dists = np.concatenate([half_dists, wall_dists_flat])
        
        # Find minimum
        min_val = np.min(all_dists)
        
        # Calculate gradient
        # grad = 0
        # For each dist in all_dists that is close to min_val, add its gradient
        # Tolerance for "active" constraints
        tol = 1e-6
        active_mask = np.abs(all_dists - min_val) < tol
        
        grad = np.zeros((N, 3))
        
        # 1. Inter-sphere gradients
        # d = |c_i - c_j| / 2
        # d/dc_i ( - d ) = - (c_i - c_j) / (2 * |c_i - c_j|) = (c_j - c_i) / (2 * |c_i - c_j|)
        # d/dc_j ( - d ) = - (c_j - c_i) / (2 * |c_j - c_i|) = (c_i - c_j) / (2 * |c_j - c_i|)
        
        # We are minimizing -min_val. So we want the gradient of -min_val.
        # If min_val is determined by dist_k, grad = - grad(dist_k).
        # Let's compute grad of -min_val directly.
        # f = - min_val.
        # df/dc = - d(min_val)/dc.
        # If min_val = d_k, df/dc = - d(d_k)/dc.
        
        # For pair (i, j) with dist d_ij:
        # d(d_ij/2)/dc_i = (c_i - c_j) / (2 * d_ij)
        # d(d_ij/2)/dc_j = (c_j - c_i) / (2 * d_ij)
        
        # So d(-min)/dc_i = - (c_i - c_j) / (2 * d_ij)
        
        active_pairs = pairs[active_mask[:len(half_dists)]]
        if len(active_pairs) > 0:
            i_act = active_pairs[:, 0]
            j_act = active_pairs[:, 1]
            # Get the specific distances for these pairs
            # We need to map active indices back to the original pairs
            # active_mask[:len(half_dists)] corresponds to the first part of all_dists
            
            # Calculate gradients for active pairs
            diff_act = c[i_act] - c[j_act]
            dists_act = dists[active_mask[:len(half_dists)]]
            # Avoid division by zero
            dists_act[dists_act < 1e-12] = 1e-12
            
            # Gradient contribution for c_i: - (c_i - c_j) / (2 * d)
            # Gradient contribution for c_j: - (c_j - c_i) / (2 * d) = (c_i - c_j) / (2 * d)
            
            grad_i = -diff_act / (2 * dists_act[:, np.newaxis])
            grad_j = diff_act / (2 * dists_act[:, np.newaxis])
            
            # Accumulate
            # Use np.add.at for accumulation
            np.add.at(grad, i_act, grad_i)
            np.add.at(grad, j_act, grad_j)
            
        # 2. Wall gradients
        # d(0.5 - |c_ik|)/dc_ik = - sign(c_ik)
        # d(-min)/dc_ik = - d(0.5 - |c_ik|)/dc_ik = sign(c_ik)
        
        active_walls = active_mask[len(half_dists):]
        if np.any(active_walls):
            # Map active wall indices back to sphere and dimension
            # wall_dists_flat index = i * 3 + k
            active_wall_indices = np.where(active_walls)[0]
            sphere_ids = active_wall_indices // 3
            dim_ids = active_wall_indices % 3
            
            # c[sphere_ids, dim_ids]
            c_vals = c[sphere_ids, dim_ids]
            signs = np.sign(c_vals)
            # If c_val is 0, sign is 0, which is a valid sub-gradient
            # Grad contribution is + sign(c_ik)
            # Because f = - (0.5 - |c|) = |c| - 0.5. d/dc = sign(c).
            
            # Accumulate
            for s, d, sg in zip(sphere_ids, dim_ids, signs):
                grad[s, d] += sg
            
        return -min_val, grad.flatten()

    # Optimization
    best_c = c
    best_val = objective_and_grad(c.flatten())[0]
    
    # Use L-BFGS-B or BFGS
    # L-BFGS-B is good for large N
    # We will run multiple iterations, checking time
    
    res = None
    try:
        # Run minimize
        # maxiter is a soft limit, we check time in a callback
        def callback(xk):
            nonlocal best_c, best_val
            if time.time() > deadline:
                raise Exception("Timeout")
            val, _ = objective_and_grad(xk)
            if val < best_val:
                best_val = val
                best_c = xk.reshape(N, 3)
            return False # Continue
        
        # BFGS is often more robust for non-smooth objectives than L-BFGS-B
        # but L-BFGS-B scales better. Let's try L-BFGS-B first.
        # If it fails, we can try BFGS.
        
        # Actually, for packing, BFGS with finite difference or analytical grad is standard.
        # Let's use BFGS.
        
        # To handle the non-smoothness, we might need to run multiple restarts or use a soft-min.
        # For now, let's try direct optimization.
        
        # We will run a loop of minimization, potentially with different initial jitters if time permits.
        
        result = opt.minimize(
            objective_and_grad,
            c.flatten(),
            method='BFGS',
            jac=True,
            options={'maxiter': 1000, 'gtol': 1e-5}
        )
        
        if result.success or result.fun < best_val:
             best_c = result.x.reshape(N, 3)
             best_val = result.fun
             
    except Exception as e:
        pass # Keep best_c found so far

    # Final polish with L-BFGS-B if time remains
    if time.time() < deadline - 10:
        try:
            result2 = opt.minimize(
                objective_and_grad,
                best_c.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'gtol': 1e-6}
            )
            if result2.fun < best_val:
                best_c = result2.x.reshape(N, 3)
                best_val = result2.fun
        except Exception:
            pass

    return best_c
