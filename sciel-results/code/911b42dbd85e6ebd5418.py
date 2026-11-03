"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the valid radius for a given set of centers."""
    # Check bounds
    min_dist_to_wall = np.min(0.5 - np.abs(centers))
    if min_dist_to_wall < 0:
        return 0.0
        
    # Compute pairwise distances
    n = len(centers)
    min_pair_dist = np.inf
    # Efficient pairwise distance calculation
    # Using broadcasting might be memory heavy for N=200 (200x200 is fine, 40000 entries)
    # But let's stick to a loop or scipy.spatial.distance.pdist for speed if needed.
    # For N=200, pdist is very fast.
    from scipy.spatial.distance import pdist
    dists = pdist(centers, 'euclidean')
    if len(dists) > 0:
        min_pair_dist = np.min(dists)
    else:
        min_pair_dist = np.inf
        
    if not np.isfinite(min_pair_dist):
        return 0.0
        
    r = min(min_dist_to_wall, min_pair_dist / 2.0)
    return max(0.0, r)

def _soft_min_neg(dists, beta):
    """
    Compute the soft minimum of distances using log-sum-exp.
    We want to maximize the minimum distance.
    Equivalent to minimizing -min(dists).
    -min(dists) = -(-logsumexp(-dists*beta)/beta) = logsumexp(-dists*beta)/beta
    """
    # To avoid overflow in exp, subtract max
    x = -dists * beta
    x_max = np.max(x)
    if not np.isfinite(x_max):
        return 0.0
    return (x_max + np.log(np.sum(np.exp(x - x_max)))) / beta

def _generate_hex_grid():
    """
    Generate a hexagonal lattice packing of N points in [-0.5, 0.5]^2.
    This provides a good initial guess close to the optimal packing structure.
    """
    # Hexagonal packing efficiency is pi/sqrt(12) ~ 0.9069
    # Area per circle ~ 1 / (N * eff)
    # r ~ 0.5 * sqrt(eff / N)
    # Let's try to fit a hexagonal grid.
    
    # Number of rows
    # If we have R rows, and roughly N/R columns.
    # Width ~ (N/R) * d
    # Height ~ (R-1) * d * sqrt(3)/2
    # We want Width <= 1 and Height <= 1.
    
    best_centers = None
    best_r = -1
    
    # Try different row counts
    for n_rows in range(10, 25):
        n_cols = math.ceil(N / n_rows)
        
        # Estimate d
        # Width constraint: n_cols * d <= 1  => d <= 1/n_cols
        # Height constraint: (n_rows-1)*d*sqrt(3)/2 + d <= 1 (approx)
        # Let's solve for d that fits both.
        d_width = 1.0 / n_cols
        d_height = 1.0 / (1.0 + (n_rows - 1) * math.sqrt(3) / 2.0)
        d = min(d_width, d_height)
        
        # Generate points
        pts = []
        for row in range(n_rows):
            y = -0.5 + d/2 + row * d * math.sqrt(3) / 2.0
            if row % 2 == 0:
                n_in_row = n_cols
                x_start = -0.5 + d/2
            else:
                # Offset row
                # If we shift by d/2, we might go out of bounds on the right if n_cols is same.
                # Standard hex packing: odd rows have n_cols-1 points if centered, or n_cols points if shifted.
                # Let's shift by d/2 and keep n_cols points, but check bounds.
                n_in_row = n_cols
                x_start = -0.5 + d
                
            for col in range(n_in_row):
                x = x_start + col * d
                # Check if inside [-0.5, 0.5]
                if -0.5 <= x <= 0.5:
                    pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) >= N:
            pts = pts[:N]
        else:
            # If we don't have enough, pad with random points? 
            # Or just skip this configuration if it's too sparse.
            # For a good start, we want exactly N.
            # If pts < N, this d is too small or grid too tight.
            # Let's try to add points or skip.
            # Actually, if d is too small, we have more points. If d is too big, fewer.
            # Our d estimation is for fitting.
            # If len(pts) < N, it means our d was too large for this grid shape?
            # Or the offset removed points?
            # Let's just use the first N if we have enough.
            if len(pts) == 0:
                continue
            
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_centers = pts.copy()
            
    if best_centers is None:
        # Fallback: square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Initial Hexagonal Layout
    centers = _generate_hex_grid()
    current_r = _compute_radius(centers)
    
    # 2. Optimization via Soft-Min Gradient Descent
    # We maximize the minimum distance.
    # Objective: -soft_min(distances, beta)
    # Gradient: derived from log-sum-exp
    
    def objective_and_grad(x, beta):
        c = x.reshape(-1, 2)
        
        # Wall constraints
        # Penalty for going out of bounds
        # We want to keep centers in [-0.5+eps, 0.5-eps]
        # Let's add a strong penalty if any center is outside.
        # But L-BFGS-B handles bounds. We'll use bounds.
        
        # Pairwise distances
        # Use pdist for speed, but we need gradients.
        # For N=200, computing all pairs is O(N^2) = 40k ops.
        # Gradient of soft min:
        # f(x) = (1/beta) * log(sum(exp(-beta * d_ij)))
        # df/dx_i = (1/beta) * sum_j [ exp(-beta*d_ij) * (-beta * dd_ij/dx_i) / sum(exp(-beta*d_ij)) ]
        #         = - sum_j [ w_ij * (x_i - x_j) / d_ij ]
        # where w_ij = exp(-beta*d_ij) / sum(exp(-beta*d_ij))
        
        # Vectorized computation of distances and gradients
        # This is O(N^2) memory if we store all diffs. 200x200x2 = 80k floats. Fine.
        
        # Compute all pairwise differences
        # D[i,j] = c[i] - c[j]
        # We only need i < j for pairs, but for gradient we need sum over all j != i.
        
        # Let's compute the full distance matrix for simplicity, then mask diagonal.
        # diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        # dists = np.linalg.norm(diff, axis=2)
        # np.fill_diagonal(dists, np.inf)
        
        # This might be slow in Python loop if not careful.
        # Let's use a loop for clarity and speed control, or vectorize.
        
        diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Soft min
        x_vals = -dists * beta
        x_max = np.max(x_vals)
        if not np.isfinite(x_max):
            # Fallback
            return 0.0, np.zeros_like(x)
        
        exp_vals = np.exp(x_vals - x_max)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp
        np.fill_diagonal(weights, 0.0)
        
        obj_val = (x_max + np.log(sum_exp)) / beta
        
        # Gradient
        # grad_i = - sum_j weights[i,j] * (c[i] - c[j]) / dists[i,j]
        # Note: dists[i,i] is inf, weights[i,i] is 0, so 0/inf is 0.
        # But we have inf in dists. 0 * inf is NaN.
        # So we must handle the division carefully.
        
        # Create a safe distance matrix for division
        safe_dists = dists.copy()
        np.fill_diagonal(safe_dists, 1.0) # Avoid division by zero/inf where weight is 0
        
        # grad = - sum_j weights[i,j] * diffs[i,j] / safe_dists[i,j]
        # diffs[i,j] is c[i]-c[j]
        
        # term[i,j] = weights[i,j] * diffs[i,j] / safe_dists[i,j]
        term = weights[:, :, np.newaxis] * diffs / safe_dists[:, :, np.newaxis]
        
        grad = -np.sum(term, axis=1)
        
        # Add wall gradient?
        # L-BFGS-B handles bounds by projecting gradient or using active set.
        # It's better to let L-BFGS-B handle the bounds via the 'bounds' parameter.
        # However, the soft-min objective doesn't penalize walls directly.
        # The radius is min(pair_min/2, wall_min).
        # If we just maximize pair_min, circles might push to walls and overlap them.
        # We need to include wall distance in the soft min.
        
        # Let's add wall distances to the set of distances to minimize.
        # Wall distances for point i:
        # w_i_x = 0.5 - c[i,0]  (distance to right wall)
        # w_i_y = 0.5 + c[i,0]  (distance to left wall)
        # w_i_x2 = 0.5 - c[i,1] (distance to top wall)
        # w_i_y2 = 0.5 + c[i,1] (distance to bottom wall)
        
        # We want to maximize min(all_pair_dists, all_wall_dists).
        # So we include wall distances in the 'dists' array for the soft min.
        
        # Let's re-implement objective_and_grad to include walls.
        
        return obj_val, grad

    def objective_and_grad_with_walls(x, beta):
        c = x.reshape(-1, 2)
        
        # Pairwise distances
        diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_pair = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists_pair, np.inf)
        
        # Wall distances
        # For each point i, 4 distances.
        # d_wall[i, 0] = 0.5 - c[i, 0]
        # d_wall[i, 1] = 0.5 + c[i, 0]
        # d_wall[i, 2] = 0.5 - c[i, 1]
        # d_wall[i, 3] = 0.5 + c[i, 1]
        
        d_wall = np.empty((N, 4))
        d_wall[:, 0] = 0.5 - c[:, 0]
        d_wall[:, 1] = 0.5 + c[:, 0]
        d_wall[:, 2] = 0.5 - c[:, 1]
        d_wall[:, 3] = 0.5 + c[:, 1]
        
        # Combine into a single list of distances for soft min
        # Total distances: N*(N-1)/2 pairs + 4*N walls
        # To compute soft min efficiently, we can just concatenate the flat arrays.
        
        # Flatten pair distances (upper triangle)
        triu_indices = np.triu_indices(N, k=1)
        dists_flat = dists_pair[triu_indices]
        
        # Flatten wall distances
        d_wall_flat = d_wall.flatten()
        
        # Combine
        all_dists = np.concatenate([dists_flat, d_wall_flat])
        
        # Soft min calculation
        x_vals = -all_dists * beta
        x_max = np.max(x_vals)
        if not np.isfinite(x_max):
             return 0.0, np.zeros_like(x)
        
        exp_vals = np.exp(x_vals - x_max)
        sum_exp = np.sum(exp_vals)
        weights_flat = exp_vals / sum_exp
        
        obj_val = (x_max + np.log(sum_exp)) / beta
        
        # Gradient calculation
        # This is trickier because the gradient of the wall part is simple,
        # but the pair part requires the sum over pairs.
        
        # Gradient from pairs
        # We need to map weights_flat back to pairs.
        # Let's compute the pair gradient separately.
        
        # Reconstruct weights for pairs
        weights_pair_flat = weights_flat[:len(dists_flat)]
        
        # We need the gradient of the pair part.
        # grad_pair_i = - sum_{j>i} w_ij * (c_i - c_j)/d_ij - sum_{j<i} w_ji * (c_i - c_j)/d_ij
        # Note w_ij is symmetric in the sense of contribution, but indices matter.
        # Actually, the soft min is over the set of values. The weight w_k corresponds to distance k.
        # For pair (i,j), the distance is d_ij. The weight is w_ij.
        # Contribution to grad_i: -w_ij * (c_i - c_j)/d_ij
        # Contribution to grad_j: -w_ij * (c_j - c_i)/d_ij = w_ij * (c_i - c_j)/d_ij
        
        # Let's compute the force vector for each pair.
        # Create a matrix of weights for pairs
        W_pair = np.zeros((N, N))
        # Map flat indices to 2D indices
        # This is slow. Let's use the full matrix approach for weights.
        
        # Alternative: Compute full weight matrix for pairs
        x_pair = -dists_pair * beta
        # Mask diagonal
        x_pair_flat = x_pair.flatten()
        # We already have x_vals which includes walls.
        # Let's just compute the soft min gradient using the full matrix logic for pairs
        # and add wall gradients analytically.
        
        # --- Pair Gradient ---
        # Recalculate weights for pairs only? No, the weights depend on ALL distances (pairs + walls).
        # So we must use the global weights.
        
        # Let's reconstruct the full pair weight matrix from weights_flat.
        # This is O(N^2) to fill.
        W_pair_mat = np.zeros((N, N))
        # We need to map triu_indices to weights_pair_flat
        # triu_indices[0] are rows, triu_indices[1] are cols
        W_pair_mat[triu_indices[0], triu_indices[1]] = weights_pair_flat
        W_pair_mat[triu_indices[1], triu_indices[0]] = weights_pair_flat
        
        # Now compute gradient
        # term[i,j] = W_pair_mat[i,j] * diffs[i,j] / dists_pair[i,j]
        safe_dists = dists_pair.copy()
        np.fill_diagonal(safe_dists, 1.0)
        
        term = W_pair_mat[:, :, np.newaxis] * diffs / safe_dists[:, :, np.newaxis]
        grad_pair = -np.sum(term, axis=1)
        
        # --- Wall Gradient ---
        # The wall distances are in the second part of weights_flat.
        weights_wall_flat = weights_flat[len(dists_flat):]
        
        # Reshape to (N, 4)
        weights_wall = weights_wall_flat.reshape(N, 4)
        
        # Gradient for each wall distance:
        # d/dx (0.5 - x) = -1
        # d/dx (0.5 + x) = +1
        # d/dy (0.5 - y) = -1
        # d/dy (0.5 + y) = +1
        
        grad_wall = np.zeros((N, 2))
        # For each point i:
        # grad_x += -w[i,0] * (-1) + w[i,1] * (1)  => w[i,0] + w[i,1] ? 
        # Wait. 
        # f = sum w_k * (-d_k) / beta ... no, f = logsumexp(-beta * d_k)/beta
        # df/dx = sum w_k * (-beta * dd_k/dx * (-1) / beta) = sum w_k * dd_k/dx
        # Let's re-verify the sign.
        # We are MINIMIZING -soft_min.
        # Objective function passed to minimize is:
        # def obj(x): return soft_min_neg(x)
        # soft_min_neg = logsumexp(-beta * d) / beta
        # d(soft_min_neg)/dx = (1/beta) * sum( exp(-beta*d) * (-beta * dd/dx) / sum(exp) )
        #                     = - sum( w * dd/dx )
        
        # So grad = - sum( w * dd/dx )
        
        # For pair (i,j): d_ij = |c_i - c_j|
        # dd_ij/dx_i = (c_i - x_j) / d_ij  (vector)
        # Contribution to grad_i: - w_ij * (c_i - c_j) / d_ij
        # This matches grad_pair calculation above.
        
        # For wall 0 (right): d = 0.5 - x. dd/dx = -1.
        # Contribution to grad_x: - w_0 * (-1) = w_0
        # For wall 1 (left): d = 0.5 + x. dd/dx = 1.
        # Contribution to grad_x: - w_1 * (1) = -w_1
        # Total grad_x from walls: w_0 - w_1
        # Wait, w_0 is weight for dist to right wall.
        # If x is close to 0.5, d_right is small, w_0 is large.
        # grad_x = w_0 - w_1.
        # If x=0.49, d_right=0.01, d_left=0.99. w_0 >> w_1. grad_x > 0.
        # We want to move x to the right? No, x is already close to right wall.
        # We want to INCREASE the distance.
        # The objective is to maximize the minimum distance.
        # So we want to minimize the "cost" which is -min_dist.
        # If min_dist is the right wall distance, we want to increase it.
        # Increasing d_right means decreasing x.
        # So gradient of cost should point towards decreasing x.
        # Cost C = -min_dist.
        # If min_dist = d_right = 0.5 - x.
        # C = -(0.5 - x) = x - 0.5.
        # dC/dx = 1.
        # So we should move in direction -grad = -1. i.e. decrease x.
        # My formula: grad_x = w_0 - w_1.
        # If w_0 is large, grad_x is positive.
        # Minimizer moves in direction -grad. So -positive = negative. Decrease x.
        # Correct.
        
        grad_wall[:, 0] = weights_wall[:, 0] - weights_wall[:, 1]
        grad_wall[:, 1] = weights_wall[:, 2] - weights_wall[:, 3]
        
        total_grad = grad_pair + grad_wall
        
        return obj_val, total_grad

    # Optimization Loop
    bounds = [(-0.5, 0.5)] * (2 * N)
    x0 = centers.flatten()
    
    # Start with low beta (smooth) and increase (sharp)
    betas = [5, 10, 20, 50, 100, 200, 500]
    
    current_x = x0.copy()
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
            
        try:
            result = minimize(
                objective_and_grad_with_walls,
                current_x,
                args=(beta,),
                method='L-BFGS-B',
                jac=True, # We provide gradient
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < objective_and_grad_with_walls(current_x, beta)[0]:
                current_x = result.x
                # Update centers
                centers = current_x.reshape(-1, 2)
                current_r = _compute_radius(centers)
                
        except Exception as e:
            pass

    # Final polish with coordinate ascent if time permits
    if time.time() < deadline - 5:
        centers = current_x.reshape(-1, 2)
        # Simple coordinate ascent to fix any numerical drift
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                orig = centers[i].copy()
                best_pos = orig
                best_r = _compute_radius(centers)
                
                # Try small moves
                for dx in [-0.001, 0, 0.001]:
                    for dy in [-0.001, 0, 0.001]:
                        if dx == 0 and dy == 0: continue
                        new_c = orig + np.array([dx, dy])
                        if new_c[0] < -0.5 or new_c[0] > 0.5 or new_c[1] < -0.5 or new_c[1] > 0.5:
                            continue
                        centers[i] = new_c
                        r = _compute_radius(centers)
                        if r > best_r:
                            best_r = r
                            best_pos = new_c.copy()
                if not np.allclose(best_pos, orig):
                    centers[i] = best_pos
                    improved = True
            if not improved:
                break
    
    final_centers = current_x.reshape(-1, 2)
    
    return final_centers

# EVOLVE-BLOCK-END
