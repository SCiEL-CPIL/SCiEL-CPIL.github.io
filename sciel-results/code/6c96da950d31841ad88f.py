import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = 0.0
    best_points = None
    
    # Helper to compute score
    def calc_score(points):
        # points: (N, 2)
        if len(points) != N:
            return -1
        # Check inside
        radii = np.linalg.norm(points, axis=1)
        if np.any(radii >= 1.0):
            return -1
        # Check overlaps
        # Compute pairwise distances
        # For N=300, 300x300 matrix is 90k elements, fine.
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Mask diagonal
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        min_boundary = np.min(1.0 - radii)
        
        r = min(min_pair_dist / 2.0, min_boundary)
        return r

    # Helper to generate hexagonal lattice
    def generate_lattice(spacing, rotation):
        k = int(1.0 / spacing) + 3
        points = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                # Rotate
                xr = x * np.cos(rotation) - y * np.sin(rotation)
                yr = x * np.sin(rotation) + y * np.cos(rotation)
                if np.hypot(xr, yr) < 1.0:
                    points.append([xr, yr])
        
        points = np.array(points)
        if len(points) < N:
            return None
            
        # Select N points
        # Strategy: Keep N closest to origin
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)
        return points[idx[:N]]

    # Optimization function
    def optimize(points, time_limit):
        points = points.copy()
        beta = 10.0
        eta = 0.01
        steps = 0
        max_steps = 2000
        
        # Pre-compute indices for pairs
        # We will use vectorized ops in the loop
        
        while time.time() < time_limit and steps < max_steps:
            # Compute distances
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            # Boundary distances
            radii = np.linalg.norm(points, axis=1)
            b_dists = 1.0 - radii
            
            # Values for soft min
            # Pairs: dists / 2
            # Boundary: b_dists
            
            # Flatten pair distances (upper triangle only to avoid double counting? 
            # The gradient formula sums over all k. If we sum over all i!=j, 
            # the gradient for c_i from pair (i,j) and (j,i) will be consistent?
            # Let's stick to i < j for pairs to be clean.
            
            i_idx, j_idx = np.triu_indices(N, k=1)
            pair_dists = dists[i_idx, j_idx] / 2.0
            
            # Combine all values
            # We need weights for each value
            # w_pair = exp(-beta * pair_dists)
            # w_bound = exp(-beta * b_dists)
            
            # To avoid underflow with large beta, we can subtract min
            all_vals = np.concatenate([pair_dists, b_dists])
            min_val = np.min(all_vals)
            # exp(-beta * (val - min_val)) is safe
            # But we need to normalize.
            
            # Let's compute log-sum-exp for stability
            # log_S = log( sum exp(-beta * v) )
            # = -beta * min_val + log( sum exp(-beta * (v - min_val)) )
            
            exp_pairs = np.exp(-beta * (pair_dists - min_val))
            exp_bounds = np.exp(-beta * (b_dists - min_val))
            
            sum_exp = np.sum(exp_pairs) + np.sum(exp_bounds)
            
            # Soft min value
            soft_min_val = min_val - np.log(sum_exp) / beta
            
            # Weights
            w_pairs = exp_pairs / sum_exp
            w_bounds = exp_bounds / sum_exp
            
            # Gradient
            grad = np.zeros_like(points)
            
            # Pair gradients
            # For pair (i, j), weight w.
            # grad_i += w * (c_i - c_j) / dist_ij
            # grad_j += w * (c_j - c_i) / dist_ij
            
            # dist_ij = 2 * pair_dists
            # We have dists matrix.
            # diff[i, j] = c_i - c_j
            
            # Vectorized update:
            # We need to scatter the gradients.
            # grad[i] += sum_j w_ij * diff[i,j] / dist[i,j]
            
            # Create a matrix of weights
            W_mat = np.zeros((N, N))
            W_mat[i_idx, j_idx] = w_pairs
            W_mat[j_idx, i_idx] = w_pairs
            
            # Avoid division by zero (dist is inf on diagonal, but W is 0 there)
            # dists has inf on diagonal.
            # We need dists for the denominator.
            # safe_dists = np.where(dists > 0, dists, 1.0)
            
            # Gradient term: W_mat * diff / safe_dists
            # diff is (N, N, 2)
            # dists is (N, N)
            
            # We want sum over j for each i.
            # grad = sum_j (W_mat[i,j] * diff[i,j] / dists[i,j])
            
            # Note: diff[i,j] = c_i - c_j.
            # The term is w_ij * (c_i - c_j) / d_ij.
            
            # Let's compute the vector field
            # term = W_mat[:, :, np.newaxis] * diff / dists[:, :, np.newaxis]
            # This will have division by inf on diagonal, but W is 0, so 0*inf is nan.
            # We must mask diagonal.
            
            mask = np.ones((N, N), dtype=bool)
            np.fill_diagonal(mask, False)
            
            safe_dists = np.where(mask, dists, 1.0)
            
            # Compute weighted differences
            # w_diff = W_mat[:, :, np.newaxis] * diff / safe_dists[:, :, np.newaxis]
            
            # Sum over axis 1 (j)
            # grad = np.sum(w_diff, axis=1)
            
            # Let's do it efficiently
            # W_mat is sparse-ish? No, dense.
            # N=300, 90k elements. Fine.
            
            w_diff = W_mat[:, :, np.newaxis] * diff / safe_dists[:, :, np.newaxis]
            grad_pairs = np.sum(w_diff, axis=1)
            
            # Boundary gradients
            # grad_i += w_bound_i * (-c_i / |c_i|)
            # b_dists = 1 - |c_i|
            # grad of b_dist wrt c_i is -c_i / |c_i|
            
            # w_bounds is (N,)
            # norm_c = radii
            # safe_norm = np.where(radii > 0, radii, 1.0)
            # grad_bounds = -w_bounds[:, np.newaxis] * points / safe_norm[:, np.newaxis]
            
            grad_bounds = np.zeros_like(points)
            mask_b = radii > 1e-6
            grad_bounds[mask_b] = -w_bounds[mask_b, np.newaxis] * points[mask_b] / radii[mask_b, np.newaxis]
            
            total_grad = grad_pairs + grad_bounds
            
            # Update
            points += eta * total_grad
            
            # Annealing
            beta *= 1.02
            eta *= 0.99
            
            steps += 1
            
        return points

    # Main loop: Try different initializations
    rotations = [0, np.pi/12, np.pi/6, np.pi/3]
    
    for rot in rotations:
        if time.time() > deadline - 30:
            break
            
        # Generate lattice
        # Start with spacing 0.09
        spacing = 0.09
        points = generate_lattice(spacing, rot)
        
        if points is None:
            continue
            
        # Optimize
        # Give this run a time budget
        run_deadline = min(deadline - 5.0, time.time() + 30.0)
        opt_points = optimize(points, run_deadline)
        
        # Polish
        # Do a few more steps with high beta
        if time.time() < deadline - 2.0:
             # Re-use optimize logic with fixed high beta?
             # Or just call optimize again with smaller eta and higher start beta?
             # Let's just rely on the annealing inside optimize.
             pass
             
        score = calc_score(opt_points)
        
        if score > best_score:
            best_score = score
            best_points = opt_points.copy()

    if best_points is None:
        # Fallback: Return a valid packing
        # Generate a simple grid
        spacing = 0.1
        points = generate_lattice(spacing, 0)
        if points is not None:
            return points
        else:
            # Should not happen
            return np.zeros((N, 2))

    return best_points
