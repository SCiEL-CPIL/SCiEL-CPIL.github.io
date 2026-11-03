"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    # Use a deterministic seed for reproducibility
    rng = np.random.RandomState(42)
    
    # Best result so far
    best_centers = None
    best_score = -np.inf
    
    # Number of restarts based on available time
    # Each optimization run takes roughly 5-15 seconds
    max_restarts = 8
    
    for restart in range(max_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Initialize with random positions inside the cube
        # Use a stratified initialization to spread points out
        centers = np.zeros((N, 3))
        
        # For better initialization, use a grid-like structure with random perturbation
        # 4x5x5 = 100 points
        grid_shape = (4, 5, 5)
        nx, ny, nz = grid_shape
        
        # Create grid positions
        xs = (np.arange(nx) + 0.5) / nx - 0.5
        ys = (np.arange(ny) + 0.5) / ny - 0.5
        zs = (np.arange(nz) + 0.5) / nz - 0.5
        
        # Create all grid points
        grid_points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
        
        # Add random perturbation
        perturbation_scale = 0.02
        centers = grid_points[:N] + rng.uniform(-perturbation_scale, perturbation_scale, size=(N, 3))
        
        # Clip to stay within bounds
        centers = np.clip(centers, -0.45, 0.45)
        
        # Define the objective function
        # We want to maximize the minimum distance, which is equivalent to minimizing a penalty
        # for close pairs and boundary violations
        
        def objective_and_grad(x):
            """
            Objective: minimize sum of squared violations
            - For each pair (i,j) with d_ij < 2r: penalty = (2r - d_ij)^2
            - For each point i with distance to boundary < r: penalty = (r - dist_to_boundary)^2
            
            We use a fixed r and minimize violations, then binary search on r.
            Actually, let's use a different approach: minimize a smooth approximation
            of the negative minimum distance.
            """
            c = x.reshape(N, 3)
            
            # Compute all pairwise distances
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(dist_sq, np.inf)
            dist = np.sqrt(dist_sq)
            
            # Get all pairwise distances (upper triangle)
            iu = np.triu_indices(N, k=1)
            pair_dists = dist[iu]
            
            # Boundary distances: distance from each point to nearest face
            # For a point (x,y,z), dist to boundary = min(0.5-|x|, 0.5-|y|, 0.5-|z|)
            abs_c = np.abs(c)
            boundary_dists = np.min(0.5 - abs_c, axis=1)
            
            # We want to maximize the minimum of all pair distances and boundary distances
            # Use a smooth approximation: -log(sum(exp(-beta * d))) / beta for large beta
            # This approximates -min(d)
            
            beta = 50.0  # sharpness parameter
            
            # For pair distances
            # We only care about distances below some threshold to save computation
            # But for gradient, we need all
            
            # Smooth min approximation using log-sum-exp
            # -min(d) ≈ (1/beta) * log(sum(exp(-beta*d)))
            
            # To avoid overflow, use the log-sum-exp trick
            def smooth_min_neg(d, beta):
                """Approximates -min(d) using log-sum-exp"""
                # d is array of distances
                # -min(d) = -max(-d)
                # max(-d) ≈ (1/beta) * log(sum(exp(-beta*d)))
                # So -min(d) ≈ -(1/beta) * log(sum(exp(-beta*d)))
                # = (1/beta) * log(1/sum(exp(-beta*d)))
                # = -(1/beta) * log(sum(exp(-beta*d)))
                
                # For numerical stability, subtract max
                neg_d = -d
                max_neg_d = np.max(neg_d)
                # max_neg_d is the largest of -d, which is -min(d)
                # log(sum(exp(-beta*d))) = max_neg_d*beta + log(sum(exp(-beta*d - max_neg_d*beta)))
                # = -beta*min(d) + log(sum(exp(beta*(min(d) - d))))
                
                # Simpler: just compute directly with care
                scaled = -beta * d
                max_scaled = np.max(scaled)
                if max_scaled > 500:
                    # Overflow protection
                    return -d[np.argmax(d)]  # fallback to exact min
                sum_exp = np.sum(np.exp(scaled - max_scaled))
                if sum_exp < 1e-300:
                    return -d[np.argmax(d)]
                log_sum = max_scaled + np.log(sum_exp)
                return -log_sum / beta
            
            # Pair distance contribution
            pair_contrib = smooth_min_neg(pair_dists, beta)
            
            # Boundary distance contribution
            boundary_contrib = smooth_min_neg(boundary_dists, beta)
            
            # Total objective: we want to maximize min(pair_min, boundary_min)
            # So minimize: max(-pair_min, -boundary_min)
            # Which is: -min(pair_min, boundary_min)
            
            # Actually, the score is min(pair_min, boundary_min)
            # So we want to maximize that, i.e., minimize -score
            # -score = -min(pair_min, boundary_min) = max(-pair_min, -boundary_min)
            
            # Using smooth max: max(a,b) ≈ (1/beta)*log(exp(beta*a) + exp(beta*b))
            a = -pair_contrib  # this is pair_min (positive)
            b = -boundary_contrib  # this is boundary_min (positive)
            
            # We want max(-a, -b) = -min(a, b)
            # smooth_max(-a, -b) ≈ (1/beta)*log(exp(-beta*a) + exp(-beta*b))
            
            # Let's just compute the score directly and use a penalty approach
            # Score = min(pair_min, boundary_min)
            # Objective to minimize = -score + penalty for violations
            
            pair_min = np.min(pair_dists)
            boundary_min = np.min(boundary_dists)
            score = min(pair_min, boundary_min)
            
            # Add a smooth penalty to encourage all distances to be above the current best
            # This creates a barrier effect
            penalty = 0.0
            for d in pair_dists:
                if d < score + 0.01:
                    penalty += (score + 0.01 - d) ** 2
            for d in boundary_dists:
                if d < score + 0.01:
                    penalty += (score + 0.01 - d) ** 2
            
            obj = -score + 0.1 * penalty
            
            # Gradient computation (numerical or analytical)
            # For simplicity, use numerical gradient via finite differences
            # But this is slow. Let's compute analytical gradient.
            
            # The gradient of -min(pair_dists) is complex. 
            # Let's use a different objective: sum of inverse distances with power
            # This is smooth and encourages all distances to be large
            
            return obj
        
        # Let's use a simpler, more efficient objective
        # Minimize: sum_{i<j} 1/d_ij^2 + sum_i 1/boundary_dist_i^2
        # This is smooth and has a well-defined gradient
        
        def objective(x):
            c = x.reshape(N, 3)
            
            # Pairwise distances
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(dist_sq, np.inf)
            
            # Avoid division by zero
            dist_sq = np.maximum(dist_sq, 1e-12)
            
            # Sum of inverse squared distances for all pairs
            iu = np.triu_indices(N, k=1)
            pair_obj = np.sum(1.0 / dist_sq[iu])
            
            # Boundary distances
            abs_c = np.abs(c)
            boundary_dists_sq = np.minimum(
                np.minimum((0.5 - c[:, 0]) ** 2, (0.5 + c[:, 0]) ** 2),
                np.minimum((0.5 - c[:, 1]) ** 2, (0.5 + c[:, 1]) ** 2)
            )
            boundary_dists_sq = np.minimum(boundary_dists_sq, (0.5 - c[:, 2]) ** 2)
            boundary_dists_sq = np.minimum(boundary_dists_sq, (0.5 + c[:, 2]) ** 2)
            boundary_dists_sq = np.maximum(boundary_dists_sq, 1e-12)
            
            boundary_obj = np.sum(1.0 / boundary_dists_sq)
            
            return pair_obj + boundary_obj
        
        def objective_grad(x):
            c = x.reshape(N, 3)
            grad = np.zeros_like(x)
            
            # Pairwise contribution
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(dist_sq, np.inf)
            dist_sq = np.maximum(dist_sq, 1e-12)
            
            # d/dx_i of sum_{i<j} 1/|x_i - x_j|^2
            # = sum_{j!=i} 2*(x_i - x_j)/|x_i - x_j|^4
            
            # For each i, grad_i = 2 * sum_{j!=i} (c_i - c_j) / dist_ij^4
            inv_dist_sq = 1.0 / dist_sq
            # Multiply diff by inv_dist_sq^2 (since d/dx of 1/r^2 = -2*(x_i-x_j)/r^4)
            # Actually: d/dx_i [1/|x_i-x_j|^2] = -2*(x_i-x_j)/|x_i-x_j|^4
            # So grad contribution from pair (i,j) to x_i is -2*(c_i-c_j)/dist_ij^4
            # And to x_j is +2*(c_i-c_j)/dist_ij^4 (by antisymmetry)
            
            # Vectorized: for each i, sum over j of (c_i - c_j) / dist_ij^4
            # diff[i,j] = c_i - c_j
            # We want sum_j diff[i,j] * inv_dist_sq[i,j]^2
            # Note: inv_dist_sq[i,j]^2 = 1/dist_ij^4
            
            weighted_diff = diff * (inv_dist_sq ** 2)[:, :, np.newaxis]
            # Set diagonal to 0
            weighted_diff[np.diag_indices(N)] = 0
            
            # grad from pairs: -2 * sum_j (c_i - c_j)/dist_ij^4 for each i
            pair_grad = -2.0 * np.sum(weighted_diff, axis=1)
            
            # Boundary contribution
            # d/dx of 1/(0.5 - x)^2 = 2/(0.5-x)^3 for x < 0.5
            # More carefully: for coordinate k, boundary_dist_k = min over faces
            # Let's compute gradient for each coordinate
            
            for i in range(N):
                for k in range(3):
                    # Distance to each of the 6 faces
                    d_left = 0.5 - c[i, k]   # face at -0.5
                    d_right = 0.5 + c[i, k]  # face at +0.5
                    d_min = min(d_left, d_right)
                    d_min_sq = max(d_min ** 2, 1e-12)
                    
                    # d/dx_k of 1/d_min^2
                    if d_left <= d_right:
                        # Nearest face is left, d = 0.5 - x
                        # d/dx of 1/(0.5-x)^2 = 2/(0.5-x)^3
                        grad[i, k] += 2.0 / (d_left ** 3)
                    else:
                        # Nearest face is right, d = 0.5 + x
                        # d/dx of 1/(0.5+x)^2 = -2/(0.5+x)^3
                        grad[i, k] -= 2.0 / (d_right ** 3)
            
            grad = pair_grad + grad.reshape(N, 3)
            return grad.ravel()
        
        # Use L-BFGS-B for optimization
        x0 = centers.ravel()
        
        try:
            result = minimize(
                objective,
                x0,
                jac=objective_grad,
                method='L-BFGS-B',
                bounds=[(-0.49, 0.49)] * (N * 3),
                options={
                    'maxiter': 500,
                    'maxfun': 1000,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'iprint': 0
                }
            )
            
            if result.success or result.fun < 1e10:
                new_centers = result.x.reshape(N, 3)
                # Compute score
                diff = new_centers[:, np.newaxis, :] - new_centers[np.newaxis, :]
                dist_sq = np.sum(diff ** 2, axis=2)
                np.fill_diagonal(dist_sq, np.inf)
                pair_min = np.sqrt(np.min(dist_sq))
                
                abs_c = np.abs(new_centers)
                boundary_dists = np.min(0.5 - abs_c, axis=1)
                boundary_min = np.min(boundary_dists)
                
                new_score = min(pair_min, boundary_min)
                
                if new_score > best_score:
                    best_score = new_score
                    best_centers = new_centers.copy()
                    
        except Exception as e:
            pass
        
        # Check time
        if time.time() > deadline - 10.0:
            break
    
    # If no valid result, fall back to grid
    if best_centers is None:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
