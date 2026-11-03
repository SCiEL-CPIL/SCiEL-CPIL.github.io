import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def compute_radius(centers):
        """
        Computes the exact maximum radius for the given centers.
        Returns r.
        """
        if len(centers) < 2:
            return 0.5
        
        # Wall distances
        dx_min = centers[:, 0] + 0.5
        dx_max = 0.5 - centers[:, 0]
        dy_min = centers[:, 1] + 0.5
        dy_max = 0.5 - centers[:, 1]
        
        min_wall_dist = np.min([np.min(dx_min), np.min(dx_max), np.min(dy_min), np.min(dy_max)])
        
        # Pairwise distances
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair_dist = np.min(dist_matrix)
        
        r = min(min_pair_dist / 2.0, min_wall_dist)
        return r

    def soft_min_obj_and_grad(x, sharpness):
        """
        Objective: -soft_min(dists)
        Gradients: -grad(soft_min(dists))
        
        x is flat array of size 2*N.
        sharpness controls how close soft_min is to true min.
        """
        centers = x.reshape(-1, 2)
        
        # 1. Pairwise distances
        # Compute all pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Soft min of pairwise distances (we want to maximize the min distance, 
        # so we minimize the negative soft min)
        # soft_min(x) = -1/b * log(sum(exp(-b*x_i)))
        # To avoid overflow, use log-sum-exp trick:
        # soft_min = -1/b * (min_val + log(sum(exp(-b*(x_i - min_val)))))
        # Actually, standard log-sum-exp is for max. For min:
        # min(x) = -max(-x)
        # max(-x) = 1/b * log(sum(exp(b*(-x_i))))
        # So soft_min(x) = -1/b * log(sum(exp(-b*x_i)))
        
        # Let's compute soft_min of dists
        # We want to maximize min(dists). So minimize -soft_min(dists).
        # soft_min(dists) approximates min(dists).
        
        # To handle the "wall" constraints similarly, we can include them in the set of distances.
        # Let's create a combined array of all constraints:
        # D = [all pairwise dists, 2*wall_dists]
        # Why 2*wall_dists? Because r = min(pair_dist/2, wall_dist).
        # So 2r = min(pair_dist, 2*wall_dist).
        # Maximizing r is equivalent to maximizing 2r.
        # So we can just maximize soft_min of [pair_dists, 2*wall_dists].
        
        # Wall distances
        wall_dists = np.concatenate([
            centers[:, 0] + 0.5,
            0.5 - centers[:, 0],
            centers[:, 1] + 0.5,
            0.5 - centers[:, 1]
        ])
        # We need to double wall distances to match the scale of pair distances in the min
        # r = min( min_pair_dist / 2, min_wall_dist )
        # 2r = min( min_pair_dist, 2 * min_wall_dist )
        
        # So let's define the vector of "effective distances"
        # For each pair (i,j), dist_ij
        # For each point i and wall k, 2 * dist_ik
        
        # This is a large vector. N=200, pairs ~ 20k. Walls 4*200=800. Total ~ 21k.
        # exp(-b * x) might be heavy but doable.
        
        # Let's build the vector
        # Upper triangle indices for pairs
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_dists = dists[i_idx, j_idx]
        
        # Double wall distances
        double_wall_dists = 2.0 * wall_dists
        
        # Combine
        all_dists = np.concatenate([pair_dists, double_wall_dists])
        
        # Compute soft min
        # To avoid overflow in exp, shift by min
        min_d = np.min(all_dists)
        # soft_min = -1/b * log(sum(exp(-b*(d - min_d)))) - min_d
        # Wait, soft_min(x) = -1/b * log(sum(exp(-b*x_i)))
        # = -1/b * log(sum(exp(-b*(x_i - min_d)) * exp(-b*min_d)))
        # = -1/b * (log(sum(exp(-b*(x_i - min_d)))) + exp(-b*min_d))
        # = -1/b * log(sum(exp(-b*(x_i - min_d)))) - min_d
        
        # Let's use this stable form
        shifted = all_dists - min_d
        # exp(-b * shifted)
        # If b is large, this can underflow for large shifted, which is fine (contribution 0)
        # But if b is small, it's fine.
        
        # We want to maximize soft_min.
        # obj = -soft_min
        # grad_obj = -grad_soft_min
        
        # Let's compute log_sum_exp_term = log(sum(exp(-b * shifted)))
        # Use np.logaddexp.reduce or similar?
        # sum_exp = np.sum(np.exp(-sharpness * shifted))
        # log_sum_exp = np.log(sum_exp)
        
        # However, if sharpness is large, -sharpness * shifted can be very negative, leading to 0.
        # This is numerically stable.
        
        sum_exp = np.sum(np.exp(-sharpness * shifted))
        if sum_exp == 0:
            # Fallback to hard min
            soft_min_val = min_d
            # Gradient is undefined or sparse, use subgradient
            # For now, return a large penalty or zero grad?
            # Better: return the hard min and zero gradient (or approximate)
            # But we want smoothness.
            # If sum_exp is 0, it means one term dominates heavily.
            # Let's just use min_d.
            # Gradient: 0 for all except the min one?
            # Let's return a small gradient to allow movement.
            return -soft_min_val, np.zeros_like(x)
            
        log_sum_exp = np.log(sum_exp)
        soft_min_val = -1.0 / sharpness * log_sum_exp - min_d
        
        # Gradient of soft_min w.r.t. all_dists:
        # d(soft_min)/d(x_i) = exp(-b*(x_i - min_d)) / sum(exp(-b*(x_j - min_d)))
        # Note: min_d depends on x, but in the derivative of the log-sum-exp form, 
        # the derivative of -1/b * log(sum(exp(-b*x_i))) is -exp(-b*x_i)/sum(exp(-b*x_j))
        # Wait, let's re-derive.
        # f(x) = -1/b * log(sum(exp(-b*x_i)))
        # df/dx_k = -1/b * (1/sum) * (-b * exp(-b*x_k)) = exp(-b*x_k) / sum(exp(-b*x_i))
        
        # So grad_soft_min_wrt_dists = exp(-b * all_dists) / sum(exp(-b * all_dists))
        # Using the shifted version:
        # grad = exp(-b * (all_dists - min_d)) / sum(exp(-b * (all_dists - min_d)))
        
        weights = np.exp(-sharpness * shifted) / sum_exp
        
        # Now we need to map these gradients back to x (centers).
        # x has 2*N elements.
        grad_x = np.zeros_like(x)
        
        # 1. Pairwise distances
        # dist_ij = ||c_i - c_j||
        # d(dist_ij)/d(c_i) = (c_i - c_j) / dist_ij
        # d(dist_ij)/d(c_j) = (c_j - c_i) / dist_ij
        
        # We have weights for each pair.
        # Let's iterate or vectorize.
        # Vectorization is tricky with indices.
        # For N=200, 20k pairs. Looping might be slow in Python.
        # Let's try to vectorize using scatter-add.
        
        # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        # We already have diff.
        # grad_dist_ij_wrt_ci = (c_i - c_j) / dist_ij
        # grad_dist_ij_wrt_cj = (c_j - c_i) / dist_ij
        
        # We can compute the gradient matrix G of shape (N, N, 2)
        # G[i, j] = grad of dist_ij w.r.t c_i
        # G[j, i] = grad of dist_ij w.r.t c_j
        
        # dists is (N, N)
        # diff is (N, N, 2)
        # Avoid division by zero
        safe_dists = np.where(dists > 1e-8, dists, 1.0)
        
        # grad_wrt_i = diff / safe_dists[:, :, np.newaxis]
        # This gives (N, N, 2) where element [i, j] is grad of dist_ij wrt c_i
        
        grad_wrt_i = diff / safe_dists[:, :, np.newaxis]
        
        # Now, for each pair (i, j) with i < j, we have a weight w_ij.
        # We add w_ij * grad_wrt_i[i, j] to grad_x[i]
        # And w_ij * grad_wrt_i[j, i] to grad_x[j]
        
        # Let's use np.add.at for accumulation
        # indices for i and j
        # We have i_idx, j_idx for pairs
        
        # Accumulate for i
        np.add.at(grad_x, i_idx, weights * grad_wrt_i[i_idx, j_idx])
        # Accumulate for j
        # grad_wrt_i[j_idx, i_idx] is grad of dist_ji wrt c_j, which is same as grad of dist_ij wrt c_j
        np.add.at(grad_x, j_idx, weights * grad_wrt_i[j_idx, i_idx])
        
        # 2. Wall distances
        # We have double_wall_dists = 2 * wall_dists
        # Let's say wall_dists array corresponds to specific constraints.
        # Order in wall_dists:
        # 0..N-1: x + 0.5 (left wall)
        # N..2N-1: 0.5 - x (right wall)
        # 2N..3N-1: y + 0.5 (bottom wall)
        # 3N..4N-1: 0.5 - y (top wall)
        
        # Weights for walls are the last 4*N elements of `weights`
        wall_weights = weights[-4*N:]
        
        # Left wall: dist = x_i + 0.5. d(dist)/d(x_i) = 1.
        # But we used 2*dist. So d(2*dist)/d(x_i) = 2.
        # The weight is for the term 2*dist.
        # So contribution to grad_x[i, 0] is wall_weights[0:N] * 2.0
        
        # Right wall: dist = 0.5 - x_i. d(dist)/d(x_i) = -1.
        # d(2*dist)/d(x_i) = -2.
        # Contribution: wall_weights[N:2N] * -2.0
        
        # Bottom wall: dist = y_i + 0.5. d(2*dist)/d(y_i) = 2.
        # Contribution: wall_weights[2N:3N] * 2.0 (to y component)
        
        # Top wall: dist = 0.5 - y_i. d(2*dist)/d(y_i) = -2.
        # Contribution: wall_weights[3N:4N] * -2.0 (to y component)
        
        # Map back to x (flat array)
        # x[2*i] is x_i, x[2*i+1] is y_i
        
        # Left
        grad_x[0::2] += wall_weights[0:N] * 2.0
        # Right
        grad_x[0::2] += wall_weights[N:2*N] * -2.0
        # Bottom
        grad_x[1::2] += wall_weights[2*N:3*N] * 2.0
        # Top
        grad_x[1::2] += wall_weights[3*N:4*N] * -2.0
        
        # Total gradient of objective (which is -soft_min)
        # grad_obj = -grad_soft_min
        # So we negate grad_x
        grad_x = -grad_x
        
        return -soft_min_val, grad_x

    def initialize_hex_grid():
        """
        Generates a high-quality initial hexagonal packing.
        """
        d = math.sqrt(2.0 / (N * math.sqrt(3)))
        
        pts = []
        row = 0
        y = -0.5
        while y <= 0.5:
            x_offset = 0.0 if row % 2 == 0 else d / 2.0
            x = -0.5 + d / 2.0 + x_offset
            while x <= 0.5:
                if -0.5 < x < 0.5 and -0.5 < y < 0.5:
                    pts.append([x, y])
                x += d
            y += d * math.sqrt(3) / 2.0
            row += 1
            
        pts = np.array(pts)
        
        if len(pts) < N:
            m = int(math.ceil(math.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            return grid[:N]
        
        dists_to_origin = np.linalg.norm(pts, axis=1)
        sorted_idx = np.argsort(dists_to_origin)
        pts = pts[sorted_idx[:N]]
        
        return pts

    def initialize_random():
        return np.random.uniform(-0.4, 0.4, (N, 2))

    # Optimization parameters
    # We will try multiple starts and different sharpness values.
    # Sharpness (b) controls the "softness".
    # Small b -> very smooth, global but loose.
    # Large b -> sharp, local but precise.
    
    best_r = 0.0
    best_centers = None
    
    # Strategy:
    # 1. Start with a coarse optimization (low sharpness) to find a good basin.
    # 2. Refine with higher sharpness.
    # 3. Use multiple random starts if time permits.
    
    sharpness_schedule = [50, 200, 500, 1000, 2000]
    
    # Initial starts: Hex grid + 2 random
    starts = [initialize_hex_grid(), initialize_random(), initialize_random()]
    
    # Time budget per start/sharpness combo
    # Total time ~ 145s
    # 3 starts * 5 sharpness = 15 optimizations.
    # ~9s each.
    
    for start_idx, start_centers in enumerate(starts):
        if time.time() > deadline - 10.0:
            break
            
        current_centers = start_centers.copy()
        
        for sharpness in sharpness_schedule:
            if time.time() > deadline - 5.0:
                break
                
            # Flatten
            x0 = current_centers.flatten()
            
            # Bounds: centers must be in [-0.5, 0.5]
            # Actually, to ensure non-overlap and inside, we can use bounds.
            # But the objective handles the radius.
            # If we use bounds [-0.5, 0.5], L-BFGS-B will keep them inside.
            # However, if they are on the boundary, the gradient might push them out,
            # but L-BFGS-B projects back.
            # Let's use tight bounds to prevent escaping the square.
            bounds = [(-0.5, 0.5)] * (2 * N)
            
            # Run L-BFGS-B
            # Max fun evals to stay within time
            max_iter = 50
            max_fun = 200
            
            try:
                res = minimize(
                    soft_min_obj_and_grad,
                    x0,
                    args=(sharpness,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': max_iter, 'maxfun': max_fun, 'disp': False}
                )
                
                if res.success or res.fun < 0: # res.fun is negative soft_min
                    new_centers = res.x.reshape(-1, 2)
                    # Ensure finite
                    if np.all(np.isfinite(new_centers)):
                        r = compute_radius(new_centers)
                        if r > best_r:
                            best_r = r
                            best_centers = new_centers.copy()
                            current_centers = new_centers.copy()
                        else:
                            # Even if not better, use it for next refinement?
                            # Usually yes, if it's a valid point.
                            current_centers = new_centers.copy()
            except Exception:
                pass

    # If no valid result, fallback
    if best_centers is None:
        best_centers = initialize_hex_grid()
        best_r = compute_radius(best_centers)
        
    # Final polish with very high sharpness if time remains
    if time.time() < deadline - 10.0:
        x0 = best_centers.flatten()
        bounds = [(-0.5, 0.5)] * (2 * N)
        try:
            res = minimize(
                soft_min_obj_and_grad,
                x0,
                args=(5000,),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 100, 'maxfun': 500, 'disp': False}
            )
            if res.success:
                new_centers = res.x.reshape(-1, 2)
                if np.all(np.isfinite(new_centers)):
                    r = compute_radius(new_centers)
                    if r > best_r:
                        best_r = r
                        best_centers = new_centers.copy()
        except Exception:
            pass
            
    return best_centers

# EVOLVE-BLOCK-END
