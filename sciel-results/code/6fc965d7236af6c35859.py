"""
Strategy: Deterministic Gradient-Based Optimization using L-BFGS-B.
Objective: Maximize the soft-minimum of pairwise distances and wall distances.
This approximates the hard constraint min(distance) with a smooth function,
allowing the use of gradient information to escape shallow local minima more
effectively than stochastic local search.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # 1. Initialization: Hexagonal Grid
    # A hexagonal grid is a known good initial guess for circle packing.
    def initialize_hex_grid():
        # Target radius is approx 0.0366. Hex spacing d = 2*r.
        # We want roughly 200 points in area 1.
        # Area of hex cell = d^2 * sqrt(3)/2.
        # 200 * d^2 * sqrt(3)/2 = 1 => d = sqrt(2/(200*sqrt(3)))
        d = math.sqrt(2.0 / (N * math.sqrt(3)))
        
        pts = []
        row = 0
        # Start from center to ensure symmetry and full coverage
        # Generate a grid larger than needed, then crop
        y = 0.0
        while y <= 0.5:
            x_offset = 0.0 if row % 2 == 0 else d / 2.0
            x = -0.5 + d / 2.0 + x_offset
            while x <= 0.5:
                pts.append([x, y])
                x += d
            y += d * math.sqrt(3) / 2.0
            row += 1
            
        # Also generate negative y part
        y = -d * math.sqrt(3) / 2.0
        while y >= -0.5:
            x_offset = 0.0 if row % 2 == 0 else d / 2.0
            x = -0.5 + d / 2.0 + x_offset
            while x <= 0.5:
                pts.append([x, y])
                x += d
            y -= d * math.sqrt(3) / 2.0
            row += 1
            
        pts = np.array(pts)
        
        # Filter points strictly inside [-0.5, 0.5]
        mask = (pts[:, 0] > -0.5) & (pts[:, 0] < 0.5) & (pts[:, 1] > -0.5) & (pts[:, 1] < 0.5)
        pts = pts[mask]
        
        if len(pts) < N:
            # Fallback: Square grid if hex generation fails or is sparse
            m = int(math.ceil(math.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            return grid[:N]
        
        # If we have more than N points, select N closest to center to maintain compactness
        # Or just take the first N? Centering is better for packing.
        # Sort by distance to origin
        dists_to_origin = np.linalg.norm(pts, axis=1)
        sorted_idx = np.argsort(dists_to_origin)
        pts = pts[sorted_idx[:N]]
        
        return pts

    def compute_radius(centers):
        """Compute the exact maximum radius for given centers."""
        if len(centers) < 2:
            return 0.5
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair = np.min(dist_matrix)
        return min(min_wall, min_pair / 2.0)

    def objective_and_grad(flat_centers, beta):
        """
        Objective: -soft_min(distances)
        soft_min(x) = -1/beta * log(sum(exp(-beta * x_i)))
        We want to maximize min(dist), so we minimize -min(dist).
        Using soft-min: minimize -soft_min(dist) = 1/beta * log(sum(exp(-beta * dist)))
        
        Actually, let's define the objective to MINIMIZE the "badness".
        Badness = -min(dists).
        Approximation: f(x) = (1/beta) * log( sum_{pairs} exp(-beta * d_ij) + sum_{walls} exp(-beta * w_i) )
        This is convex? No, d_ij is non-convex. But it's smooth.
        
        Gradient of f(x):
        Let S = sum_{k} exp(-beta * d_k)
        f = (1/beta) * log(S)
        df/dx = (1/beta) * (1/S) * sum_k (-beta * exp(-beta * d_k) * dd_k/dx)
              = -(1/S) * sum_k exp(-beta * d_k) * dd_k/dx
              = - sum_k p_k * dd_k/dx
        where p_k = exp(-beta * d_k) / S are softmax probabilities.
        """
        centers = flat_centers.reshape(-1, 2)
        n = len(centers)
        
        # Pairwise distances
        # For memory efficiency with N=200, cdist is fine (200x200)
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(centers)
        # Flatten wall distances: for each point, distance to x_min, x_max, y_min, y_max
        # w_i = min(0.5 - x_i, 0.5 + x_i, 0.5 - y_i, 0.5 + y_i)
        # Actually, 0.5 - |x_i| is the distance to the nearest vertical wall.
        # 0.5 - |y_i| is the distance to the nearest horizontal wall.
        # So for each point i, we have 2 wall constraints.
        
        # Combine all distances into a single vector for the soft-min calculation
        # We need to be careful with gradients.
        
        # Let's compute the soft-min value and gradient directly.
        
        # Pairwise terms
        # d_ij = ||c_i - c_j||
        # grad_{c_i} d_ij = (c_i - c_j) / d_ij
        # grad_{c_j} d_ij = (c_j - c_i) / d_ij
        
        # Wall terms
        # w_{i,x} = 0.5 - |x_i|
        # grad_{x_i} w_{i,x} = -sign(x_i)
        # w_{i,y} = 0.5 - |y_i|
        # grad_{y_i} w_{i,y} = -sign(y_i)
        
        # To avoid numerical issues with exp(-beta * d) when beta is large and d is small,
        # we can use the log-sum-exp trick, but for N=200, direct sum is okay if beta is not too huge.
        # Let's cap beta to avoid overflow.
        
        # Flatten distances
        # We will accumulate the gradient.
        grad = np.zeros_like(flat_centers)
        
        # Pairwise
        # Get upper triangle indices
        i_idx, j_idx = np.triu_indices(n, k=1)
        d_pairs = dist_matrix[i_idx, j_idx]
        
        # Weights for pairs
        # To prevent overflow in exp(-beta * d), we can shift by min_d
        # But for gradient, we need relative weights.
        # Let's compute exp_val = exp(-beta * (d - min_d))
        min_d_pair = np.min(d_pairs)
        exp_pairs = np.exp(-beta * (d_pairs - min_d_pair))
        sum_exp_pairs = np.sum(exp_pairs)
        
        # Wall distances
        w_x = 0.5 - np.abs(centers[:, 0])
        w_y = 0.5 - np.abs(centers[:, 1])
        w_all = np.concatenate([w_x, w_y])
        min_d_wall = np.min(w_all)
        
        # We need to combine pair and wall terms.
        # The soft-min is over the set of ALL distances (pairs and walls).
        # Let D be the vector of all distances.
        # f = (1/beta) log( sum exp(-beta D_k) )
        
        # To compute this efficiently:
        # Let M = min(D).
        # sum exp(-beta D_k) = exp(-beta M) * sum exp(-beta (D_k - M))
        # f = -M + (1/beta) log( sum exp(-beta (D_k - M)) )
        
        # Gradient:
        # df/dx = - sum p_k * d(D_k)/dx
        # where p_k = exp(-beta (D_k - M)) / sum_j exp(-beta (D_j - M))
        
        # We need to compute p_k for all k (pairs and walls).
        
        # Let's build the full vector of distances and their gradients?
        # No, let's just sum the contributions.
        
        # Shifted distances
        d_pairs_shifted = d_pairs - min_d_pair
        # Note: min_d_pair might not be the global min if a wall is closer.
        # Let global_min = min(min_d_pair, min_d_wall)
        global_min = min(min_d_pair, min_d_wall)
        
        # Recalculate shifted distances relative to global_min
        d_pairs_shifted = d_pairs - global_min
        w_x_shifted = w_x - global_min
        w_y_shifted = w_y - global_min
        
        exp_pairs = np.exp(-beta * d_pairs_shifted)
        exp_wx = np.exp(-beta * w_x_shifted)
        exp_wy = np.exp(-beta * w_y_shifted)
        
        sum_exp = np.sum(exp_pairs) + np.sum(exp_wx) + np.sum(exp_wy)
        
        # Objective value
        val = -global_min + (1.0 / beta) * np.log(sum_exp)
        
        # Gradient
        # grad = - sum p_k * grad(D_k)
        
        # Pair contributions
        # p_pair_k = exp_pairs[k] / sum_exp
        # grad(D_k) for pair (i,j) is:
        #   for c_i: (c_i - c_j) / d_k
        #   for c_j: (c_j - c_i) / d_k
        
        # We need to accumulate this into grad.
        # grad is shape (N, 2). flat_centers is (2N,).
        
        # Create a gradient array for centers
        grad_centers = np.zeros_like(centers)
        
        # For each pair (i, j)
        # vector = c_i - c_j
        # d = ||vector||
        # contribution to c_i: - p_k * (vector / d)
        # contribution to c_j: - p_k * (-vector / d) = p_k * (vector / d)
        
        # Vectorized:
        # vectors = centers[i_idx] - centers[j_idx]  # shape (M, 2)
        # inv_d = 1 / d_pairs  # shape (M,)
        # weights = exp_pairs / sum_exp  # shape (M,)
        # grad_contrib = vectors * (weights * inv_d)[:, np.newaxis]  # shape (M, 2)
        
        # np.add.at(grad_centers, i_idx, -grad_contrib)
        # np.add.at(grad_centers, j_idx, grad_contrib)
        
        vectors = centers[i_idx] - centers[j_idx]
        # Handle d=0 case (shouldn't happen with valid init, but safe)
        inv_d = np.where(d_pairs > 1e-10, 1.0 / d_pairs, 0.0)
        weights = exp_pairs / sum_exp
        grad_contrib = vectors * (weights * inv_d)[:, np.newaxis]
        
        np.add.at(grad_centers, i_idx, -grad_contrib)
        np.add.at(grad_centers, j_idx, grad_contrib)
        
        # Wall contributions
        # For w_x[i] = 0.5 - |x_i|
        # grad_{x_i} w_x[i] = -sign(x_i)
        # Contribution to grad_centers[i, 0]: - p_wx_i * (-sign(x_i)) = p_wx_i * sign(x_i)
        
        p_wx = exp_wx / sum_exp
        p_wy = exp_wy / sum_exp
        
        # sign(x)
        sign_x = np.sign(centers[:, 0])
        sign_y = np.sign(centers[:, 1])
        
        # If x=0, sign is 0, which is fine (gradient of abs is undefined but 0 is a valid subgradient)
        
        grad_centers[:, 0] += p_wx * sign_x
        grad_centers[:, 1] += p_wy * sign_y
        
        # Flatten gradient
        grad_flat = grad_centers.reshape(-1)
        
        return val, grad_flat

    # 2. Optimization Loop
    # We will run L-BFGS-B with an increasing beta schedule.
    # Start with low beta (smooth, easy to optimize) and increase beta (sharp, accurate).
    
    centers = initialize_hex_grid()
    current_r = compute_radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    # Beta schedule
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    # Time management
    time_per_beta = 15.0  # Seconds per beta step
    
    for beta in betas:
        if time.time() > deadline - 5.0:
            break
            
        # Define objective and gradient for this beta
        def obj_func(flat_c):
            val, grad = objective_and_grad(flat_c, beta)
            return val, grad
        
        # Run L-BFGS-B
        # Bounds: [-0.5, 0.5] for all coordinates
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            res = minimize(
                obj_func,
                centers.reshape(-1),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            new_centers = res.x.reshape(-1, 2)
            
            # Compute actual radius
            new_r = compute_radius(new_centers)
            
            if new_r > best_r:
                best_r = new_r
                best_centers = new_centers.copy()
            
            # Update current centers for next iteration
            # Even if not better than global best, it might be better than current for next beta
            # But usually, higher beta converges to the same or better local min.
            # Let's use the result if it's valid.
            if np.all(np.isfinite(new_centers)):
                centers = new_centers
                
        except Exception as e:
            print(f"Optimization failed at beta {beta}: {e}")
            # Keep current centers
            
    # Final polish: If we have time, do a few iterations of high-beta gradient steps
    # or just return the best found.
    
    # Ensure validity
    if not np.all(np.isfinite(best_centers)):
        # Fallback
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers

# EVOLVE-BLOCK-END
