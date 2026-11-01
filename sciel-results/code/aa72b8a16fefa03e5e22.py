import numpy as np
from scipy.optimize import minimize

def construct():
    N = 50
    
    def get_score(centers):
        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Wall distances
        abs_centers = np.abs(centers)
        max_coord = np.max(abs_centers, axis=1)
        wall_dists = 0.5 - max_coord
        
        min_pair_dist = np.min(dist[~np.eye(N, dtype=bool)])
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    def objective(centers_flat):
        centers = centers_flat.reshape(N, 3)
        # We want to maximize the radius, so minimize negative radius.
        # To help the optimizer, we can add a small regularization or just use the min.
        # The min function is non-differentiable, but the gradient is zero almost everywhere 
        # except at the minimum. This can stall gradient-based methods.
        # A common trick is to use a soft-min.
        # Let's try minimizing the negative of the sum of a power of the distances, 
        # or just the negative min and hope L-BFGS-B handles it.
        # Actually, L-BFGS-B is a gradient-based method. If the gradient is 0, it stops.
        # We need a smooth objective.
        # Let's use the "soft min" of the distances.
        # d_i are the constraints (pair dists / 2 and wall dists).
        # We want max(min(d_i)).
        # Equivalent to min(-min(d_i)).
        # Soft min: -1/beta * log(sum(exp(-beta * d_i)))
        # This is smooth and approximates min(d_i) for large beta.
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        # Exclude diagonal
        mask = ~np.eye(N, dtype=bool)
        pair_dists = dist[mask] / 2.0
        
        abs_centers = np.abs(centers)
        max_coord = np.max(abs_centers, axis=1)
        wall_dists = 0.5 - max_coord
        
        # Combine all distances
        all_dists = np.concatenate([pair_dists, wall_dists])
        
        # Soft min parameter
        beta = 100.0
        # Prevent overflow in exp
        min_d = np.min(all_dists)
        shifted = all_dists - min_d
        soft_min = min_d - (1.0 / beta) * np.log(np.sum(np.exp(-beta * shifted)))
        
        return -soft_min

    def gradient(centers_flat):
        centers = centers_flat.reshape(N, 3)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        mask = ~np.eye(N, dtype=bool)
        pair_dists = dist[mask] / 2.0
        
        abs_centers = np.abs(centers)
        max_coord = np.max(abs_centers, axis=1)
        wall_dists = 0.5 - max_coord
        
        all_dists = np.concatenate([pair_dists, wall_dists])
        
        beta = 100.0
        min_d = np.min(all_dists)
        shifted = all_dists - min_d
        weights = np.exp(-beta * shifted)
        weights /= np.sum(weights)
        
        grad = np.zeros_like(centers)
        
        # Pair gradients
        # d(soft_min)/d(centers[i]) = sum_j weights_{ij} * d(pair_dist_{ij}/2)/d(centers[i])
        # d(||c_i - c_j||/2)/d(c_i) = (c_i - c_j) / ||c_i - c_j||
        
        # This is computationally expensive to do analytically with soft min.
        # Let's just use numerical gradient or a simpler objective.
        # Given the time limit, let's stick to a simpler approach:
        # Maximize the sum of squared distances? No, that's not the score.
        
        # Alternative: Use the current gradient ascent but with better parameters and restarts.
        # The current code does this. Let's improve the current code's optimization loop.
        pass

    # Actually, let's stick to the current algorithm structure but make it more robust.
    # 1. Better initialization: Random points in the cube, then optimize.
    # 2. Better optimization: Use the existing gradient logic but with adaptive step size and restarts.
    
    best_centers = None
    best_score = -1.0
    
    np.random.seed(42) # Deterministic
    
    for restart in range(5):
        # Initialization
        if restart == 0:
            # Grid init
            ticks = np.array([-0.375, -0.125, 0.125, 0.375])
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            idx = np.random.permutation(64)[:50]
            centers = grid[idx].copy()
        else:
            # Random init
            centers = np.random.uniform(-0.45, 0.45, (N, 3))
        
        # Optimization
        max_iter = 3000
        step_size = 0.01
        delta_active = 0.01
        
        for i in range(max_iter):
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            
            abs_centers = np.abs(centers)
            max_coord = np.max(abs_centers, axis=1)
            wall_dists = 0.5 - max_coord
            
            min_pair_dist = np.min(dist[~np.eye(N, dtype=bool)])
            min_wall_dist = np.min(wall_dists)
            current_r = min(min_pair_dist / 2.0, min_wall_dist)
            
            if i % 100 == 0:
                if current_r > best_score:
                    best_score = current_r
                    best_centers = centers.copy()
            
            active_pairs = np.argwhere(dist < 2 * (current_r + delta_active))
            active_pairs = active_pairs[active_pairs[:, 0] < active_pairs[:, 1]]
            active_walls = np.where(wall_dists < current_r + delta_active)[0]
            
            grad = np.zeros_like(centers)
            
            for i_idx, j_idx in active_pairs:
                v = centers[i_idx] - centers[j_idx]
                norm_v = np.linalg.norm(v)
                if norm_v > 1e-9:
                    v_norm = v / norm_v
                    grad[i_idx] += v_norm
                    grad[j_idx] -= v_norm
            
            for i_idx in active_walls:
                limiting_coords = np.argmax(abs_centers[i_idx])
                direction = -np.sign(centers[i_idx, limiting_coords])
                grad[i_idx, limiting_coords] += direction
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 0:
                grad /= grad_norm
            
            centers += step_size * grad
            centers = np.clip(centers, -0.5, 0.5)
            
            step_size *= 0.999
            delta_active *= 0.999
            
    return best_centers
