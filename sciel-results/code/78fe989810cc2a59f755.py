import numpy as np
import time
import scipy.optimize as opt

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # 1. Generate initial hexagonal lattice
    d = 0.076
    points_list = []
    dy = d * np.sqrt(3) / 2
    n_rows = int(np.ceil(1.0 / dy)) + 2
    for j in range(-n_rows, n_rows + 1):
        y = j * dy
        if y < -0.55 or y > 0.55:
            continue
        x_offset = (j % 2) * d / 2
        n_cols = int(np.ceil(1.0 / d)) + 2
        for i in range(-n_cols, n_cols + 1):
            x = i * d + x_offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points_list.append([x, y])
    
    points = np.array(points_list)
    
    if len(points) > N:
        np.random.shuffle(points)
        points = points[:N]
    elif len(points) < N:
        remaining = N - len(points)
        new_pts = np.random.uniform(-0.45, 0.45, size=(remaining, 2))
        points = np.vstack([points, new_pts])
        
    points[:, 0] = np.clip(points[:, 0], -0.49, 0.49)
    points[:, 1] = np.clip(points[:, 1], -0.49, 0.49)
    
    def get_radius(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_c = np.min(dist) / 2.0
        
        dx = np.minimum(pts[:, 0] + 0.5, 0.5 - pts[:, 0])
        dy = np.minimum(pts[:, 1] + 0.5, 0.5 - pts[:, 1])
        min_b = np.min(np.minimum(dx, dy))
        
        return min(min_c, min_b)

    current_r = get_radius(points)
    best_points = points.copy()
    best_r = current_r

    # Soft-min objective and gradient
    def soft_min_obj(p_flat, beta):
        p = p_flat.reshape(N, 2)
        
        # Pairwise distances
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist = np.sqrt(dist_sq)
        np.fill_diagonal(dist, 0) # Diagonal is 0, but we will exclude it
        
        # Wall distances
        d_left = p[:, 0] + 0.5
        d_right = 0.5 - p[:, 0]
        d_bottom = p[:, 1] + 0.5
        d_top = 0.5 - p[:, 1]
        
        # Collect all distances
        # Pairwise: upper triangle
        i_idx, j_idx = np.triu_indices(N, k=1)
        d_pairs = dist[i_idx, j_idx]
        
        # Walls: 4 per point
        d_walls = np.vstack([d_left, d_right, d_bottom, d_top]).flatten()
        
        all_d = np.concatenate([d_pairs, d_walls])
        
        # Log-sum-exp for stability
        # We want min(all_d). 
        # softmin = -1/beta * log(sum(exp(-beta * d)))
        # log(sum(exp(-beta * d))) = -beta * max_d + log(sum(exp(-beta * (d - max_d))))
        max_d = np.max(all_d)
        shifted = all_d - max_d
        exp_shifted = np.exp(-beta * shifted)
        sum_exp = np.sum(exp_shifted)
        
        if sum_exp == 0:
            return 0.0, np.zeros_like(p_flat)
            
        log_sum = np.log(sum_exp)
        
        # Objective: maximize softmin -> minimize -softmin
        # softmin = max_d - 1/beta * log_sum
        # obj = -softmin = -max_d + 1/beta * log_sum
        
        # Gradient
        # d/dp ( -softmin ) = - d/dp (softmin)
        # d/dp (softmin) = 1/sum_exp * sum( -beta * exp(-beta*d_k) * d/dp d_k )
        #                 = -beta / sum_exp * sum( exp(-beta*d_k) * grad_d_k )
        # So d/dp (obj) = beta / sum_exp * sum( exp(-beta*d_k) * grad_d_k )
        
        weights = beta * exp_shifted / sum_exp
        
        grad = np.zeros_like(p)
        
        # Pairwise gradients
        # For pair (i, j) with distance d_ij:
        # grad_p_i d_ij = (p_i - p_j) / d_ij
        # grad_p_j d_ij = (p_j - p_i) / d_ij
        # Contribution to grad: w_ij * grad_d_ij
        
        d_pairs_safe = np.where(d_pairs > 1e-10, d_pairs, 1e-10)
        vecs = p[i_idx] - p[j_idx]
        unit_vecs = vecs / d_pairs_safe[:, np.newaxis]
        
        # w_pairs corresponds to d_pairs
        w_pairs = weights[:len(d_pairs)]
        
        # Add to grad
        # grad[i] += w * unit_vec
        # grad[j] -= w * unit_vec
        np.add.at(grad, i_idx, w_pairs[:, np.newaxis] * unit_vecs)
        np.add.at(grad, j_idx, -w_pairs[:, np.newaxis] * unit_vecs)
        
        # Wall gradients
        # d_left = p_x + 0.5 -> grad = (1, 0)
        # d_right = 0.5 - p_x -> grad = (-1, 0)
        # d_bottom = p_y + 0.5 -> grad = (0, 1)
        # d_top = 0.5 - p_y -> grad = (0, -1)
        
        w_walls = weights[len(d_pairs):]
        
        # w_walls is ordered: [d_left_0, d_right_0, d_bottom_0, d_top_0, d_left_1, ...]
        # Reshape to (N, 4)
        w_walls_reshaped = w_walls.reshape(N, 4)
        
        # Grad contribution for each point i:
        # w_left * (1, 0) + w_right * (-1, 0) + w_bottom * (0, 1) + w_top * (0, -1)
        grad[:, 0] += w_walls_reshaped[:, 0] - w_walls_reshaped[:, 1]
        grad[:, 1] += w_walls_reshaped[:, 2] - w_walls_reshaped[:, 3]
        
        return -max_d + (1.0 / beta) * log_sum, grad.flatten()

    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Annealing beta
    beta = 10.0
    
    while time.time() < deadline:
        p_flat = best_points.flatten()
        
        try:
            res = opt.minimize(
                soft_min_obj, 
                p_flat, 
                args=(beta,), 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10}
            )
            
            new_points = res.x.reshape(N, 2)
            new_actual_r = get_radius(new_points)
            
            if new_actual_r > best_r:
                best_r = new_actual_r
                best_points = new_points.copy()
            
            # Increase beta to sharpen the minimum
            beta *= 1.2
            if beta > 1000:
                beta = 1000
                # If beta is maxed and we are not improving, we might be stuck
                # But let's keep trying with high beta for precision
                
        except Exception:
            break
            
    return best_points

if __name__ == "__main__":
    pts = construct()
    # ... print logic ...
