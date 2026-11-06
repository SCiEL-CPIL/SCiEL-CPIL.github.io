import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial.distance import pdist, squareform

N = 200
DEADLINE = 120.0  # Evaluator kills at 150s, keep margin

def _compute_radius(points):
    """Compute the radius of the packing given centers."""
    # points: (N, 2)
    # Center-center distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_center_dist = np.sqrt(np.min(dist_sq))
    
    # Wall distances
    # Square is [-0.5, 0.5]
    dx = 0.5 - np.abs(points[:, 0])
    dy = 0.5 - np.abs(points[:, 1])
    min_wall_dist = np.min(np.concatenate([dx, dy]))
    
    # Radius is half the minimum clear distance
    r = min(min_center_dist, min_wall_dist) / 2.0
    return r

def _generate_hex_grid(n):
    """Generate a hexagonal grid of n points centered in [-0.5, 0.5]^2."""
    # We want roughly n points.
    # Hex packing: area per point ~ d^2 * sqrt(3)/2
    # Total area 1. n * d^2 * sqrt(3)/2 ~ 1 => d ~ sqrt(2/(n*sqrt(3)))
    d = np.sqrt(2.0 / (n * np.sqrt(3)))
    
    dy = d * np.sqrt(3) / 2.0
    dx = d
    
    # Generate a grid slightly larger than needed
    n_rows = int(np.ceil(1.0 / dy)) + 2
    n_cols = int(np.ceil(1.0 / dx)) + 2
    
    points = []
    # Center the grid
    # y from -0.5 to 0.5
    # x offset alternates
    
    for j in range(-n_rows, n_rows + 1):
        y = j * dy
        if y < -0.5 or y > 0.5:
            continue
        x_offset = (j % 2) * dx / 2.0
        for i in range(-n_cols, n_cols + 1):
            x = i * dx + x_offset
            if -0.5 <= x <= 0.5:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have more than n, select the best subset?
    # For now, if more, we can just take the first n or random.
    # But random might break symmetry.
    # A better heuristic: keep points that are "central" or just take first n from a centered generation.
    # Since the generation is centered, the first n might be biased.
    # Let's sort by distance to center and take the n closest? No, that's not right for packing.
    # Actually, for hex grid, the "best" subset of size n is tricky.
    # However, if we generate exactly n points by adjusting d, it's better.
    # Let's try to adjust d to get closer to n.
    
    # Re-estimate d to get closer to n points
    # Number of points ~ Area / Area_per_point = 1 / (d^2 * sqrt(3)/2)
    # So d ~ sqrt(2 / (n * sqrt(3)))
    # This is what we did.
    
    # If we have too many, we can try to remove the ones furthest from the center?
    # Or just random sample.
    # Let's try to keep the grid structure as much as possible.
    # If len(points) > n:
    #    We can try to find a subgrid.
    #    Or just random.
    #    Random is fine for initialization, the optimizer will fix it.
    
    if len(points) > n:
        # Shuffle and take first n
        idx = np.random.choice(len(points), n, replace=False)
        points = points[idx]
    elif len(points) < n:
        # Add random points
        remaining = n - len(points)
        new_pts = np.random.uniform(-0.45, 0.45, size=(remaining, 2))
        points = np.vstack([points, new_pts])
        
    # Clip to valid range
    points[:, 0] = np.clip(points[:, 0], -0.49, 0.49)
    points[:, 1] = np.clip(points[:, 1], -0.49, 0.49)
    
    return points

def _optimize_repulsion(points, deadline):
    """
    Optimize the packing using a repulsion-based energy function.
    We minimize E = sum_{i<j} max(0, 2r - d_ij)^2 + sum_i max(0, r - b_i)^2
    where r is a target radius we try to increase.
    """
    current_r = _compute_radius(points)
    best_points = points.copy()
    best_r = current_r
    
    # We will try to increase r
    r = current_r
    step_r = 0.0005  # Initial step
    min_step_r = 1e-5
    max_iter = 10000
    
    # Bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Objective function for L-BFGS-B
    def objective_and_grad(p_flat, r_target):
        p = p_flat.reshape(N, 2)
        E = 0.0
        grad = np.zeros_like(p)
        
        # Boundary violations
        # dist to boundary
        dx = np.minimum(p[:, 0] + 0.5, 0.5 - p[:, 0])
        dy = np.minimum(p[:, 1] + 0.5, 0.5 - p[:, 1])
        b_dist = np.minimum(dx, dy)
        
        viol_b = np.maximum(0.0, r_target - b_dist)
        E += np.sum(viol_b**2)
        
        # Gradient for boundary
        for i in range(N):
            if viol_b[i] > 0:
                x, y = p[i]
                d_left = x + 0.5
                d_right = 0.5 - x
                d_bottom = y + 0.5
                d_top = 0.5 - y
                
                min_d = min(d_left, d_right, d_bottom, d_top)
                
                gx = 0.0
                gy = 0.0
                
                if min_d == d_left:
                    gx = 1.0
                elif min_d == d_right:
                    gx = -1.0
                elif min_d == d_bottom:
                    gy = 1.0
                elif min_d == d_top:
                    gy = -1.0
                
                grad[i, 0] += -2.0 * viol_b[i] * gx
                grad[i, 1] += -2.0 * viol_b[i] * gy

        # Pairwise violations
        # Use pdist for efficiency?
        # pdist returns condensed vector.
        # But we need gradient.
        # Let's stick to matrix for clarity, N=200 is small.
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # viol = max(0, 2*r - dist)
        viol_pairs = np.maximum(0.0, 2.0 * r_target - dist)
        
        # Sum of squares for E
        # Sum over i<j
        i_idx, j_idx = np.triu_indices(N, k=1)
        v = viol_pairs[i_idx, j_idx]
        d = dist[i_idx, j_idx]
        
        E += np.sum(v**2)
        
        # Gradient
        mask = v > 0
        if np.any(mask):
            i_m = i_idx[mask]
            j_m = j_idx[mask]
            v_m = v[mask]
            d_m = d[mask]
            
            d_safe = np.where(d_m > 1e-10, d_m, 1e-10)
            
            vec = p[i_m] - p[j_m]
            unit = vec / d_safe[:, np.newaxis]
            
            # dT/dp_i = -2 * v * (p_i - p_j) / d
            # dT/dp_j =  2 * v * (p_i - p_j) / d
            
            np.add.at(grad, i_m, -2.0 * v_m[:, np.newaxis] * unit)
            np.add.at(grad, j_m,  2.0 * v_m[:, np.newaxis] * unit)
            
        return E, grad.flatten()

    while time.time() < deadline:
        new_r = r + step_r
        
        p_flat = points.flatten()
        
        try:
            res = opt.minimize(
                objective_and_grad, 
                p_flat, 
                args=(new_r,), 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 50, 'gtol': 1e-6, 'ftol': 1e-8}
            )
            
            new_points = res.x.reshape(N, 2)
            new_actual_r = _compute_radius(new_points)
            
            if new_actual_r > best_r:
                best_r = new_actual_r
                best_points = new_points.copy()
                points = new_points
                r = new_actual_r
                # Increase step if we succeeded
                step_r = min(step_r * 1.05, 0.002)
            else:
                # Failed to improve
                # Reduce step
                step_r *= 0.5
                if step_r < min_step_r:
                    # Try a few more times or break
                    # Let's try random restart if stuck
                    # Or just break
                    break
        except Exception:
            break

    return best_points, best_r

def _bottleneck_local_opt(points, r, deadline):
    """
    Local optimization focusing on the bottleneck constraints.
    We identify the pairs and walls that are at the minimum distance.
    We try to move these points to increase the minimum distance.
    """
    # This is a simpler approach: just use the repulsion optimizer with a tighter tolerance
    # or a different method.
    # For now, let's just run the repulsion optimizer again with a smaller step.
    return _optimize_repulsion(points, deadline)

def _polish(points, deadline):
    """
    Final high-precision polish pass.
    Use Nelder-Mead or similar for fine tuning?
    Or just run L-BFGS-B with very tight tolerance.
    """
    current_r = _compute_radius(points)
    
    # Define a function that maximizes the minimum distance
    # This is a max-min problem.
    # We can use scipy.optimize.minimize with a negative objective?
    # Or just use the repulsion method with a very small step.
    
    # Let's try to use a direct optimization of the minimum distance.
    # f(p) = -min(distances)
    # This is non-smooth.
    # Smoothing: f(p) = -log(sum(exp(-distances^2 / T))) / N  (soft-min)
    
    T = 0.001  # Temperature for soft-min
    
    def soft_min_dist(p_flat):
        p = p_flat.reshape(N, 2)
        # Center distances
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        
        # Wall distances
        dx = np.minimum(p[:, 0] + 0.5, 0.5 - p[:, 0])
        dy = np.minimum(p[:, 1] + 0.5, 0.5 - p[:, 1])
        wall_dist_sq = np.minimum(dx**2, dy**2)
        
        # Combine all distances squared
        # We want to maximize the minimum distance.
        # So minimize -soft_min(distances)
        # soft_min(x) = -T * log(sum(exp(-x^2/T)))
        # We have dist_sq.
        # dist = sqrt(dist_sq)
        # exp(-dist^2/T)
        
        # To avoid overflow, subtract max
        # But dist_sq can be large.
        
        # Let's use the distances directly.
        dist_c = np.sqrt(dist_sq)
        dist_w = np.sqrt(wall_dist_sq)
        
        # Flatten all distances
        # dist_c has N*N elements (with inf on diagonal)
        # dist_w has N elements
        
        # Mask out inf
        mask_c = dist_c < np.inf
        dists_c = dist_c[mask_c]
        dists_w = dist_w
        
        all_dists = np.concatenate([dists_c, dists_w])
        
        # Soft min
        # m = min(all_dists)
        # soft_min = m - T * log(1/N * sum(exp(-(d-m)^2/T)))
        # This is more stable.
        
        m = np.min(all_dists)
        exp_terms = np.exp(-(all_dists - m)**2 / T)
        log_sum = np.log(np.mean(exp_terms))
        soft_min_val = m - T * log_sum
        
        return -soft_min_val
    
    # Gradient is complex. Let's skip analytic gradient and use Nelder-Mead?
    # Nelder-Mead is slow for 400 dimensions.
    
    # Alternative: use the repulsion optimizer with a very small step_r.
    # This is effectively what we did.
    
    # Let's try one more L-BFGS-B run with the current radius as target.
    r_target = current_r
    
    def objective_and_grad(p_flat, r_target):
        p = p_flat.reshape(N, 2)
        E = 0.0
        grad = np.zeros_like(p)
        
        # Boundary violations
        dx = np.minimum(p[:, 0] + 0.5, 0.5 - p[:, 0])
        dy = np.minimum(p[:, 1] + 0.5, 0.5 - p[:, 1])
        b_dist = np.minimum(dx, dy)
        
        viol_b = np.maximum(0.0, r_target - b_dist)
        E += np.sum(viol_b**2)
        
        for i in range(N):
            if viol_b[i] > 0:
                x, y = p[i]
                d_left = x + 0.5
                d_right = 0.5 - x
                d_bottom = y + 0.5
                d_top = 0.5 - y
                
                min_d = min(d_left, d_right, d_bottom, d_top)
                
                gx = 0.0
                gy = 0.0
                
                if min_d == d_left:
                    gx = 1.0
                elif min_d == d_right:
                    gx = -1.0
                elif min_d == d_bottom:
                    gy = 1.0
                elif min_d == d_top:
                    gy = -1.0
                
                grad[i, 0] += -2.0 * viol_b[i] * gx
                grad[i, 1] += -2.0 * viol_b[i] * gy

        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        viol_pairs = np.maximum(0.0, 2.0 * r_target - dist)
        
        i_idx, j_idx = np.triu_indices(N, k=1)
        v = viol_pairs[i_idx, j_idx]
        d = dist[i_idx, j_idx]
        
        E += np.sum(v**2)
        
        mask = v > 0
        if np.any(mask):
            i_m = i_idx[mask]
            j_m = j_idx[mask]
            v_m = v[mask]
            d_m = d[mask]
            
            d_safe = np.where(d_m > 1e-10, d_m, 1e-10)
            
            vec = p[i_m] - p[j_m]
            unit = vec / d_safe[:, np.newaxis]
            
            np.add.at(grad, i_m, -2.0 * v_m[:, np.newaxis] * unit)
            np.add.at(grad, j_m,  2.0 * v_m[:, np.newaxis] * unit)
            
        return E, grad.flatten()

    bounds = [(-0.5, 0.5)] * (2 * N)
    p_flat = points.flatten()
    
    try:
        res = opt.minimize(
            objective_and_grad, 
            p_flat, 
            args=(r_target,), 
            jac=True, 
            method='L-BFGS-B', 
            bounds=bounds, 
            options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10}
        )
        
        new_points = res.x.reshape(N, 2)
        new_r = _compute_radius(new_points)
        
        if new_r > current_r:
            return new_points, new_r
    except Exception:
        pass
    
    return points, current_r

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # 1. Generate initial configuration
    points = _generate_hex_grid(N)
    
    # 2. Optimize using repulsion
    points, r = _optimize_repulsion(points, deadline)
    
    # 3. Polish
    if time.time() < deadline - 5.0:
        points, r = _polish(points, deadline)
        
    # 4. Final check and return
    final_r = _compute_radius(points)
    
    return points

# Test
if __name__ == "__main__":
    pts = construct()
    r = _compute_radius(pts)
    print(f"Shape: {pts.shape}")
    print(f"Radius: {r}")
