import numpy as np
import time
import scipy.optimize as opt

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill
    
    # 1. Generate a good initial hexagonal lattice configuration
    # We want ~200 points in [-0.5, 0.5]^2
    # Hexagonal packing density is sqrt(3)/2 ~ 0.866
    # Area per point ~ 1/200 = 0.005
    # d^2 * sqrt(3)/2 ~ 0.005 => d^2 ~ 0.00577 => d ~ 0.076
    
    d = 0.076
    points_list = []
    # Generate points on a hexagonal grid covering the square
    # y step is d * sqrt(3) / 2
    dy = d * np.sqrt(3) / 2
    
    # Determine number of rows
    n_rows = int(np.ceil(1.0 / dy)) + 2
    # Start from -0.5 - dy
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
    
    # We might have more or fewer than N points
    # Strategy: If we have more, randomly sample N. If fewer, add random points.
    # Better: If we have more, we can try to find the best subset? Too slow.
    # Just random sampling from the lattice is a decent start, but might break symmetry.
    # Let's try to center the lattice if possible, or just take the first N?
    # Taking first N from a centered generation is okay.
    # However, random sampling might lead to better "randomness" for the optimizer.
    
    if len(points) > N:
        # Shuffle and take first N
        np.random.shuffle(points)
        points = points[:N]
    elif len(points) < N:
        # Add random points
        remaining = N - len(points)
        new_pts = np.random.uniform(-0.45, 0.45, size=(remaining, 2))
        points = np.vstack([points, new_pts])
        
    # Clip to valid range (slightly inside to avoid immediate boundary issues)
    points[:, 0] = np.clip(points[:, 0], -0.49, 0.49)
    points[:, 1] = np.clip(points[:, 1], -0.49, 0.49)
    
    # Function to calculate the radius
    def get_radius(pts):
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_c = np.min(dist) / 2.0
        
        # Boundary distances
        dx = np.minimum(pts[:, 0] + 0.5, 0.5 - pts[:, 0])
        dy = np.minimum(pts[:, 1] + 0.5, 0.5 - pts[:, 1])
        min_b = np.min(np.minimum(dx, dy))
        
        return min(min_c, min_b)
    
    current_r = get_radius(points)
    
    # Optimization using L-BFGS-B with an energy function
    # We minimize E = sum(violations^2) where violations are for a target radius r_target
    # We will iteratively increase r_target.
    
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
        # Only active where viol_b > 0
        # d/dp_i (r - b_dist)^2 = -2(r - b_dist) * grad(b_dist)
        # grad(b_dist) points inward.
        
        # We compute the gradient contribution for each point
        # This part is tricky to vectorize perfectly without loops or complex indexing
        # because of the min(dx, dy) and the conditionals.
        # For speed, we can use a loop over N=200, which is fast.
        
        for i in range(N):
            if viol_b[i] > 0:
                x, y = p[i]
                d_left = x + 0.5
                d_right = 0.5 - x
                d_bottom = y + 0.5
                d_top = 0.5 - y
                
                min_d = min(d_left, d_right, d_bottom, d_top)
                
                # Gradient of b_dist
                gx = 0.0
                gy = 0.0
                
                if min_d == d_left:
                    gx = 1.0  # Move right
                elif min_d == d_right:
                    gx = -1.0 # Move left
                elif min_d == d_bottom:
                    gy = 1.0  # Move up
                elif min_d == d_top:
                    gy = -1.0 # Move down
                
                # dE/dp = -2 * viol * grad(b_dist)
                # Wait, E = (r - b)^2. dE/db = -2(r-b).
                # db/dx = gx. dE/dx = -2(r-b)*gx.
                # We want to MINIMIZE E. The optimizer moves in direction -grad.
                # So -grad = 2(r-b)*gx. This points inward. Correct.
                
                grad[i, 0] += -2.0 * viol_b[i] * gx
                grad[i, 1] += -2.0 * viol_b[i] * gy

        # Pairwise violations
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # viol = max(0, 2*r - dist)
        viol_pairs = np.maximum(0.0, 2.0 * r_target - dist)
        
        # Sum of squares for E
        # Note: sum over i<j is 0.5 * sum over i!=j
        # But we can just sum the upper triangle to be precise and avoid double counting in E.
        # For gradient, we need to be careful.
        
        # Let's use the upper triangle indices
        i_idx, j_idx = np.triu_indices(N, k=1)
        v = viol_pairs[i_idx, j_idx]
        d = dist[i_idx, j_idx]
        
        E += np.sum(v**2)
        
        # Gradient
        # For pair (i, j):
        # T = (2r - d)^2
        # dT/dp_i = -2(2r-d) * (p_i - p_j)/d
        # dT/dp_j = -2(2r-d) * (p_j - p_i)/d = 2(2r-d) * (p_i - p_j)/d
        
        # Only add if v > 0
        mask = v > 0
        if np.any(mask):
            i_m = i_idx[mask]
            j_m = j_idx[mask]
            v_m = v[mask]
            d_m = d[mask]
            
            # Avoid division by zero
            d_safe = np.where(d_m > 1e-10, d_m, 1e-10)
            
            vec = p[i_m] - p[j_m]
            unit = vec / d_safe[:, np.newaxis]
            
            # Magnitude factor
            factor = 2.0 * v_m / d_safe
            
            # grad[i] += -factor * vec ?
            # dT/dp_i = -2 * v * (p_i - p_j) / d
            # So grad[i] += -2 * v * unit
            # grad[j] +=  2 * v * unit
            
            np.add.at(grad, i_m, -2.0 * v_m[:, np.newaxis] * unit)
            np.add.at(grad, j_m,  2.0 * v_m[:, np.newaxis] * unit)
            
        return E, grad.flatten()

    bounds = [(-0.5, 0.5)] * (2 * N)
    
    best_points = points.copy()
    best_r = current_r
    
    # Iterative radius increase
    # Start with current_r, try to increase it.
    r = current_r
    step_r = 0.001
    max_steps = 5000 # Safety cap
    
    # We can use a smarter step size or just fixed.
    # Fixed is safer for stability.
    
    while time.time() < deadline:
        new_r = r + step_r
        
        # Optimization
        p_flat = points.flatten()
        
        try:
            # Use L-BFGS-B
            # We don't need perfect convergence, just improvement.
            res = opt.minimize(
                objective_and_grad, 
                p_flat, 
                args=(new_r,), 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 30, 'gtol': 1e-5, 'ftol': 1e-7}
            )
            
            # Check if the solution is valid for new_r
            # We check the actual radius of the resulting points
            new_points = res.x.reshape(N, 2)
            new_actual_r = get_radius(new_points)
            
            # If the actual radius is close to new_r (or at least better than current best), accept
            # We accept if new_actual_r > best_r
            if new_actual_r > best_r:
                best_r = new_actual_r
                best_points = new_points.copy()
                points = new_points
                r = new_actual_r # Update r to the achieved radius
                # Maybe increase step if we are doing well?
                # For now, keep step constant or slightly increase.
                # step_r *= 1.05
            else:
                # If we can't improve, we might be stuck.
                # We can try to reduce step_r or break.
                # If new_actual_r is significantly less than new_r, we failed to pack tighter.
                # Let's break if we can't make progress.
                # Or we can try to continue with the current points but smaller step?
                # If we just failed to reach new_r, maybe we should try new_r = r + step_r/2?
                # Or just break and return best.
                # Let's try a few retries with smaller steps if we stall.
                # For simplicity, break.
                break
        except Exception:
            break

    return best_points

# Test
if __name__ == "__main__":
    pts = construct()
    r = get_radius(pts) if 'get_radius' in dir() else None
    # Re-define get_radius for test scope if needed, but construct has it internal.
    # Let's just print shape.
    print(f"Shape: {pts.shape}")
    # Calculate radius manually for test
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_c = np.min(dist) / 2.0
    dx = np.minimum(pts[:, 0] + 0.5, 0.5 - pts[:, 0])
    dy = np.minimum(pts[:, 1] + 0.5, 0.5 - pts[:, 1])
    min_b = np.min(np.minimum(dx, dy))
    print(f"Radius: {min(min_c, min_b)}")
