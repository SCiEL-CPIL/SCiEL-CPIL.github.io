"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    # Variables: x_0, y_0, r_0, x_1, y_1, r_1, ...
    # Total variables: 3 * N
    
    def get_params(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        return x, y, r
    
    def objective(v):
        x, y, r = get_params(v)
        
        # Calculate pairwise distances
        # d_ij = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
        # Overlap = max(0, r_i + r_j - d_ij)
        
        # Efficient pairwise calculation
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        
        # Fill diagonal with inf to ignore self-overlap
        np.fill_diagonal(d, np.inf)
        
        r_sum = r[:, None] + r[None, :]
        
        # Overlap amount
        overlap = r_sum - d
        # Penalty for overlaps
        # Using squared penalty for smoothness
        # Only positive overlaps contribute
        pos_overlap = np.maximum(overlap, 0)
        overlap_penalty = np.sum(pos_overlap**2)
        
        # Wall penalties
        # x_i < r_i  => r_i - x_i > 0
        # x_i > 1 - r_i => x_i - (1-r_i) > 0 => r_i - (1-x_i) > 0
        wall_pen_x = np.maximum(r - x, 0)**2 + np.maximum(r - (1 - x), 0)**2
        wall_pen_y = np.maximum(r - y, 0)**2 + np.maximum(r - (1 - y), 0)**2
        wall_penalty = np.sum(wall_pen_x) + np.sum(wall_pen_y)
        
        # Reward for large radii
        # We want to maximize sum(r), so minimize -lambda * sum(r)
        # Lambda needs to be scaled appropriately.
        # If lambda is too small, radii stay small. If too large, it might fight the penalty.
        # Let's try a high lambda.
        lambda_reward = 100.0
        reward = -lambda_reward * np.sum(r)
        
        return overlap_penalty + wall_penalty + reward
    
    def gradient(v):
        # Numerical gradient is slow. Let's provide analytical if possible, 
        # or let L-BFGS-B use numerical. 
        # For 78 variables, numerical gradient (79 evaluations per step) is okay-ish 
        # but analytical is better. 
        # Given time constraints, let's try numerical first for simplicity, 
        # but optimize the objective evaluation speed.
        # Actually, providing an analytical gradient is crucial for speed.
        
        x, y, r = get_params(v)
        n = N
        
        grad = np.zeros(3 * n)
        
        # 1. Overlap Penalty Gradient
        # P = sum_{i<j} max(0, r_i + r_j - d_ij)^2
        # dP/dr_i = 2 * max(0, r_i + r_j - d_ij) * (1 - dx_ij / d_ij)  (for each j)
        # dP/dx_i = 2 * max(0, r_i + r_j - d_ij) * (-dx_ij / d_ij)
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, 1.0) # Avoid div by zero
        
        r_sum = r[:, None] + r[None, :]
        overlap = r_sum - d
        pos_overlap = np.maximum(overlap, 0)
        
        # Gradient contribution from overlap
        # For r_i:
        # sum_j 2 * pos_overlap[i,j]
        grad_r = 2.0 * np.sum(pos_overlap, axis=1)
        
        # For x_i:
        # sum_j 2 * pos_overlap[i,j] * (-(x_i - x_j) / d_ij)
        # Note: d_ij is symmetric.
        # term = pos_overlap * (-dx / d)
        # We need to be careful with the diagonal.
        # pos_overlap on diagonal is 0 (since d=0, r_sum>0, but we set d=1? No, we should handle diagonal properly).
        # Let's re-calculate with explicit loop or careful masking.
        
        # Mask for i != j
        mask = ~np.eye(n, dtype=bool)
        
        # Gradient w.r.t x
        # g_x[i] = sum_{j != i} 2 * pos_overlap[i,j] * (-(x_i - x_j) / d_ij)
        # Since pos_overlap is symmetric, and (x_i-x_j)/d is antisymmetric? No.
        # Let's compute the force vector.
        # F_x[i] = sum_j 2 * pos_overlap[i,j] * (x_i - x_j) / d_ij  <-- This is the force pushing i away from j?
        # d/dx_i (r_i + r_j - d_ij)^2 = 2(r_i+r_j-d_ij) * (-d/dx_i d_ij)
        # d/dx_i d_ij = (x_i - x_j) / d_ij
        # So gradient = -2 * pos_overlap[i,j] * (x_i - x_j) / d_ij
        
        # We can compute this matrix-wise.
        # term_x[i,j] = 2 * pos_overlap[i,j] * (x_i - x_j) / d_ij
        # grad_x[i] = - sum_j term_x[i,j]
        
        # Handle d=0 case (should not happen if initialized well, but just in case)
        safe_d = np.where(d < 1e-10, 1e-10, d)
        
        # Only non-diagonal terms matter
        pos_overlap_safe = np.where(mask, pos_overlap, 0.0)
        
        # Gradient w.r.t r
        grad_r = 2.0 * np.sum(pos_overlap_safe, axis=1)
        
        # Gradient w.r.t x
        # factor = pos_overlap_safe / safe_d
        # g_x[i] = - sum_j 2 * factor[i,j] * (x_i - x_j)
        # Note: sum_j (x_i - x_j) * factor[i,j]
        
        # Let's use matrix operations
        # G_x = -2 * (pos_overlap_safe / safe_d) @ (x[None, :] - x[:, None])? 
        # Wait, (x_i - x_j) is row i, col j.
        # Sum over j (columns) for each i (row).
        # dx_mat = x[:, None] - x[None, :]
        # grad_x = -2.0 * np.sum((pos_overlap_safe / safe_d) * dx_mat, axis=1)
        
        grad_x = -2.0 * np.sum((pos_overlap_safe / safe_d) * dx, axis=1)
        grad_y = -2.0 * np.sum((pos_overlap_safe / safe_d) * dy, axis=1)
        
        # 2. Wall Penalty Gradient
        # P_wall_x = sum_i max(0, r_i - x_i)^2 + max(0, r_i - (1 - x_i))^2
        # dP/dx_i = 2*max(0, r_i-x_i)*(-1) + 2*max(0, r_i-(1-x_i))*(1)
        # dP/dr_i = 2*max(0, r_i-x_i)*(1) + 2*max(0, r_i-(1-x_i))*(1)
        
        # Left wall: x_i < r_i
        left_viol = np.maximum(r - x, 0)
        # Right wall: x_i > 1 - r_i  => 1 - x_i < r_i
        right_viol = np.maximum(r - (1 - x), 0)
        # Bottom wall: y_i < r_i
        bot_viol = np.maximum(r - y, 0)
        # Top wall: y_i > 1 - r_i
        top_viol = np.maximum(r - (1 - y), 0)
        
        grad_x += -2.0 * left_viol + 2.0 * right_viol
        grad_y += -2.0 * bot_viol + 2.0 * top_viol
        
        grad_r += 2.0 * (left_viol + right_viol + bot_viol + top_viol)
        
        # 3. Reward Gradient
        # J = ... - lambda * sum(r)
        # dJ/dr_i = -lambda
        lambda_reward = 100.0
        grad_r -= lambda_reward
        
        # Assemble gradient
        grad[0::3] = grad_x
        grad[1::3] = grad_y
        grad[2::3] = grad_r
        
        return grad

    # Initialization
    # Jittered grid
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                # Add jitter to break symmetry
                jitter = 0.01
                x0 = (i + 0.5) / side + np.random.uniform(-jitter, jitter)
                y0 = (j + 0.5) / side + np.random.uniform(-jitter, jitter)
                # Initial radius small enough to not overlap immediately in grid
                r0 = 0.5 / side * 0.9
                cells.append([x0, y0, r0])
    
    # Ensure we have exactly N
    while len(cells) < N:
        cells.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9), 0.01])
    
    v0 = np.array(cells).flatten()
    
    # Constraints:
    # 0 <= x <= 1, 0 <= y <= 1, r > 0
    # L-BFGS-B bounds
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((1e-5, 0.5)) # r (radius cannot exceed 0.5 in unit square)
    
    # Run optimization
    best_v = v0
    best_score = -np.sum(get_params(v0)[2])
    
    # Try a few restarts if time permits
    num_restarts = 5
    
    for k in range(num_restarts):
        if time.time() > deadline:
            break
        
        if k > 0:
            # Perturb previous best or use new random
            if k % 2 == 0:
                v0 = best_v + np.random.normal(0, 0.01, size=3*N)
            else:
                # New jittered grid
                cells = []
                for i in range(side):
                    for j in range(side):
                        if len(cells) < N:
                            jitter = 0.05
                            x0 = (i + 0.5) / side + np.random.uniform(-jitter, jitter)
                            y0 = (j + 0.5) / side + np.random.uniform(-jitter, jitter)
                            r0 = 0.5 / side * 0.9
                            cells.append([x0, y0, r0])
                while len(cells) < N:
                    cells.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9), 0.01])
                v0 = np.array(cells).flatten()
        
        # Clip v0 to bounds
        for i in range(N):
            v0[3*i] = np.clip(v0[3*i], 0.0, 1.0)
            v0[3*i+1] = np.clip(v0[3*i+1], 0.0, 1.0)
            v0[3*i+2] = np.clip(v0[3*i+2], 1e-5, 0.5)
        
        try:
            res = minimize(
                objective, 
                v0, 
                method='L-BFGS-B', 
                jac=gradient, 
                bounds=bounds, 
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < objective(best_v):
                # Check validity
                x, y, r = get_params(res.x)
                # Simple validity check: no overlaps, inside bounds
                valid = True
                # Check bounds
                if np.any(x < 0) or np.any(x > 1) or np.any(y < 0) or np.any(y > 1) or np.any(r < 0):
                    valid = False
                # Check overlaps
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                            if dist < r[i] + r[j] - 1e-7:
                                valid = False
                                break
                        if not valid: break
                
                if valid:
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_v = res.x.copy()
                        
        except Exception as e:
            pass

    # Final result
    x, y, r = get_params(best_v)
    
    # Post-processing: Ensure strict validity by shrinking radii slightly if needed
    # This is a safety net.
    # Check overlaps again
    for _ in range(10):
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                sum_r = r[i] + r[j]
                if dist < sum_r:
                    # Shrink both by half the overlap
                    overlap = sum_r - dist
                    shrink = overlap / 2.0 + 1e-5
                    r[i] -= shrink
                    r[j] -= shrink
                    if r[i] < 1e-5: r[i] = 1e-5
                    if r[j] < 1e-5: r[j] = 1e-5
        
        # Check walls
        for i in range(N):
            min_dist_wall = min(x[i], 1-x[i], y[i], 1-y[i])
            if r[i] > min_dist_wall:
                r[i] = min_dist_wall - 1e-5
                if r[i] < 1e-5: r[i] = 1e-5

    # Construct output array
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [x[i], y[i], r[i]]
        
    return result

# EVOLVE-BLOCK-END
