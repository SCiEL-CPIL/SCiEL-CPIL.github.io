import numpy as np
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s out of 150s, safety margin

    def get_hex_init():
        """
        Generate a hexagonal close-packing-like initialization.
        This is much closer to the optimal packing geometry than a square grid,
        allowing the optimizer to focus on tightening constraints rather than
        re-arranging the topology.
        """
        n = N
        # Arrange in a hexagonal grid pattern
        # 6 columns, 6 rows (36 cells, take 32)
        # Actually, for 32 circles, a 6x6 grid is 36. We can trim 4.
        # Or use a more compact hex arrangement.
        # Let's use a standard hex lattice with spacing s.
        # To fit 32 circles, we need roughly sqrt(32) ~ 5.6x5.6 area.
        
        xy = np.zeros((n, 2))
        r = np.zeros(n)
        
        # Hex grid parameters
        # Rows: 0, 1, 2, 3, 4, 5
        # Cols: 0, 1, 2, 3, 4, 5
        # Offset odd rows by 0.5 * spacing_x
        
        # Let's define a grid that is slightly compressed to allow growth
        cols = 6
        rows = 6
        
        # Spacing
        # If we want r ~ 0.08, spacing ~ 0.16
        # 6 * 0.16 = 0.96 < 1.0. Good.
        
        dx = 1.0 / cols
        dy = 1.0 / rows
        
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx >= n:
                    break
                x = (col + 0.5) * dx
                y = (row + 0.5) * dy
                if row % 2 == 1:
                    x += 0.5 * dx
                
                # Keep within bounds
                x = np.clip(x, 0.1, 0.9)
                y = np.clip(y, 0.1, 0.9)
                
                xy[idx] = [x, y]
                r[idx] = 0.07 # Start with a reasonable radius
                idx += 1
        
        return np.concatenate([xy.flatten(), r])

    def compute_penalty_and_grad(params):
        """
        Compute the penalty function and its gradient for smooth optimization.
        Penalty = -sum(r) + lambda_boundary * ||violations||^2 + lambda_overlap * ||overlaps||^2
        """
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]
        
        # 1. Boundary Constraints
        # d_left = x - r >= 0
        # d_right = 1 - x - r >= 0
        # d_bottom = y - r >= 0
        # d_top = 1 - y - r >= 0
        
        d_left = xy[:, 0] - r
        d_right = 1 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1 - xy[:, 1] - r
        
        # Smooth penalty for violations: max(0, -d)^2
        viol_left = np.maximum(0, -d_left)
        viol_right = np.maximum(0, -d_right)
        viol_bottom = np.maximum(0, -d_bottom)
        viol_top = np.maximum(0, -d_top)
        
        boundary_penalty = np.sum(viol_left**2) + np.sum(viol_right**2) + \
                           np.sum(viol_bottom**2) + np.sum(viol_top**2)
        
        # Gradients for boundary
        grad_xy = np.zeros_like(xy)
        grad_r = np.zeros_like(r)
        
        # d/dx_left = 1, d/dr_left = -1
        # If viol > 0: d/dx = 2*viol * 1, d/dr = 2*viol * (-1)
        mask_left = viol_left > 0
        grad_xy[mask_left, 0] += 2 * viol_left[mask_left]
        grad_r[mask_left] -= 2 * viol_left[mask_left]
        
        mask_right = viol_right > 0
        grad_xy[mask_right, 0] -= 2 * viol_right[mask_right]
        grad_r[mask_right] -= 2 * viol_right[mask_right]
        
        mask_bottom = viol_bottom > 0
        grad_xy[mask_bottom, 1] += 2 * viol_bottom[mask_bottom]
        grad_r[mask_bottom] -= 2 * viol_bottom[mask_bottom]
        
        mask_top = viol_top > 0
        grad_xy[mask_top, 1] -= 2 * viol_top[mask_top]
        grad_r[mask_top] -= 2 * viol_top[mask_top]
        
        # 2. Overlap Constraints
        # dist(i,j) >= r_i + r_j
        # viol = max(0, r_i + r_j - dist)
        
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist_sq = dx**2 + dy**2
        # Avoid division by zero
        dist = np.sqrt(dist_sq + 1e-12)
        
        r_sum = r[:, None] + r[None, :]
        overlap = np.maximum(0, r_sum - dist)
        
        # Only consider i < j
        iu = np.triu_indices(n, k=1)
        overlap_iu = overlap[iu]
        
        overlap_penalty = np.sum(overlap_iu**2)
        
        # Gradients for overlap
        # Let v_ij = max(0, r_i + r_j - d_ij)
        # d/dx_i = -2 v_ij * (x_i - x_j) / d_ij
        # d/dx_j =  2 v_ij * (x_i - x_j) / d_ij
        # d/dr_i = 2 v_ij
        # d/dr_j = 2 v_ij
        
        # Vectorized gradient accumulation is tricky with pairs.
        # We can use np.add.at or a loop over active pairs if few.
        # For 32 circles, ~500 pairs. A loop is fast enough.
        
        active_mask = overlap_iu > 0
        if np.any(active_mask):
            i_idx, j_idx = iu[0][active_mask], iu[1][active_mask]
            v_vals = overlap_iu[active_mask]
            d_vals = dist[i_idx, j_idx]
            
            dx_vals = dx[i_idx, j_idx]
            dy_vals = dy[i_idx, j_idx]
            
            # Gradient factor: 2 * v / d
            factor = 2 * v_vals / d_vals
            
            # Update xy[i] -= factor * (dx, dy)
            # Update xy[j] += factor * (dx, dy)
            # Note: dx = x_i - x_j. Force on i is -grad, so we want to push i away from j?
            # Penalty increases when overlap increases.
            # Grad of penalty w.r.t x_i is -2v * (x_i - x_j)/d.
            # So descent direction is -Grad = +2v * (x_i - x_j)/d.
            # This pushes i in direction (x_i - x_j), i.e., away from j. Correct.
            
            grad_xy[i_idx, 0] -= factor * dx_vals
            grad_xy[i_idx, 1] -= factor * dy_vals
            grad_xy[j_idx, 0] += factor * dx_vals
            grad_xy[j_idx, 1] += factor * dy_vals
            
            grad_r[i_idx] += 2 * v_vals
            grad_r[j_idx] += 2 * v_vals
        
        # Objective: Minimize Penalty - Sum(r)
        # So we subtract 1 from grad_r (since d/dr(-sum(r)) = -1)
        grad_r -= 1.0
        
        # Combine weights
        w_boundary = 10.0
        w_overlap = 10.0
        
        total_penalty = w_boundary * boundary_penalty + w_overlap * overlap_penalty
        total_grad_xy = w_boundary * grad_xy + w_overlap * grad_xy # Wait, I calculated grads for specific terms.
        
        # Recalculate combined gradient properly
        # Boundary part:
        g_b_xy = np.zeros_like(xy)
        g_b_r = np.zeros_like(r)
        g_b_xy[mask_left, 0] += 2 * viol_left[mask_left]
        g_b_r[mask_left] -= 2 * viol_left[mask_left]
        g_b_xy[mask_right, 0] -= 2 * viol_right[mask_right]
        g_b_r[mask_right] -= 2 * viol_right[mask_right]
        g_b_xy[mask_bottom, 1] += 2 * viol_bottom[mask_bottom]
        g_b_r[mask_bottom] -= 2 * viol_bottom[mask_bottom]
        g_b_xy[mask_top, 1] -= 2 * viol_top[mask_top]
        g_b_r[mask_top] -= 2 * viol_top[mask_top]
        
        # Overlap part:
        g_o_xy = np.zeros_like(xy)
        g_o_r = np.zeros_like(r)
        if np.any(active_mask):
            g_o_xy[i_idx, 0] -= factor * dx_vals
            g_o_xy[i_idx, 1] -= factor * dy_vals
            g_o_xy[j_idx, 0] += factor * dx_vals
            g_o_xy[j_idx, 1] += factor * dy_vals
            g_o_r[i_idx] += 2 * v_vals
            g_o_r[j_idx] += 2 * v_vals
            
        # Total Gradient
        grad_xy_final = w_boundary * g_b_xy + w_overlap * g_o_xy
        grad_r_final = w_boundary * g_b_r + w_overlap * g_o_r - 1.0
        
        return total_penalty, np.concatenate([grad_xy_final.flatten(), grad_r_final])

    def is_valid(params):
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]
        
        if np.any(r < -1e-9):
            return False
        if np.any(xy[:, 0] - r < -1e-9) or np.any(1 - xy[:, 0] - r < -1e-9):
            return False
        if np.any(xy[:, 1] - r < -1e-9) or np.any(1 - xy[:, 1] - r < -1e-9):
            return False
        
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        if np.any(dist[iu] < min_dist[iu] - 1e-9):
            return False
        return True

    def get_score(params):
        n = N
        r = params[2*n:]
        return np.sum(r)

    # Main Optimization Loop
    # Start with hex init
    params = get_hex_init()
    
    best_params = params.copy()
    best_score = get_score(params)
    
    # Adaptive Gradient Descent
    # We will use a simple gradient descent with line search or fixed step decay.
    # Since the problem is non-convex, we need to be careful.
    # The hex init is likely in the basin of attraction for the global optimum.
    
    lr = 0.01
    lr_decay = 0.999
    min_lr = 1e-6
    
    # Run for a fixed number of iterations or until time limit
    max_iters = 20000
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute penalty and gradient
        penalty, grad = compute_penalty_and_grad(params)
        
        # Update
        params -= lr * grad
        
        # Ensure radii are non-negative
        params[2*N:] = np.maximum(0, params[2*N:])
        
        # Check validity and update best
        if is_valid(params):
            score = get_score(params)
            if score > best_score:
                best_score = score
                best_params = params.copy()
        
        # Decay learning rate
        lr *= lr_decay
        if lr < min_lr:
            # Try a smaller step or break if converged
            # If we have a valid best, we can try to polish with smaller steps
            # Or just break if improvement is negligible
            pass
            
        # Every 100 steps, try a "restart" with a slightly perturbed version of best
        # to escape local minima if stuck?
        # Given the hex init is good, this might not be needed, but let's add a simple escape mechanism.
        if i % 500 == 0 and i > 0:
            # Check if we are making progress
            current_penalty, _ = compute_penalty_and_grad(params)
            # If penalty is not decreasing, perturb
            # Simple perturbation
            if i > 1000:
                # Add small noise
                noise = np.random.normal(0, 0.01, params.shape)
                # Only apply to positions to keep radii stable
                noise[2*N:] = 0
                params += noise
                
    # Final Polish: Use L-BFGS on the best found to tighten
    if time.time() < deadline - 5:
        def penalty_function(p):
            val, _ = compute_penalty_and_grad(p)
            return val
        
        def penalty_grad(p):
            _, g = compute_penalty_and_grad(p)
            return g
        
        try:
            from scipy.optimize import minimize
            res = minimize(penalty_function, best_params, jac=penalty_grad, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            if is_valid(res.x):
                score = get_score(res.x)
                if score > best_score:
                    best_score = score
                    best_params = res.x
        except Exception:
            pass

    # Final validation and clipping
    if best_params is not None:
        n = N
        xy = best_params[:2*n].reshape(n, 2).copy()
        r = best_params[2*n:].copy()
        r = np.clip(r, 0, 0.5)
        xy[:, 0] = np.clip(xy[:, 0], r, 1-r)
        xy[:, 1] = np.clip(xy[:, 1], r, 1-r)
        final_params = np.concatenate([xy.flatten(), r])
        if is_valid(final_params):
            return np.column_stack([xy, r])

    # Fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])

if __name__ == "__main__":
    res = construct()
    print(res)
