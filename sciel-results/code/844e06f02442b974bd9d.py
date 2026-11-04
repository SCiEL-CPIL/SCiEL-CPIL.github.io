"""
Circle packing in unit square using a Voronoi-based dual representation.
Centers are optimized; radii are derived from Voronoi cell inradii.
Combines SA global exploration, best-fit local search, and smooth soft-min
gradient polish with hard sub-gradient refinement.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 145.0  # Safety margin from 150s limit


def construct():
    start_time = time.time()

    def compute_radii(x, y):
        """Compute radii from centers: min of half-distance to neighbors and boundary."""
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)

        min_dist = np.min(dist, axis=1)
        r = np.minimum(r_bound, 0.5 * min_dist)
        r = np.maximum(r, 1e-6)
        return r

    def _grow_radii(x, y, r, max_iters=500):
        """Try to increase radii while maintaining feasibility.
        Uses a simple greedy approach: for each circle, try to increase its radius
        slightly and check if all constraints are still satisfied.
        """
        x = x.copy()
        y = y.copy()
        r = r.copy()
        
        for it in range(max_iters):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            improved = False
            for i in range(N):
                # Try increasing radius i by small amount
                dr = 0.001
                r_new = r.copy()
                r_new[i] += dr
                
                # Check validity
                if np.any(r_new < 0) or np.any(r_new > 0.5):
                    continue
                if np.any(x - r_new < -1e-10) or np.any(x + r_new > 1 + 1e-10):
                    continue
                if np.any(y - r_new < -1e-10) or np.any(y + r_new > 1 + 1e-10):
                    continue
                
                # Check pairwise constraints
                valid = True
                for j in range(N):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < r_new[i] + r_new[j] - 1e-10:
                        valid = False
                        break
                
                if valid:
                    r[i] = r_new[i]
                    improved = True
            
            if not improved:
                break
        
        return r

    def _maximize_radii(x, y, r, max_iters=300):
        """More aggressive radius maximization using binary search per circle.
        For each circle, find the maximum radius it can have given fixed positions.
        """
        x = x.copy()
        y = y.copy()
        r = r.copy()
        
        for it in range(max_iters):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            improved = False
            for i in range(N):
                # Binary search for max radius of circle i
                lo = r[i]
                hi = 0.5  # Max possible radius
                
                # Quick check: what's the theoretical max?
                r_bound_i = np.minimum(np.minimum(x[i], 1.0 - x[i]),
                                       np.minimum(y[i], 1.0 - y[i]))
                min_neighbor_dist = np.inf
                for j in range(N):
                    if i == j:
                        continue
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    min_neighbor_dist = min(min_neighbor_dist, d)
                hi = min(hi, r_bound_i, 0.5 * min_neighbor_dist)
                
                if hi <= lo:
                    continue
                
                # Binary search
                best_r = lo
                for _ in range(20):
                    mid = 0.5 * (lo + hi)
                    r_test = r.copy()
                    r_test[i] = mid
                    
                    # Check validity
                    if np.any(r_test < 0) or np.any(r_test > 0.5):
                        hi = mid
                        continue
                    if np.any(x - r_test < -1e-10) or np.any(x + r_test > 1 + 1e-10):
                        hi = mid
                        continue
                    if np.any(y - r_test < -1e-10) or np.any(y + r_test > 1 + 1e-10):
                        hi = mid
                        continue
                    
                    valid = True
                    for j in range(N):
                        if i == j:
                            continue
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        d = np.sqrt(dx*dx + dy*dy)
                        if d < r_test[i] + r_test[j] - 1e-10:
                            valid = False
                            break
                    
                    if valid:
                        best_r = mid
                        lo = mid
                    else:
                        hi = mid
                
                if best_r > r[i] + 1e-10:
                    r[i] = best_r
                    improved = True
            
            if not improved:
                break
        
        return r

    def smooth_objective_and_grad(params, beta=50.0):
        """Compute -sum(r) and its gradient with respect to params = [x, y].
        Uses analytic gradient of the log-sum-exp soft-min.
        """
        x = params[:N]
        y = params[N:2*N]

        # Boundary constraint
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        # Pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-30)
        np.fill_diagonal(dist, 1e10)

        # Build constraint matrix: c_ij for all i,j
        # c_ij = 0.5 * dist(i,j) for i != j
        # c_ii = r_bound[i]
        c = 0.5 * dist
        np.fill_diagonal(c, r_bound)

        # Soft-min computation
        neg_beta_c = -beta * c
        max_neg = np.max(neg_beta_c, axis=1, keepdims=True)
        exp_terms = np.exp(neg_beta_c - max_neg)
        sum_exp = np.sum(exp_terms, axis=1, keepdims=True)
        log_sum = np.log(sum_exp)
        r = -(max_neg + log_sum) / beta
        r = r[:, 0]

        # Weights: w_ij = exp(-beta*c_ij) / sum_j' exp(-beta*c_ij')
        w = exp_terms / sum_exp  # (N, N)

        # Gradient of c_ij w.r.t. parameters
        dist_safe = dist.copy()
        np.fill_diagonal(dist_safe, 1.0)

        # Off-diagonal gradients
        dc_dx_i = 0.5 * dx / dist_safe  # dc_ij/dx_i
        dc_dy_i = 0.5 * dy / dist_safe  # dc_ij/dy_i
        dc_dx_j = -0.5 * dx / dist_safe  # dc_ij/dx_j
        dc_dy_j = -0.5 * dy / dist_safe  # dc_ij/dy_j

        # Diagonal gradients for r_bound_i
        # Use smooth approximation: weight each boundary by how close it is
        bounds_4 = np.stack([x, 1.0-x, y, 1.0-y], axis=1)  # (N, 4)
        beta_b = 100.0
        neg_bb = -beta_b * bounds_4
        max_bb = np.max(neg_bb, axis=1, keepdims=True)
        exp_bb = np.exp(neg_bb - max_bb)
        sum_bb = np.sum(exp_bb, axis=1, keepdims=True)
        w_b = exp_bb / sum_bb  # (N, 4)

        # Gradient of softmin(x, 1-x, y, 1-y) w.r.t. x_i:
        grad_rb_dx = w_b[:, 0] - w_b[:, 1]
        # Gradient w.r.t. y_i:
        grad_rb_dy = w_b[:, 2] - w_b[:, 3]

        # Set diagonal entries
        np.fill_diagonal(dc_dx_i, grad_rb_dx)
        np.fill_diagonal(dc_dy_i, grad_rb_dy)
        np.fill_diagonal(dc_dx_j, grad_rb_dx)
        np.fill_diagonal(dc_dy_j, grad_rb_dy)

        # Total gradient w.r.t. x_k: sum_i dr_i/dx_k
        dr_dx_self = np.sum(w * dc_dx_i, axis=1)  # (N,)
        dr_dy_self = np.sum(w * dc_dy_i, axis=1)  # (N,)
        dr_dx_cross = np.sum(w * dc_dx_j, axis=0)  # (N,)
        dr_dy_cross = np.sum(w * dc_dy_j, axis=0)  # (N,)

        # Subtract double-counted diagonal term
        dr_dx = dr_dx_self + dr_dx_cross - np.diag(w) * np.diag(dc_dx_i)
        dr_dy = dr_dy_self + dr_dy_cross - np.diag(w) * np.diag(dc_dy_i)

        # Objective is -sum(r), so gradient is -dr/dparams
        grad = np.concatenate([-dr_dx, -dr_dy])

        obj_val = -np.sum(r)
        return obj_val, grad

    def check_validity(x, y, r):
        if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
            return False
        if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
            return False
        if np.any(r < -1e-9):
            return False

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(min_dist, 0)

        if np.any(dist < min_dist - 1e-9):
            return False
        return True

    def _score(x, y, r):
        """Compute the objective value (sum of radii) if valid, else -inf."""
        if check_validity(x, y, r):
            return np.sum(r)
        # Try scaling down to make valid
        for scale in [0.999, 0.99, 0.98, 0.95, 0.90]:
            r_test = r * scale
            if check_validity(x, y, r_test):
                return np.sum(r_test)
        return -np.inf

    def make_initial_guess(seed, mode='grid'):
        rng = np.random.RandomState(seed)
        if mode == 'grid':
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N)
            y = np.zeros(N)
            idx = 0
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    x[idx] = (i + 0.5) / side
                    y[idx] = (j + 0.5) / side
                    idx += 1
            x += rng.uniform(-0.08, 0.08, N)
            y += rng.uniform(-0.08, 0.08, N)
        elif mode == 'hex':
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N)
            y = np.zeros(N)
            idx = 0
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    x[idx] = (i + 0.5) / side + (0.5 / side) * (j % 2)
                    y[idx] = (j + 0.5) / side
                    idx += 1
            x += rng.uniform(-0.05, 0.05, N)
            y += rng.uniform(-0.05, 0.05, N)
        else:  # random
            x = rng.uniform(0.1, 0.9, N)
            y = rng.uniform(0.1, 0.9, N)

        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        return x, y

    def voronoi_greedy_init(seed):
        """Robust largest-empty-circle greedy initialization using Voronoi vertices and boundary sampling."""
        rng = np.random.RandomState(seed)
        
        # Start with a grid of candidate points
        n_grid = 12
        grid_x, grid_y = np.meshgrid(np.linspace(0.1, 0.9, n_grid), np.linspace(0.1, 0.9, n_grid))
        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()
        
        # Add boundary midpoints
        boundary_x = [0.5, 0.5, 0.1, 0.9, 0.1, 0.9, 0.5, 0.5]
        boundary_y = [0.1, 0.9, 0.1, 0.9, 0.5, 0.5, 0.1, 0.9]
        
        all_x = np.concatenate([grid_x, boundary_x])
        all_y = np.concatenate([grid_y, boundary_y])
        
        # Greedy selection of N points that maximize minimum distance
        selected_x = []
        selected_y = []
        remaining_x = all_x.copy()
        remaining_y = all_y.copy()
        
        for _ in range(N):
            if len(remaining_x) == 0:
                break
            
            if len(selected_x) == 0:
                # Pick the center
                idx = np.argmin((remaining_x - 0.5)**2 + (remaining_y - 0.5)**2)
            else:
                # Pick the point that maximizes minimum distance to selected points
                min_dists = np.min(
                    np.sqrt((remaining_x[:, None] - np.array(selected_x)[None, :])**2 + 
                           (remaining_y[:, None] - np.array(selected_y)[None, :])**2),
                    axis=1
                )
                idx = np.argmax(min_dists)
            
            selected_x.append(remaining_x[idx])
            selected_y.append(remaining_y[idx])
            remaining_x = np.delete(remaining_x, idx)
            remaining_y = np.delete(remaining_y, idx)
        
        # Add some random jitter
        for i in range(N):
            selected_x[i] += rng.uniform(-0.02, 0.02)
            selected_y[i] += rng.uniform(-0.02, 0.02)
        
        x = np.array(selected_x)
        y = np.array(selected_y)
        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        return x, y

    def sa_optimize(x0, y0, seed, max_iters=3000):
        """Simulated annealing to optimize center positions with vectorized forces."""
        rng = np.random.RandomState(seed)
        x = x0.copy()
        y = y0.copy()
        r = compute_radii(x, y)
        best_val = np.sum(r)
        best_x = x.copy()
        best_y = y.copy()

        # Cooling schedule
        T_start = 0.05
        T_end = 0.0005
        cool_rate = (T_end / T_start) ** (1.0 / max_iters)

        # Repulsion force parameters
        repulse_k = 2.0
        boundary_k = 5.0

        for it in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break

            T = T_start * (cool_rate ** it)

            # Compute current energy
            r_curr = compute_radii(x, y)
            curr_val = np.sum(r_curr)

            # Compute forces (vectorized)
            fx = np.zeros(N)
            fy = np.zeros(N)

            # Boundary repulsion (vectorized)
            fx += boundary_k * np.clip(0.1 - x, 0, None)
            fx -= boundary_k * np.clip(x - 0.9, 0, None)
            fy += boundary_k * np.clip(0.1 - y, 0, None)
            fy -= boundary_k * np.clip(y - 0.9, 0, None)

            # Pairwise repulsion (vectorized)
            dx = x[:, None] - x[None, :]  # (N, N)
            dy = y[:, None] - y[None, :]  # (N, N)
            d = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(d, 1e10)  # Avoid division by zero on diagonal

            # Force magnitude: repulse_k / (d^2 + eps)
            f_mag = repulse_k / (d**2 + 0.01)
            np.fill_diagonal(f_mag, 0)

            # Force on i from j: -f * (x_i - x_j) / d = f * (x_j - x_i) / d
            fx += np.sum(-f_mag * dx / d, axis=1)
            fy += np.sum(-f_mag * dy / d, axis=1)

            # Propose move: combine force direction with random perturbation
            scale = 0.02 * T / T_start + 0.005
            dx_prop = fx * scale + rng.normal(0, scale, N)
            dy_prop = fy * scale + rng.normal(0, scale, N)

            x_new = x + dx_prop
            y_new = y + dy_prop

            # Clip to valid range
            x_new = np.clip(x_new, 0.05, 0.95)
            y_new = np.clip(y_new, 0.05, 0.95)

            # Evaluate new state
            r_new = compute_radii(x_new, y_new)
            new_val = np.sum(r_new)

            # Acceptance criterion
            delta = new_val - curr_val
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                x = x_new
                y = y_new
                r = r_new
                if new_val > best_val:
                    best_val = new_val
                    best_x = x.copy()
                    best_y = y.copy()

        return best_x, best_y, compute_radii(best_x, best_y), best_val

    def best_fit_local_search(x, y, step_sizes=None, max_rounds=60):
        """Discrete best-fit local search with adaptive step sizes."""
        if step_sizes is None:
            step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]

        # 8 directions
        dirs = np.array([
            [1, 0], [-1, 0], [0, 1], [0, -1],
            [1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)],
            [-1/np.sqrt(2), 1/np.sqrt(2)], [-1/np.sqrt(2), -1/np.sqrt(2)]
        ])  # (8, 2)

        x = x.copy()
        y = y.copy()
        r = compute_radii(x, y)
        curr_val = np.sum(r)

        for rnd in range(max_rounds):
            if time.time() - start_time > DEADLINE - 5:
                break

            best_improvement = 1e-10
            best_i = -1
            best_dir = None
            best_step = None

            for step in step_sizes:
                # For each circle i and each direction d, compute new position
                x_new = x[:, None] + dirs[None, :, 0] * step  # (N, 8)
                y_new = y[:, None] + dirs[None, :, 1] * step  # (N, 8)

                # Mask invalid positions
                valid = (x_new >= 0.01) & (x_new <= 0.99) & (y_new >= 0.01) & (y_new <= 0.99)

                for i in range(N):
                    for d in range(8):
                        if not valid[i, d]:
                            continue

                        x_temp = x.copy()
                        y_temp = y.copy()
                        x_temp[i] = x_new[i, d]
                        y_temp[i] = y_new[i, d]

                        r_new = compute_radii(x_temp, y_temp)
                        new_val = np.sum(r_new)
                        improvement = new_val - curr_val

                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_i = i
                            best_dir = d
                            best_step = step

                # Early exit if we found a good improvement with this step size
                if best_improvement > step * 0.1:
                    break

            if best_i >= 0:
                x[best_i] += dirs[best_dir, 0] * best_step
                y[best_i] += dirs[best_dir, 1] * best_step
                r = compute_radii(x, y)
                curr_val = np.sum(r)
            else:
                break

        return x, y, compute_radii(x, y), np.sum(compute_radii(x, y))

    def hard_subgradient_polish(x, y, num_steps=30, step_size=0.005):
        """Refine the solution using the exact sub-gradient of the hard min function."""
        x = x.copy()
        y = y.copy()

        for _ in range(num_steps):
            if time.time() - start_time > DEADLINE - 2:
                break

            # Compute radii and active constraints
            r_bound = np.minimum(
                np.minimum(x, 1.0 - x),
                np.minimum(y, 1.0 - y)
            )

            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(dist, np.inf)

            half_dist = 0.5 * dist

            grad_x = np.zeros(N)
            grad_y = np.zeros(N)

            # For each circle i, find the active constraint
            for i in range(N):
                min_d = np.min(half_dist[i])
                min_j = np.argmin(half_dist[i])

                if r_bound[i] < min_d:
                    # Boundary is active
                    bounds = [x[i], 1.0-x[i], y[i], 1.0-y[i]]
                    min_idx = np.argmin(bounds)
                    if min_idx == 0:
                        grad_x[i] += 1.0
                    elif min_idx == 1:
                        grad_x[i] -= 1.0
                    elif min_idx == 2:
                        grad_y[i] += 1.0
                    elif min_idx == 3:
                        grad_y[i] -= 1.0
                else:
                    # Neighbor min_j is active
                    d = dist[i, min_j]
                    if d > 1e-10:
                        gx = 0.5 * (x[i] - x[min_j]) / d
                        gy = 0.5 * (y[i] - y[min_j]) / d
                        grad_x[i] += gx
                        grad_y[i] += gy
                        grad_x[min_j] -= gx
                        grad_y[min_j] -= gy

            # Normalize gradient
            norm = np.sqrt(np.sum(grad_x**2) + np.sum(grad_y**2))
            if norm < 1e-10:
                break

            # Adaptive step size
            alpha = step_size / max(norm, 1.0)

            x_new = x + alpha * grad_x
            y_new = y + alpha * grad_y
            x_new = np.clip(x_new, 0.05, 0.95)
            y_new = np.clip(y_new, 0.05, 0.95)

            # Check if improvement
            r_new = compute_radii(x_new, y_new)
            val_new = np.sum(r_new)
            r_curr = compute_radii(x, y)
            val_curr = np.sum(r_curr)

            if val_new > val_curr:
                x = x_new
                y = y_new
            else:
                # Try smaller step
                alpha *= 0.5
                x_new = x + alpha * grad_x
                y_new = y + alpha * grad_y
                x_new = np.clip(x_new, 0.05, 0.95)
                y_new = np.clip(y_new, 0.05, 0.95)
                r_new = compute_radii(x_new, y_new)
                val_new = np.sum(r_new)
                if val_new > val_curr:
                    x = x_new
                    y = y_new
                else:
                    break

        return x, y, compute_radii(x, y), np.sum(compute_radii(x, y))

    def gradient_polish(x0, y0, use_smooth=True, beta_start=20.0, beta_end=500.0, num_steps=5):
        """Final gradient-based refinement using L-BFGS-B with analytic gradient."""
        params0 = np.concatenate([x0, y0])
        bounds = [(0.01, 0.99)] * (2 * N)

        # Geometric progression of betas
        num_steps = max(num_steps, 5)
        betas = np.geomspace(beta_start, beta_end, num_steps)

        current_params = params0.copy()
        best_result = None
        best_val = -np.inf

        def get_val_from_params(params):
            x = params[:N]
            y = params[N:2*N]
            r = compute_radii(x, y)
            if check_validity(x, y, r):
                return np.sum(r)
            for scale in [0.999, 0.99, 0.95, 0.90]:
                r_test = r * scale
                if check_validity(x, y, r_test):
                    return np.sum(r_test)
            return -np.inf

        def update_best(params):
            nonlocal best_result, best_val
            val = get_val_from_params(params)
            if val > best_val:
                best_val = val
                r_hard = compute_radii(params[:N], params[N:])
                if check_validity(params[:N], params[N:], r_hard):
                    best_result = (params[:N].copy(), params[N:].copy(), r_hard, val)
                else:
                    for scale in [0.999, 0.99, 0.98]:
                        r_test = r_hard * scale
                        if check_validity(params[:N], params[N:], r_test):
                            best_result = (params[:N].copy(), params[N:].copy(), r_test, np.sum(r_test))
                            break

        # Initial check
        update_best(current_params)

        for beta in betas:
            if time.time() - start_time > DEADLINE - 5:
                break

            try:
                res = minimize(smooth_objective_and_grad, current_params,
                              method='L-BFGS-B', bounds=bounds,
                              args=(beta,),
                              jac=True,
                              options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100})
                current_params = res.x
                update_best(current_params)
            except Exception:
                pass

        # Final hard polish with very high beta
        if time.time() - start_time < DEADLINE - 2:
            beta_final = min(beta_end * 2, 5000.0)
            try:
                res2 = minimize(smooth_objective_and_grad, current_params,
                                method='L-BFGS-B', bounds=bounds,
                                args=(beta_final,),
                                jac=True,
                                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 100})
                current_params = res2.x
                update_best(current_params)
            except Exception:
                pass

        if best_result is not None:
            return best_result
        return None

    # Initial best: grid configuration
    side = int(np.ceil(np.sqrt(N)))
    best_x = np.zeros(N)
    best_y = np.zeros(N)
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            best_x[idx] = (i + 0.5) / side
            best_y[idx] = (j + 0.5) / side
            idx += 1
    best_r = compute_radii(best_x, best_y)
    best_val = np.sum(best_r)

    # Multi-start with SA + best-fit + hard subgradient + gradient polish
    modes = ['grid', 'hex', 'random', 'voronoi']
    num_starts = 20
    for i in range(num_starts):
        if time.time() - start_time > DEADLINE - 20:
            break

        mode = modes[i % len(modes)]
        if mode == 'voronoi':
            x0, y0 = voronoi_greedy_init(i)
        else:
            x0, y0 = make_initial_guess(i, mode)

        # Run SA
        x_sa, y_sa, r_sa, val_sa = sa_optimize(x0, y0, seed=i, max_iters=2000)

        # Best-fit local search on SA result
        x_bf, y_bf, r_bf, val_bf = best_fit_local_search(x_sa, y_sa)
        
        # Grow radii polish
        if time.time() - start_time < DEADLINE - 15:
            r_grown = _grow_radii(x_bf, y_bf, r_bf, max_iters=200)
            val_grown = _score(x_bf, y_bf, r_grown)
            if val_grown > val_bf:
                r_bf = r_grown
                val_bf = val_grown
        
        # Aggressive radius maximization
        if time.time() - start_time < DEADLINE - 15:
            r_max = _maximize_radii(x_bf, y_bf, r_bf, max_iters=100)
            val_max = _score(x_bf, y_bf, r_max)
            if val_max > val_bf:
                r_bf = r_max
                val_bf = val_max

        # Hard subgradient polish
        if time.time() - start_time < DEADLINE - 15:
            x_hs, y_hs, r_hs, val_hs = hard_subgradient_polish(x_bf, y_bf, num_steps=30, step_size=0.005)
            if val_hs > val_bf:
                x_bf, y_bf, r_bf, val_bf = x_hs, y_hs, r_hs, val_hs

        # Gradient polish on best-fit result
        if time.time() - start_time < DEADLINE - 10:
            polished = gradient_polish(x_bf, y_bf, use_smooth=True)
            if polished is not None:
                xp, yp, rp, vp = polished
                if vp > val_bf:
                    x_sa, y_sa, r_sa, val_sa = xp, yp, rp, vp

        # Update best
        if val_sa > best_val:
            best_val = val_sa
            best_x = x_sa.copy()
            best_y = y_sa.copy()
            best_r = r_sa.copy()

        # Also try best-fit on the initial guess directly (skip SA for some starts)
        if i % 4 == 3 and time.time() - start_time < DEADLINE - 15:
            x_bf2, y_bf2, r_bf2, val_bf2 = best_fit_local_search(x0, y0)
            if time.time() - start_time < DEADLINE - 10:
                x_hs2, y_hs2, r_hs2, val_hs2 = hard_subgradient_polish(x_bf2, y_bf2, num_steps=30, step_size=0.005)
                if val_hs2 > val_bf2:
                    x_bf2, y_bf2, r_bf2, val_bf2 = x_hs2, y_hs2, r_hs2, val_hs2
                if time.time() - start_time < DEADLINE - 10:
                    polished2 = gradient_polish(x_bf2, y_bf2, use_smooth=True)
                    if polished2 is not None:
                        xp2, yp2, rp2, vp2 = polished2
                        if vp2 > val_bf2:
                            val_bf2 = vp2
                            x_bf2, y_bf2, r_bf2 = xp2, yp2, rp2
            if val_bf2 > best_val:
                best_val = val_bf2
                best_x = x_bf2.copy()
                best_y = y_bf2.copy()
                best_r = r_bf2.copy()

    # Final high-precision polish: run multiple passes on the best solution found
    if time.time() - start_time < DEADLINE - 5:
        # First do best-fit on the best solution
        x_bf_final, y_bf_final, r_bf_final, val_bf_final = best_fit_local_search(best_x, best_y)
        if val_bf_final > best_val:
            best_val = val_bf_final
            best_x = x_bf_final
            best_y = y_bf_final
            best_r = r_bf_final

        # Grow radii polish
        if time.time() - start_time < DEADLINE - 5:
            r_grown = _grow_radii(best_x, best_y, best_r, max_iters=500)
            val_grown = _score(best_x, best_y, r_grown)
            if val_grown > best_val:
                best_val = val_grown
                best_r = r_grown
        
        # Aggressive radius maximization
        if time.time() - start_time < DEADLINE - 5:
            r_max = _maximize_radii(best_x, best_y, best_r, max_iters=200)
            val_max = _score(best_x, best_y, r_max)
            if val_max > best_val:
                best_val = val_max
                best_r = r_max

        # Hard subgradient polish
        x_hs_final, y_hs_final, r_hs_final, val_hs_final = hard_subgradient_polish(best_x, best_y, num_steps=50, step_size=0.003)
        if val_hs_final > best_val:
            best_val = val_hs_final
            best_x = x_hs_final
            best_y = y_hs_final
            best_r = r_hs_final

        # Final grow radii
        if time.time() - start_time < DEADLINE - 3:
            r_grown2 = _grow_radii(best_x, best_y, best_r, max_iters=300)
            val_grown2 = _score(best_x, best_y, r_grown2)
            if val_grown2 > best_val:
                best_val = val_grown2
                best_r = r_grown2
        
        # Final aggressive radius maximization
        if time.time() - start_time < DEADLINE - 3:
            r_max2 = _maximize_radii(best_x, best_y, best_r, max_iters=200)
            val_max2 = _score(best_x, best_y, r_max2)
            if val_max2 > best_val:
                best_val = val_max2
                best_r = r_max2

        # Multiple gradient polish passes with increasing precision
        for polish_pass in range(5):
            if time.time() - start_time > DEADLINE - 3:
                break
            beta_s = 50.0 * (10 ** polish_pass)
            beta_e = 1000.0 * (10 ** polish_pass)
            final_polish = gradient_polish(best_x, best_y, use_smooth=True,
                                           beta_start=min(beta_s, 5000.0),
                                           beta_end=min(beta_e, 15000.0),
                                           num_steps=8)
            if final_polish is not None:
                xp, yp, rp, vp = final_polish
                if vp > best_val:
                    best_val = vp
                    best_x = xp
                    best_y = yp
                    best_r = rp
                # Do another best-fit pass after gradient polish
                if time.time() - start_time < DEADLINE - 3:
                    x_bf2, y_bf2, r_bf2, val_bf2 = best_fit_local_search(best_x, best_y)
                    if val_bf2 > best_val:
                        best_val = val_bf2
                        best_x = x_bf2
                        best_y = y_bf2
                        best_r = r_bf2
                # Hard subgradient polish after gradient polish
                if time.time() - start_time < DEADLINE - 3:
                    x_hs2, y_hs2, r_hs2, val_hs2 = hard_subgradient_polish(best_x, best_y, num_steps=30, step_size=0.002)
                    if val_hs2 > best_val:
                        best_val = val_hs2
                        best_x = x_hs2
                        best_y = y_hs2
                        best_r = r_hs2

    # Final validity check
    if not check_validity(best_x, best_y, best_r):
        for scale in np.linspace(0.99, 0.90, 20):
            r_test = best_r * scale
            if check_validity(best_x, best_y, r_test):
                best_r = r_test
                break
        else:
            best_r = compute_radii(best_x, best_y)

    # Final attempt to grow radii after validity check
    if time.time() - start_time < DEADLINE - 2:
        r_final = _grow_radii(best_x, best_y, best_r, max_iters=100)
        if check_validity(best_x, best_y, r_final) and np.sum(r_final) > np.sum(best_r):
            best_r = r_final
        
        # Final aggressive radius maximization
        if time.time() - start_time < DEADLINE - 2:
            r_final2 = _maximize_radii(best_x, best_y, best_r, max_iters=150)
            if check_validity(best_x, best_y, r_final2) and np.sum(r_final2) > np.sum(best_r):
                best_r = r_final2

    return np.column_stack((best_x, best_y, best_r))
