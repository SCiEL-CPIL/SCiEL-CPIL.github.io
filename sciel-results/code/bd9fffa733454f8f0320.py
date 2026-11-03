"""
Circle packing in unit square using simulated annealing with repulsion forces.
Centers are optimized; radii are derived from local geometry.
SA provides global exploration to escape local minima.
Smooth soft-min surrogate with analytic gradient for fast L-BFGS-B polish.
Discrete best-fit local search for deterministic neighborhood improvement.
Added: Delaunay-style force-directed initialization and Hard Sub-gradient polish.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 115.0  # Safety margin from 120s limit


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

    def compute_radii_smooth(x, y, beta=50.0):
        """Smooth approximation of radii using log-sum-exp soft-min."""
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)

        half_dist = 0.5 * dist
        full_constraints = half_dist.copy()
        np.fill_diagonal(full_constraints, r_bound)

        neg_beta = -beta * full_constraints
        max_neg_row = np.max(neg_beta, axis=1, keepdims=True)
        log_sum_row = np.log(np.sum(np.exp(neg_beta - max_neg_row), axis=1, keepdims=True))
        soft_min_all = -(max_neg_row + log_sum_row) / beta
        soft_min_all = soft_min_all[:, 0]

        r = np.maximum(soft_min_all, 1e-6)
        return r, soft_min_all

    def smooth_objective_and_grad(params, beta=50.0):
        """Compute -sum(r) and its gradient with respect to params = [x, y]."""
        x = params[:N]
        y = params[N:2*N]

        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-30)
        np.fill_diagonal(dist, 1e10)

        half_dist = 0.5 * dist
        full_constraints = half_dist.copy()
        np.fill_diagonal(full_constraints, r_bound)

        neg_beta_c = -beta * full_constraints
        max_nb = np.max(neg_beta_c, axis=1, keepdims=True)
        exp_terms = np.exp(neg_beta_c - max_nb)
        log_sum = np.log(np.sum(exp_terms, axis=1, keepdims=True))
        r = -(max_nb + log_sum) / beta
        r = r[:, 0]

        w = exp_terms / np.sum(exp_terms, axis=1, keepdims=True)

        dist_safe = dist.copy()
        np.fill_diagonal(dist_safe, 1.0)
        
        d_c_dx_i = 0.5 * dx / dist_safe
        d_c_dy_i = 0.5 * dy / dist_safe
        d_c_dx_j = -0.5 * dx / dist_safe
        d_c_dy_j = -0.5 * dy / dist_safe

        all_bounds = np.stack([x, 1.0-x, y, 1.0-y], axis=1)
        min_idx = np.argmin(all_bounds, axis=1)
        
        grad_rb_dx = np.zeros(N)
        grad_rb_dy = np.zeros(N)
        for i in range(N):
            if min_idx[i] == 0: grad_rb_dx[i] = 1.0
            elif min_idx[i] == 1: grad_rb_dx[i] = -1.0
            elif min_idx[i] == 2: grad_rb_dy[i] = 1.0
            elif min_idx[i] == 3: grad_rb_dy[i] = -1.0

        np.fill_diagonal(d_c_dx_i, grad_rb_dx)
        np.fill_diagonal(d_c_dy_i, grad_rb_dy)
        np.fill_diagonal(d_c_dx_j, grad_rb_dx)
        np.fill_diagonal(d_c_dy_j, grad_rb_dy)

        row_sum_dx_i = np.sum(w * d_c_dx_i, axis=1)
        row_sum_dy_i = np.sum(w * d_c_dy_i, axis=1)
        col_sum_dx_j = np.sum(w * d_c_dx_j, axis=0)
        col_sum_dy_j = np.sum(w * d_c_dy_j, axis=0)

        dr_dx = row_sum_dx_i + col_sum_dx_j
        dr_dy = row_sum_dy_i + col_sum_dy_j

        grad = np.concatenate([-dr_dx, -dr_dy])
        obj_val = -np.sum(r)
        return obj_val, grad

    def hard_subgradient_polish(x, y, num_steps=20, step_size=0.005):
        """
        Refine the solution using the exact sub-gradient of the hard min function.
        This helps snap to the exact local optimum defined by the current contact graph.
        """
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
            # Constraints for circle i: [half_dist[i, j] for j != i, r_bound[i]]
            # We need the gradient of the min constraint for each i
            
            grad_x = np.zeros(N)
            grad_y = np.zeros(N)
            
            # For each circle i, find the active constraint
            # We can vectorize this partially, but explicit loop is safer for logic
            for i in range(N):
                # Find min neighbor distance
                min_d = np.min(half_dist[i])
                min_j = np.argmin(half_dist[i])
                
                # Check if boundary is smaller
                if r_bound[i] < min_d:
                    # Boundary is active
                    # Gradient of r_bound[i]
                    # r_bound = min(x, 1-x, y, 1-y)
                    bounds = [x[i], 1.0-x[i], y[i], 1.0-y[i]]
                    min_idx = np.argmin(bounds)
                    if min_idx == 0: # x
                        grad_x[i] += 1.0
                    elif min_idx == 1: # 1-x
                        grad_x[i] -= 1.0
                    elif min_idx == 2: # y
                        grad_y[i] += 1.0
                    elif min_idx == 3: # 1-y
                        grad_y[i] -= 1.0
                else:
                    # Neighbor min_j is active
                    # Constraint is 0.5 * dist(i, min_j)
                    # Gradient w.r.t x_i: 0.5 * (x_i - x_j) / dist
                    # Gradient w.r.t x_j: 0.5 * (x_j - x_i) / dist
                    d = dist[i, min_j]
                    if d > 1e-10:
                        gx = 0.5 * (x[i] - x[min_j]) / d
                        gy = 0.5 * (y[i] - y[min_j]) / d
                        grad_x[i] += gx
                        grad_y[i] += gy
                        grad_x[min_j] -= gx
                        grad_y[min_j] -= gy

            # We want to maximize sum(r), so we move in the direction of the gradient
            # However, the gradient of sum(r) is the sum of the individual gradients
            # Note: The sub-gradient of min is the gradient of the active term.
            # So grad_x, grad_y above IS the sub-gradient of sum(r) w.r.t positions.
            
            # Normalize gradient to avoid huge steps
            norm = np.sqrt(np.sum(grad_x**2) + np.sum(grad_y**2))
            if norm < 1e-10:
                break
                
            # Adaptive step size: shrink if we are oscillating or if improvement is small
            # For simplicity, use a fixed small step, but we can do a line search
            # Here we use a fixed step with damping
            alpha = step_size / norm
            
            x_new = x + alpha * grad_x
            y_new = y + alpha * grad_y
            
            # Clip to bounds
            x_new = np.clip(x_new, 0.01, 0.99)
            y_new = np.clip(y_new, 0.01, 0.99)
            
            # Check validity and improvement
            r_old = compute_radii(x, y)
            val_old = np.sum(r_old)
            
            r_new = compute_radii(x_new, y_new)
            val_new = np.sum(r_new)
            
            if val_new > val_old - 1e-10:
                x = x_new
                y = y_new
            else:
                # If it didn't improve, try a smaller step
                alpha *= 0.5
                x_new = x + alpha * grad_x
                y_new = y + alpha * grad_y
                x_new = np.clip(x_new, 0.01, 0.99)
                y_new = np.clip(y_new, 0.01, 0.99)
                r_new = compute_radii(x_new, y_new)
                val_new = np.sum(r_new)
                if val_new > val_old - 1e-10:
                    x = x_new
                    y = y_new
                else:
                    # No improvement possible in this direction, break or random jitter
                    break

        return x, y

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
        elif mode == 'delaunay':
            # Force-directed relaxation to spread points
            x = rng.uniform(0.1, 0.9, N)
            y = rng.uniform(0.1, 0.9, N)
            for _ in range(50):
                fx = np.zeros(N)
                fy = np.zeros(N)
                # Repulsion
                dx = x[:, None] - x[None, :]
                dy = y[:, None] - y[None, :]
                d = np.sqrt(dx**2 + dy**2)
                np.fill_diagonal(d, 1e10)
                f_mag = 1.0 / (d**2 + 0.01)
                np.fill_diagonal(f_mag, 0)
                fx += np.sum(-f_mag * dx / d, axis=1)
                fy += np.sum(-f_mag * dy / d, axis=1)
                # Boundary attraction
                fx += 5.0 * np.clip(0.1 - x, 0, None)
                fx -= 5.0 * np.clip(x - 0.9, 0, None)
                fy += 5.0 * np.clip(0.1 - y, 0, None)
                fy -= 5.0 * np.clip(y - 0.9, 0, None)
                
                x += 0.05 * fx
                y += 0.05 * fy
                x = np.clip(x, 0.1, 0.9)
                y = np.clip(y, 0.1, 0.9)
        else:  # random
            x = rng.uniform(0.1, 0.9, N)
            y = rng.uniform(0.1, 0.9, N)

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

        T_start = 0.05
        T_end = 0.0005
        cool_rate = (T_end / T_start) ** (1.0 / max_iters)

        repulse_k = 2.0
        boundary_k = 5.0

        for it in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break

            T = T_start * (cool_rate ** it)

            r_curr = compute_radii(x, y)
            curr_val = np.sum(r_curr)

            fx = np.zeros(N)
            fy = np.zeros(N)

            fx += boundary_k * np.clip(0.1 - x, 0, None)
            fx -= boundary_k * np.clip(x - 0.9, 0, None)
            fy += boundary_k * np.clip(0.1 - y, 0, None)
            fy -= boundary_k * np.clip(y - 0.9, 0, None)

            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(d, 1e10)

            f_mag = repulse_k / (d**2 + 0.01)
            np.fill_diagonal(f_mag, 0)

            fx += np.sum(-f_mag * dx / d, axis=1)
            fy += np.sum(-f_mag * dy / d, axis=1)

            scale = 0.02 * T / T_start + 0.005
            dx_prop = fx * scale + rng.normal(0, scale, N)
            dy_prop = fy * scale + rng.normal(0, scale, N)

            x_new = x + dx_prop
            y_new = y + dy_prop

            x_new = np.clip(x_new, 0.05, 0.95)
            y_new = np.clip(y_new, 0.05, 0.95)

            r_new = compute_radii(x_new, y_new)
            new_val = np.sum(r_new)

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

    def best_fit_local_search(x, y, step_sizes=None, max_rounds=80):
        """Discrete best-fit local search with adaptive step sizes."""
        if step_sizes is None:
            step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
        
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        directions = [(dx/np.sqrt(2), dy/np.sqrt(2)) for dx, dy in directions]

        x = x.copy()
        y = y.copy()
        r = compute_radii(x, y)
        curr_val = np.sum(r)

        for rnd in range(max_rounds):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            best_improvement = 0.0
            best_move = None
            
            for step in step_sizes:
                for i in range(N):
                    for dx_dir, dy_dir in directions:
                        x_new_i = x[i] + dx_dir * step
                        y_new_i = y[i] + dy_dir * step
                        
                        if x_new_i < 0.01 or x_new_i > 0.99:
                            continue
                        if y_new_i < 0.01 or y_new_i > 0.99:
                            continue
                        
                        x_temp = x.copy()
                        y_temp = y.copy()
                        x_temp[i] = x_new_i
                        y_temp[i] = y_new_i
                        
                        r_new = compute_radii(x_temp, y_temp)
                        new_val = np.sum(r_new)
                        improvement = new_val - curr_val
                        
                        if improvement > best_improvement + 1e-10:
                            best_improvement = improvement
                            best_move = (i, dx_dir, dy_dir, step)
            
            if best_move is not None:
                i, dx_dir, dy_dir, step = best_move
                x[i] += dx_dir * step
                y[i] += dy_dir * step
                r = compute_radii(x, y)
                curr_val = np.sum(r)
            else:
                break

        return x, y, compute_radii(x, y), np.sum(compute_radii(x, y))

    def gradient_polish(x0, y0, use_smooth=True, beta_start=20.0, beta_end=500.0, num_steps=5):
        """Final gradient-based refinement using L-BFGS-B with analytic gradient."""
        params0 = np.concatenate([x0, y0])
        bounds = [(0.01, 0.99)] * (2 * N)
        
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

        update_best(current_params)

        for beta in betas:
            if time.time() - start_time > DEADLINE - 5:
                break
                
            try:
                res = minimize(smooth_objective_and_grad, current_params, 
                              method='L-BFGS-B', bounds=bounds,
                              args=(beta,),
                              jac=True,
                              options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 50})
                current_params = res.x
                update_best(current_params)
            except Exception:
                pass

        if time.time() - start_time < DEADLINE - 2:
            beta_final = min(beta_end * 2, 2000.0)
            try:
                res2 = minimize(smooth_objective_and_grad, current_params,
                                method='L-BFGS-B', bounds=bounds,
                                args=(beta_final,),
                                jac=True,
                                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
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

    # Multi-start with SA + best-fit + gradient polish
    modes = ['grid', 'hex', 'random', 'delaunay']
    num_starts = 50  # Increased starts
    for i in range(num_starts):
        if time.time() - start_time > DEADLINE - 20:
            break

        mode = modes[i % len(modes)]
        x0, y0 = make_initial_guess(i, mode)

        # Run SA
        x_sa, y_sa, r_sa, val_sa = sa_optimize(x0, y0, seed=i, max_iters=3000)

        # Best-fit local search on SA result
        x_bf, y_bf, r_bf, val_bf = best_fit_local_search(x_sa, y_sa)

        # Gradient polish on best-fit result
        if time.time() - start_time < DEADLINE - 10:
            polished = gradient_polish(x_bf, y_bf, use_smooth=True)
            if polished is not None:
                xp, yp, rp, vp = polished
                if vp > val_bf:
                    x_sa, y_sa, r_sa, val_sa = xp, yp, rp, vp

        # NEW: Hard Sub-gradient polish
        if time.time() - start_time < DEADLINE - 5:
            x_hp, y_hp = hard_subgradient_polish(x_sa, y_sa, num_steps=10, step_size=0.005)
            r_hp = compute_radii(x_hp, y_hp)
            val_hp = np.sum(r_hp)
            if val_hp > val_sa:
                x_sa, y_sa, r_sa, val_sa = x_hp, y_hp, r_hp, val_hp

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

    # Final high-precision polish: run gradient polish on the best solution found
    if time.time() - start_time < DEADLINE - 5:
        x_bf_final, y_bf_final, r_bf_final, val_bf_final = best_fit_local_search(best_x, best_y)
        if val_bf_final > best_val:
            best_val = val_bf_final
            best_x = x_bf_final
            best_y = y_bf_final
            best_r = r_bf_final

        for polish_pass in range(3):
            if time.time() - start_time > DEADLINE - 3:
                break
            beta_s = 50.0 * (10 ** polish_pass)
            beta_e = 1000.0 * (10 ** polish_pass)
            final_polish = gradient_polish(best_x, best_y, use_smooth=True, 
                                           beta_start=min(beta_s, 2000.0), 
                                           beta_end=min(beta_e, 5000.0), 
                                           num_steps=6)
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
                # Final Hard Polish
                if time.time() - start_time < DEADLINE - 2:
                    x_hp, y_hp = hard_subgradient_polish(best_x, best_y, num_steps=5, step_size=0.002)
                    r_hp = compute_radii(x_hp, y_hp)
                    val_hp = np.sum(r_hp)
                    if val_hp > best_val:
                        best_val = val_hp
                        best_x = x_hp
                        best_y = y_hp
                        best_r = r_hp

    # Final validity check
    if not check_validity(best_x, best_y, best_r):
        for scale in np.linspace(0.99, 0.90, 20):
            r_test = best_r * scale
            if check_validity(best_x, best_y, r_test):
                best_r = r_test
                break
        else:
            best_r = compute_radii(best_x, best_y)

    return np.column_stack((best_x, best_y, best_r))
