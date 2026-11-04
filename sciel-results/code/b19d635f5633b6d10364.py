"""
Circle packing in unit square: 26 circles, maximize sum of radii.
Uses simulated annealing with physics-based forces, discrete local search,
and smooth gradient-based polish (L-BFGS-B with soft-min surrogate).
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 142.0  # Safety margin from 150s limit


def construct():
    start_time = time.time()

    def compute_radii(x, y):
        """Compute max valid radii from centers."""
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

    def check_validity(x, y, r):
        """Check if circles are valid (disjoint, inside unit square)."""
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

    def smooth_objective_and_grad(params, beta=50.0):
        """
        Smooth objective: -sum(r_i) where r_i = softmin over constraints.
        Uses log-sum-exp for numerical stability.
        Returns (objective, gradient).
        """
        x = params[:N]
        y = params[N:2*N]

        # Pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-30)
        np.fill_diagonal(dist, 1e10)

        # Constraint matrix: c[i,j] = 0.5*dist(i,j) for i!=j, c[i,i] = boundary_dist(i)
        c = 0.5 * dist

        # Boundary distances (4 sides)
        bounds_4 = np.stack([x, 1.0 - x, y, 1.0 - y], axis=1)  # (N, 4)
        beta_b = max(beta * 2, 100.0)
        neg_bb = -beta_b * bounds_4
        max_bb = np.max(neg_bb, axis=1, keepdims=True)
        exp_bb = np.exp(neg_bb - max_bb)
        sum_bb = np.sum(exp_bb, axis=1, keepdims=True)
        r_bound_smooth = -(max_bb[:, 0] + np.log(sum_bb[:, 0])) / beta_b

        np.fill_diagonal(c, r_bound_smooth)

        # Soft-min: r_i = -(1/beta) * log(sum_j exp(-beta * c_ij))
        neg_beta_c = -beta * c
        max_neg = np.max(neg_beta_c, axis=1, keepdims=True)
        exp_terms = np.exp(neg_beta_c - max_neg)
        sum_exp = np.sum(exp_terms, axis=1, keepdims=True)
        log_sum = np.log(sum_exp)
        r = -(max_neg[:, 0] + log_sum[:, 0]) / beta

        # Weights for gradient
        w = exp_terms / sum_exp  # (N, N)

        # Gradient of c_ij w.r.t. parameters
        dist_safe = dist.copy()
        np.fill_diagonal(dist_safe, 1.0)

        dc_dx_i = 0.5 * dx / dist_safe
        dc_dy_i = 0.5 * dy / dist_safe
        dc_dx_j = -0.5 * dx / dist_safe
        dc_dy_j = -0.5 * dy / dist_safe

        # Diagonal: gradient of boundary soft-min
        w_b = exp_bb / sum_bb  # (N, 4)
        grad_rb_dx = w_b[:, 0] - w_b[:, 1]
        grad_rb_dy = w_b[:, 2] - w_b[:, 3]

        np.fill_diagonal(dc_dx_i, grad_rb_dx)
        np.fill_diagonal(dc_dy_i, grad_rb_dy)
        np.fill_diagonal(dc_dx_j, grad_rb_dx)
        np.fill_diagonal(dc_dy_j, grad_rb_dy)

        # dr_i/dp = sum_j w_ij * dc_ij/dp
        # Total grad w.r.t. x_k = sum_i dr_i/dx_k
        # dr_i/dx_k = sum_j w_ij * (dc_dx_i[i,j] if k==i else dc_dx_j[i,j] if k==j else 0)
        row_sum_dx_i = np.sum(w * dc_dx_i, axis=1)
        row_sum_dy_i = np.sum(w * dc_dy_i, axis=1)
        col_sum_dx_j = np.sum(w * dc_dx_j, axis=0)
        col_sum_dy_j = np.sum(w * dc_dy_j, axis=0)

        diag_w = np.diag(w)
        diag_dc_dx = np.diag(dc_dx_i)
        diag_dc_dy = np.diag(dc_dy_i)

        dr_dx = col_sum_dx_j + row_sum_dx_i - diag_w * diag_dc_dx
        dr_dy = col_sum_dy_j + row_sum_dy_i - diag_w * diag_dc_dy

        grad = np.concatenate([-dr_dx, -dr_dy])
        obj_val = -np.sum(r)
        return obj_val, grad

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
            x += rng.uniform(-0.06, 0.06, N)
            y += rng.uniform(-0.06, 0.06, N)
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
            x += rng.uniform(-0.04, 0.04, N)
            y += rng.uniform(-0.04, 0.04, N)
        elif mode == 'random':
            x = rng.uniform(0.1, 0.9, N)
            y = rng.uniform(0.1, 0.9, N)
        elif mode == 'ring':
            # Place circles in concentric rings
            x = np.zeros(N)
            y = np.zeros(N)
            x[0] = 0.5
            y[0] = 0.5
            # Ring 1: 6 circles
            for k in range(6):
                angle = 2 * np.pi * k / 6
                x[k+1] = 0.5 + 0.25 * np.cos(angle)
                y[k+1] = 0.5 + 0.25 * np.sin(angle)
            # Ring 2: 12 circles
            for k in range(12):
                angle = 2 * np.pi * k / 12 + np.pi / 12
                x[k+7] = 0.5 + 0.45 * np.cos(angle)
                y[k+7] = 0.5 + 0.45 * np.sin(angle)
            # Remaining 7
            for k in range(7):
                angle = 2 * np.pi * k / 7 + np.pi / 7
                x[k+19] = 0.5 + 0.35 * np.cos(angle)
                y[k+19] = 0.5 + 0.35 * np.sin(angle)
            x += rng.uniform(-0.03, 0.03, N)
            y += rng.uniform(-0.03, 0.03, N)
        else:
            x = rng.uniform(0.1, 0.9, N)
            y = rng.uniform(0.1, 0.9, N)

        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        return x, y

    def sa_optimize(x0, y0, seed, max_iters=5000):
        """Simulated annealing with physics-based forces."""
        rng = np.random.RandomState(seed)
        x = x0.copy()
        y = y0.copy()
        r = compute_radii(x, y)
        best_val = np.sum(r)
        best_x = x.copy()
        best_y = y.copy()

        T_start = 0.04
        T_end = 0.0002
        cool_rate = (T_end / T_start) ** (1.0 / max_iters)

        repulse_k = 1.5
        boundary_k = 8.0

        for it in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break

            T = T_start * (cool_rate ** it)

            r_curr = compute_radii(x, y)
            curr_val = np.sum(r_curr)

            # Forces
            fx = np.zeros(N)
            fy = np.zeros(N)

            # Boundary repulsion
            fx += boundary_k * np.clip(0.08 - x, 0, None)
            fx -= boundary_k * np.clip(x - 0.92, 0, None)
            fy += boundary_k * np.clip(0.08 - y, 0, None)
            fy -= boundary_k * np.clip(y - 0.92, 0, None)

            # Pairwise repulsion
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(d, 1e10)

            f_mag = repulse_k / (d**2 + 0.005)
            np.fill_diagonal(f_mag, 0)

            fx += np.sum(-f_mag * dx / d, axis=1)
            fy += np.sum(-f_mag * dy / d, axis=1)

            # Propose move
            scale = 0.015 * T / T_start + 0.003
            dx_prop = fx * scale + rng.normal(0, scale, N)
            dy_prop = fy * scale + rng.normal(0, scale, N)

            x_new = np.clip(x + dx_prop, 0.03, 0.97)
            y_new = np.clip(y + dy_prop, 0.03, 0.97)

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

    def best_fit_local_search(x, y, max_rounds=80):
        """Discrete best-fit local search."""
        step_sizes = [0.015, 0.008, 0.004, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002]

        directions = []
        for k in range(16):
            angle = 2 * np.pi * k / 16
            directions.append((np.cos(angle), np.sin(angle)))

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

                        if x_new_i < 0.005 or x_new_i > 0.995:
                            continue
                        if y_new_i < 0.005 or y_new_i > 0.995:
                            continue

                        x_temp = x.copy()
                        y_temp = y.copy()
                        x_temp[i] = x_new_i
                        y_temp[i] = y_new_i

                        r_new = compute_radii(x_temp, y_temp)
                        new_val = np.sum(r_new)
                        improvement = new_val - curr_val

                        if improvement > best_improvement + 1e-12:
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

    def gradient_polish(x0, y0, beta_start=20.0, beta_end=1000.0, num_steps=6):
        """Gradient-based polish using L-BFGS-B with increasing beta."""
        params0 = np.concatenate([x0, y0])
        bounds = [(0.002, 0.998)] * (2 * N)

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
            for scale in [0.9999, 0.9995, 0.999, 0.998, 0.995, 0.99, 0.98, 0.95, 0.90]:
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
                    for scale in [0.9999, 0.9995, 0.999, 0.998, 0.995, 0.99, 0.98]:
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
                              options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-13, 'maxls': 100})
                current_params = res.x
                update_best(current_params)
            except Exception:
                pass

        # Final high-precision pass
        if time.time() - start_time < DEADLINE - 3:
            beta_final = min(beta_end * 5, 5000.0)
            try:
                res2 = minimize(smooth_objective_and_grad, current_params,
                                method='L-BFGS-B', bounds=bounds,
                                args=(beta_final,),
                                jac=True,
                                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100})
                current_params = res2.x
                update_best(current_params)
            except Exception:
                pass

        return best_result

    # Initial best: grid
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

    # Multi-start
    modes = ['grid', 'hex', 'random', 'ring', 'grid', 'hex', 'random', 'ring']
    num_starts = 40
    for i in range(num_starts):
        if time.time() - start_time > DEADLINE - 30:
            break

        mode = modes[i % len(modes)]
        x0, y0 = make_initial_guess(i, mode)

        # SA
        x_sa, y_sa, r_sa, val_sa = sa_optimize(x0, y0, seed=i, max_iters=5000)

        # Best-fit
        x_bf, y_bf, r_bf, val_bf = best_fit_local_search(x_sa, y_sa)

        # Gradient polish
        if time.time() - start_time < DEADLINE - 15:
            polished = gradient_polish(x_bf, y_bf, beta_start=15.0, beta_end=1000.0, num_steps=6)
            if polished is not None:
                xp, yp, rp, vp = polished
                if vp > val_bf:
                    x_sa, y_sa, r_sa, val_sa = xp, yp, rp, vp

        if val_sa > best_val:
            best_val = val_sa
            best_x = x_sa.copy()
            best_y = y_sa.copy()
            best_r = r_sa.copy()

        # Extra: best-fit on initial for some starts
        if i % 4 == 3 and time.time() - start_time < DEADLINE - 20:
            x_bf2, y_bf2, r_bf2, val_bf2 = best_fit_local_search(x0, y0)
            if time.time() - start_time < DEADLINE - 12:
                polished2 = gradient_polish(x_bf2, y_bf2, beta_start=15.0, beta_end=1000.0, num_steps=6)
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

    # Final polish on best
    if time.time() - start_time < DEADLINE - 10:
        x_bf_final, y_bf_final, r_bf_final, val_bf_final = best_fit_local_search(best_x, best_y)
        if val_bf_final > best_val:
            best_val = val_bf_final
            best_x = x_bf_final
            best_y = y_bf_final
            best_r = r_bf_final

        for polish_pass in range(5):
            if time.time() - start_time > DEADLINE - 5:
                break
            beta_s = 20.0 * (5 ** polish_pass)
            beta_e = 500.0 * (5 ** polish_pass)
            final_polish = gradient_polish(best_x, best_y,
                                           beta_start=min(beta_s, 5000.0),
                                           beta_end=min(beta_e, 20000.0),
                                           num_steps=6)
            if final_polish is not None:
                xp, yp, rp, vp = final_polish
                if vp > best_val:
                    best_val = vp
                    best_x = xp
                    best_y = yp
                    best_r = rp
                if time.time() - start_time < DEADLINE - 5:
                    x_bf2, y_bf2, r_bf2, val_bf2 = best_fit_local_search(best_x, best_y)
                    if val_bf2 > best_val:
                        best_val = val_bf2
                        best_x = x_bf2
                        best_y = y_bf2
                        best_r = r_bf2

    # Final validity check
    if not check_validity(best_x, best_y, best_r):
        for scale in np.linspace(0.999, 0.90, 50):
            r_test = best_r * scale
            if check_validity(best_x, best_y, r_test):
                best_r = r_test
                break
        else:
            best_r = compute_radii(best_x, best_y)

    return np.column_stack((best_x, best_y, best_r))
