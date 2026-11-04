"""
Circle packing in unit square using Voronoi-based inradius optimization.
Combines SA exploration with smooth gradient polish and explicit gradient computation.
Uses soft-min surrogate for better convergence, then hard polish.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 140.0  # Safety margin from 150s limit


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
        r_bound_arr = np.stack([x, 1.0 - x, y, 1.0 - y], axis=1)  # (N, 4)

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-30)
        np.fill_diagonal(dist, np.inf)
        half_dist = 0.5 * dist  # (N, N)

        # For each i, constraints: 4 boundary + (N-1) neighbor distances
        # Soft-min: r_i ≈ -(1/beta) * log(sum_k exp(-beta * c_ik))

        nb_bound = -beta * r_bound_arr  # (N, 4)
        nb_neigh = -beta * half_dist
        nb_neigh = np.where(np.eye(N, dtype=bool), -np.inf, nb_neigh)

        max_bound = np.max(nb_bound, axis=1)  # (N,)
        max_neigh = np.max(nb_neigh, axis=1)  # (N,)
        max_all = np.maximum(max_bound, max_neigh)  # (N,)

        exp_bound = np.exp(nb_bound - max_all[:, None])  # (N, 4)
        sum_bound = np.sum(exp_bound, axis=1)  # (N,)

        exp_neigh = np.exp(nb_neigh - max_all[:, None])  # (N, N)
        sum_neigh = np.sum(exp_neigh, axis=1)  # (N,)

        total_sum = sum_bound + sum_neigh  # (N,)
        r = -(max_all + np.log(total_sum + 1e-30)) / beta  # (N,)
        r = np.maximum(r, 1e-6)
        return r

    def compute_objective_and_grad(params):
        """Compute -sum(radii) and its gradient using active constraint identification."""
        x = params[:N].copy()
        y = params[N:2*N].copy()

        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-30)
        np.fill_diagonal(dist, np.inf)
        r_neighbor = 0.5 * np.min(dist, axis=1)

        r = np.minimum(r_bound, r_neighbor)
        r = np.maximum(r, 1e-10)

        obj_val = -np.sum(r)

        grad = np.zeros(2 * N)

        for i in range(N):
            if r_bound[i] <= r_neighbor[i] + 1e-12:
                # Boundary constraint active
                bounds = [x[i], 1.0 - x[i], y[i], 1.0 - y[i]]
                min_idx = np.argmin(bounds)
                if min_idx == 0:  # x is min -> dr/dx = +1
                    grad[i] = -1.0
                elif min_idx == 1:  # 1-x is min -> dr/dx = -1
                    grad[i] = 1.0
                elif min_idx == 2:  # y is min -> dr/dy = +1
                    grad[N + i] = -1.0
                else:  # 1-y is min -> dr/dy = -1
                    grad[N + i] = 1.0
            else:
                # Neighbor constraint active
                dx_i = x[i] - x
                dy_i = y[i] - y
                dist_i = np.sqrt(dx_i**2 + dy_i**2 + 1e-30)
                dist_i[i] = np.inf
                j = np.argmin(dist_i)

                d = dist_i[j]
                if d > 1e-15:
                    gx = 0.5 * (x[i] - x[j]) / d
                    gy = 0.5 * (y[i] - y[j]) / d
                    grad[i] += -gx
                    grad[N + i] += -gy
                    grad[j] += gx
                    grad[N + j] += gy

        return obj_val, grad

    def compute_objective_and_grad_smooth(params, beta=100.0):
        """Smooth version using log-sum-exp for the min operation with analytic gradient."""
        x = params[:N].copy()
        y = params[N:2*N].copy()

        r_bound_arr = np.stack([x, 1.0 - x, y, 1.0 - y], axis=1)  # (N, 4)

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-30)
        np.fill_diagonal(dist, np.inf)
        half_dist = 0.5 * dist  # (N, N)

        # Soft-min computation
        nb_bound = -beta * r_bound_arr  # (N, 4)
        nb_neigh = -beta * half_dist
        nb_neigh = np.where(np.eye(N, dtype=bool), -np.inf, nb_neigh)

        max_bound = np.max(nb_bound, axis=1)
        max_neigh = np.max(nb_neigh, axis=1)
        max_all = np.maximum(max_bound, max_neigh)

        exp_bound = np.exp(nb_bound - max_all[:, None])
        sum_bound = np.sum(exp_bound, axis=1)

        exp_neigh = np.exp(nb_neigh - max_all[:, None])
        sum_neigh = np.sum(exp_neigh, axis=1)

        total_sum = sum_bound + sum_neigh
        r = -(max_all + np.log(total_sum + 1e-30)) / beta
        r = np.maximum(r, 1e-6)

        obj_val = -np.sum(r)

        # Gradient of soft-min
        # d(r_i)/d(c_ik) = -exp(-beta*c_ik) / sum_j exp(-beta*c_ij)
        # where c_ik are the constraints

        grad = np.zeros(2 * N)

        # Weights for each constraint
        w_bound = exp_bound / total_sum[:, None]  # (N, 4)
        w_neigh = exp_neigh / total_sum[:, None]  # (N, N)

        for i in range(N):
            # Boundary contributions
            # c_i0 = x_i, dc/dx_i = 1
            # c_i1 = 1-x_i, dc/dx_i = -1
            # c_i2 = y_i, dc/dy_i = 1
            # c_i3 = 1-y_i, dc/dy_i = -1
            # d(r_i)/dx_i = -w_i0 * 1 + w_i1 * (-1) = -w_i0 - w_i1
            # Wait: r_i = -(1/beta)*log(sum exp(-beta*c_ik))
            # d(r_i)/d(c_ik) = -(1/beta) * (-beta * exp(-beta*c_ik) / sum) = exp(-beta*c_ik)/sum = w_ik
            # So d(r_i)/dx_i = w_i0 * d(c_i0)/dx_i + w_i1 * d(c_i1)/dx_i + ...
            # = w_i0 * 1 + w_i1 * (-1) + w_i2 * 0 + w_i3 * 0
            # = w_i0 - w_i1
            # Similarly for y: w_i2 - w_i3

            grad[i] += -(w_bound[i, 0] - w_bound[i, 1])  # negative because we minimize -sum(r)
            grad[N + i] += -(w_bound[i, 2] - w_bound[i, 3])

            # Neighbor contributions
            # c_i(4+j) = 0.5 * dist(i,j) for j != i
            # d(c_ij)/dx_i = 0.5 * (x_i - x_j) / dist(i,j)
            # d(c_ij)/dx_j = 0.5 * (x_j - x_i) / dist(i,j)

            for j in range(N):
                if i == j:
                    continue
                d = dist[i, j]
                if d < 1e-15:
                    continue
                w = w_neigh[i, j]
                gx_i = 0.5 * (x[i] - x[j]) / d
                gy_i = 0.5 * (y[i] - y[j]) / d
                gx_j = 0.5 * (x[j] - x[i]) / d
                gy_j = 0.5 * (y[j] - y[i]) / d

                grad[i] += -w * gx_i
                grad[N + i] += -w * gy_i
                grad[j] += -w * gx_j
                grad[N + j] += -w * gy_j

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
        elif mode == 'spiral':
            x = np.zeros(N)
            y = np.zeros(N)
            for k in range(N):
                t = k / (N - 1)
                angle = t * 6 * np.pi
                radius = 0.45 * t
                x[k] = 0.5 + radius * np.cos(angle)
                y[k] = 0.5 + radius * np.sin(angle)
            x += rng.uniform(-0.03, 0.03, N)
            y += rng.uniform(-0.03, 0.03, N)
        elif mode == 'diagonal':
            x = np.zeros(N)
            y = np.zeros(N)
            for k in range(N):
                t = k / (N - 1)
                x[k] = 0.1 + 0.8 * t
                y[k] = 0.1 + 0.8 * (1.0 - t)
            x += rng.uniform(-0.05, 0.05, N)
            y += rng.uniform(-0.05, 0.05, N)
        else:  # random
            x = rng.uniform(0.1, 0.9, N)
            y = rng.uniform(0.1, 0.9, N)

        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        return x, y

    def sa_optimize(x0, y0, seed, max_iters=3000):
        """Simulated annealing to optimize center positions."""
        rng = np.random.RandomState(seed)
        x = x0.copy()
        y = y0.copy()
        r = compute_radii(x, y)
        best_val = np.sum(r)
        best_x = x.copy()
        best_y = y.copy()

        T_start = 0.03
        T_end = 0.001
        cool_rate = (T_end / T_start) ** (1.0 / max_iters)

        repulse_k = 1.5
        boundary_k = 3.0

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

            scale = 0.015 * T / T_start + 0.003
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

    def gradient_polish(x0, y0, beta_start=10.0, beta_end=5000.0, num_steps=7):
        """Final gradient-based refinement using L-BFGS-B with smooth objective."""
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

        def store_best(params, val):
            nonlocal best_result, best_val
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

        init_val = get_val_from_params(current_params)
        if init_val > best_val:
            store_best(current_params, init_val)

        for beta in betas:
            if time.time() - start_time > DEADLINE - 5:
                break

            def objective_smooth(params):
                x = params[:N]
                y = params[N:2*N]
                r = compute_radii_smooth(x, y, beta=beta)
                return -np.sum(r)

            def grad_smooth(params):
                x = params[:N]
                y = params[N:2*N]
                # Use analytic gradient
                val, grad = compute_objective_and_grad_smooth(params, beta=beta)
                return grad

            try:
                res = minimize(objective_smooth, current_params, jac=grad_smooth,
                               method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-10})
                current_params = res.x
                val = get_val_from_params(current_params)
                if val > best_val:
                    store_best(current_params, val)
            except Exception:
                pass

        # Final hard polish with explicit gradients
        if time.time() - start_time < DEADLINE - 2:
            def objective_hard(params):
                x = params[:N]
                y = params[N:2*N]
                r = compute_radii(x, y)
                return -np.sum(r)

            def grad_hard(params):
                x = params[:N]
                y = params[N:2*N]
                val, grad = compute_objective_and_grad(params)
                return grad

            try:
                res2 = minimize(objective_hard, current_params, jac=grad_hard,
                                method='L-BFGS-B', bounds=bounds,
                                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12})
                current_params = res2.x
                val = get_val_from_params(current_params)
                if val > best_val:
                    store_best(current_params, val)
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

    # Multi-start with SA
    modes = ['grid', 'hex', 'random', 'spiral', 'diagonal']
    num_starts = 25
    for i in range(num_starts):
        if time.time() - start_time > DEADLINE - 20:
            break

        mode = modes[i % len(modes)]
        x0, y0 = make_initial_guess(i, mode)

        x_sa, y_sa, r_sa, val_sa = sa_optimize(x0, y0, seed=i, max_iters=3000)

        if val_sa > best_val:
            best_val = val_sa
            best_x = x_sa
            best_y = y_sa
            best_r = r_sa

        # Gradient polish on SA result
        if time.time() - start_time < DEADLINE - 12:
            polished = gradient_polish(x_sa, y_sa)
            if polished is not None:
                xp, yp, rp, vp = polished
                if vp > best_val:
                    best_val = vp
                    best_x = xp
                    best_y = yp
                    best_r = rp

    # Final high-precision polish
    if time.time() - start_time < DEADLINE - 10:
        final_polish = gradient_polish(best_x, best_y, beta_start=200.0, beta_end=5000.0, num_steps=5)
        if final_polish is not None:
            xp, yp, rp, vp = final_polish
            if vp > best_val:
                best_val = vp
                best_x = xp
                best_y = yp
                best_r = rp

    # Post-optimization radius growth
    if time.time() - start_time < DEADLINE - 5:
        for grow_iter in range(100):
            if time.time() - start_time > DEADLINE - 3:
                break
            improved = False
            for i in range(N):
                dx_all = best_x[i] - best_x
                dy_all = best_y[i] - best_y
                dist_all = np.sqrt(dx_all**2 + dy_all**2)
                dist_all[i] = np.inf
                max_r = np.minimum(
                    np.minimum(best_x[i], 1.0 - best_x[i]),
                    np.minimum(best_y[i], 1.0 - best_y[i])
                )
                for j in range(N):
                    if j == i:
                        continue
                    max_r = min(max_r, 0.5 * (dist_all[j] - best_r[j]))
                new_r_i = min(max_r, best_r[i] * 1.001)
                if new_r_i > best_r[i] + 1e-12:
                    old_r = best_r[i]
                    best_r[i] = new_r_i
                    if check_validity(best_x, best_y, best_r):
                        improved = True
                    else:
                        best_r[i] = old_r
            if not improved:
                break

    # Final validity check
    if not check_validity(best_x, best_y, best_r):
        for scale in np.linspace(0.999, 0.90, 30):
            r_test = best_r * scale
            if check_validity(best_x, best_y, r_test):
                best_r = r_test
                break
        else:
            best_r = compute_radii(best_x, best_y)

    return np.column_stack((best_x, best_y, best_r))
