"""
Circle packing in unit square using SA + smooth soft-min gradient polish + hard sub-gradient.
Combines the current program's SA + gradient polish design with the inspiration's
hard sub-gradient polish and improved smooth gradient computation.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 115.0  # Safety margin from 120s limit


def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    def compute_radii(x, y):
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
        r = np.maximum(r, 1e-8)
        return r

    def smooth_objective_and_grad(params, beta=50.0):
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
        max_neg = np.max(neg_beta_c, axis=1, keepdims=True)
        exp_terms = np.exp(neg_beta_c - max_neg)
        log_sum = np.log(np.sum(exp_terms, axis=1, keepdims=True))
        r_soft = -(max_neg + log_sum) / beta
        r_soft = r_soft[:, 0]

        objective_val = -np.sum(r_soft)

        w = exp_terms / np.sum(exp_terms, axis=1, keepdims=True)

        dist_safe = dist.copy()
        np.fill_diagonal(dist_safe, 1.0)
        ux = dx / dist_safe
        uy = dy / dist_safe
        np.fill_diagonal(ux, 0.0)
        np.fill_diagonal(uy, 0.0)

        W_ux = w * ux
        W_T_ux = w.T * ux
        W_uy = w * uy
        W_T_uy = w.T * uy

        grad_x = 0.5 * (W_ux.sum(axis=1) + W_T_ux.sum(axis=1))
        grad_y = 0.5 * (W_uy.sum(axis=1) + W_T_uy.sum(axis=1))

        # Boundary gradient contribution
        beta_bound = 500.0
        boundaries = np.stack([x, 1.0 - x, y, 1.0 - y], axis=1)
        neg_bb = -beta_bound * boundaries
        max_bb = np.max(neg_bb, axis=1, keepdims=True)
        exp_bb = np.exp(neg_bb - max_bb)
        softmax_b = exp_bb / np.sum(exp_bb, axis=1, keepdims=True)

        wii = w[np.arange(N), np.arange(N)]
        grad_x += wii * (softmax_b[:, 0] - softmax_b[:, 1])
        grad_y += wii * (softmax_b[:, 2] - softmax_b[:, 3])

        grad = np.concatenate([grad_x, grad_y])
        return objective_val, grad

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

    def hard_subgradient_polish(x, y, num_steps=30, step_size=0.003):
        """
        Refine using exact sub-gradient of the hard min function.
        Snaps to the exact local optimum defined by the current contact graph.
        """
        x = x.copy()
        y = y.copy()
        best_val = np.sum(compute_radii(x, y))
        best_x = x.copy()
        best_y = y.copy()

        for step in range(num_steps):
            if elapsed() > DEADLINE - 2:
                break

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

            for i in range(N):
                min_d = np.min(half_dist[i])
                min_j = np.argmin(half_dist[i])

                if r_bound[i] < min_d:
                    bounds = [x[i], 1.0 - x[i], y[i], 1.0 - y[i]]
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
                    d = dist[i, min_j]
                    if d > 1e-10:
                        gx = 0.5 * (x[i] - x[min_j]) / d
                        gy = 0.5 * (y[i] - y[min_j]) / d
                        grad_x[i] += gx
                        grad_y[i] += gy
                        grad_x[min_j] -= gx
                        grad_y[min_j] -= gy

            norm = np.sqrt(np.sum(grad_x**2) + np.sum(grad_y**2))
            if norm < 1e-10:
                break

            alpha = step_size / (1.0 + 0.1 * step)
            x_new = x + alpha * grad_x
            y_new = y + alpha * grad_y
            x_new = np.clip(x_new, 0.001, 0.999)
            y_new = np.clip(y_new, 0.001, 0.999)

            r_new = compute_radii(x_new, y_new)
            val_new = np.sum(r_new)

            if val_new > best_val:
                best_val = val_new
                best_x = x_new.copy()
                best_y = y_new.copy()
                x = x_new
                y = y_new

        return best_x, best_y, compute_radii(best_x, best_y), best_val

    def grow_radii(x, y, r, max_iter=50):
        best_r = r.copy()
        best_val = np.sum(r)
        for _ in range(max_iter):
            if elapsed() > DEADLINE - 5:
                break
            improved = False
            for i in range(N):
                r_bound_i = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
                min_neighbor = np.inf
                for j in range(N):
                    if j == i:
                        continue
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    min_neighbor = min(min_neighbor, 0.5*(d - r[j]))
                max_r_i = min(r_bound_i, min_neighbor)
                if max_r_i > r[i] + 1e-10:
                    r[i] = max_r_i
                    improved = True
            if not improved:
                break
        if np.sum(r) > best_val:
            if check_validity(x, y, r):
                return r, np.sum(r)
        return best_r, best_val

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
        elif mode == 'ring':
            cx, cy = 0.5, 0.5
            rc = 0.12
            x = np.zeros(N)
            y = np.zeros(N)
            for k in range(6):
                angle = k * np.pi / 3
                x[k] = cx + rc * np.cos(angle)
                y[k] = cy + rc * np.sin(angle)
            x[0] = cx
            y[0] = cy
            n_outer = N - 6
            for k in range(n_outer):
                angle = 2 * np.pi * k / n_outer + rng.uniform(-0.1, 0.1)
                rr = 0.38 + rng.uniform(-0.02, 0.02)
                x[6+k] = cx + rr * np.cos(angle)
                y[6+k] = cy + rr * np.sin(angle)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
        elif mode == 'spiral':
            # Archimedean spiral initialization
            x = np.zeros(N)
            y = np.zeros(N)
            for k in range(N):
                t = k / (N - 1)
                theta = 2 * np.pi * 2.5 * t + rng.uniform(-0.2, 0.2)
                rr = 0.05 + 0.35 * t
                x[k] = 0.5 + rr * np.cos(theta)
                y[k] = 0.5 + rr * np.sin(theta)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
        elif mode == 'diagonal':
            # Diagonal band initialization
            x = np.zeros(N)
            y = np.zeros(N)
            for k in range(N):
                t = k / (N - 1)
                x[k] = 0.1 + 0.8 * t + rng.uniform(-0.03, 0.03)
                y[k] = 0.1 + 0.8 * t + rng.uniform(-0.03, 0.03)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
        else:  # random
            x = rng.uniform(0.1, 0.9, N)
            y = rng.uniform(0.1, 0.9, N)
        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        return x, y

    def sa_optimize(x0, y0, seed, max_iters=6000):
        rng = np.random.RandomState(seed)
        x = x0.copy()
        y = y0.copy()
        r = compute_radii(x, y)
        best_val = np.sum(r)
        best_x = x.copy()
        best_y = y.copy()

        T_start = 0.03
        T_end = 0.0002
        cool_rate = (T_end / T_start) ** (1.0 / max_iters)

        repulse_k = 1.5
        boundary_k = 4.0

        for it in range(max_iters):
            if elapsed() > DEADLINE - 15:
                break

            T = T_start * (cool_rate ** it)
            r_curr = compute_radii(x, y)
            curr_val = np.sum(r_curr)

            fx = np.zeros(N)
            fy = np.zeros(N)

            fx += boundary_k * np.clip(0.07 - x, 0, None)
            fx -= boundary_k * np.clip(x - 0.93, 0, None)
            fy += boundary_k * np.clip(0.07 - y, 0, None)
            fy -= boundary_k * np.clip(y - 0.93, 0, None)

            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(d, 1e10)

            f_mag = repulse_k / (d**2 + 0.006)
            np.fill_diagonal(f_mag, 0)

            fx += np.sum(-f_mag * dx / d, axis=1)
            fy += np.sum(-f_mag * dy / d, axis=1)

            scale = 0.012 * T / T_start + 0.003
            dx_prop = fx * scale + rng.normal(0, scale, N)
            dy_prop = fy * scale + rng.normal(0, scale, N)

            x_new = x + dx_prop
            y_new = y + dy_prop
            x_new = np.clip(x_new, 0.02, 0.98)
            y_new = np.clip(y_new, 0.02, 0.98)

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

    def gradient_polish(x0, y0, beta_start=15.0, beta_end=2000.0, num_steps=8):
        params0 = np.concatenate([x0, y0])
        bounds = [(0.005, 0.995)] * (2 * N)
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
            for scale in [0.999, 0.995, 0.99, 0.98, 0.95]:
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
                    for scale in [0.999, 0.995, 0.99, 0.98, 0.95]:
                        r_test = r_hard * scale
                        if check_validity(params[:N], params[N:], r_test):
                            best_result = (params[:N].copy(), params[N:].copy(), r_test, np.sum(r_test))
                            break

        init_val = get_val_from_params(current_params)
        if init_val > best_val:
            store_best(current_params, init_val)

        for beta in betas:
            if elapsed() > DEADLINE - 4:
                break

            def objective_smooth(params):
                val, _ = smooth_objective_and_grad(params, beta=beta)
                return val

            def grad_smooth(params):
                _, g = smooth_objective_and_grad(params, beta=beta)
                return g

            try:
                res = minimize(objective_smooth, current_params, method='L-BFGS-B',
                               jac=grad_smooth,
                               bounds=bounds,
                               options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 40})
                current_params = res.x
                val = get_val_from_params(current_params)
                if val > best_val:
                    store_best(current_params, val)
            except Exception:
                pass

        # Hard sub-gradient polish after smooth polish
        if elapsed() < DEADLINE - 3:
            x_hs = current_params[:N].copy()
            y_hs = current_params[N:2*N].copy()
            x_hs, y_hs, r_hs, val_hs = hard_subgradient_polish(x_hs, y_hs, num_steps=25, step_size=0.002)
            if check_validity(x_hs, y_hs, r_hs):
                if val_hs > best_val:
                    store_best(np.concatenate([x_hs, y_hs]), val_hs)

        # Final hard polish
        if elapsed() < DEADLINE - 2:
            def objective_hard(params):
                x = params[:N]
                y = params[N:2*N]
                r = compute_radii(x, y)
                return -np.sum(r)

            try:
                res2 = minimize(objective_hard, current_params, method='L-BFGS-B',
                                bounds=bounds,
                                options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40})
                current_params = res2.x
                val = get_val_from_params(current_params)
                if val > best_val:
                    store_best(current_params, val)
            except Exception:
                pass

        if best_result is not None:
            return best_result
        return None

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

    # Multi-start with SA
    modes = ['grid', 'hex', 'ring', 'spiral', 'diagonal', 'random']
    num_starts = 30
    for i in range(num_starts):
        if elapsed() > DEADLINE - 25:
            break

        mode = modes[i % len(modes)]
        x0, y0 = make_initial_guess(i, mode)

        x_sa, y_sa, r_sa, val_sa = sa_optimize(x0, y0, seed=i, max_iters=6000)

        if val_sa > best_val:
            best_val = val_sa
            best_x = x_sa
            best_y = y_sa
            best_r = r_sa

        # Gradient polish
        if elapsed() < DEADLINE - 15:
            polished = gradient_polish(x_sa, y_sa)
            if polished is not None:
                xp, yp, rp, vp = polished
                if vp > best_val:
                    best_val = vp
                    best_x = xp
                    best_y = yp
                    best_r = rp

    # Final high-precision polish
    if elapsed() < DEADLINE - 5:
        final_polish = gradient_polish(best_x, best_y, beta_start=50.0, beta_end=5000.0, num_steps=12)
        if final_polish is not None:
            xp, yp, rp, vp = final_polish
            if vp > best_val:
                best_val = vp
                best_x = xp
                best_y = yp
                best_r = rp

    # Hard sub-gradient polish on the best solution
    if elapsed() < DEADLINE - 3:
        x_hs, y_hs, r_hs, val_hs = hard_subgradient_polish(best_x, best_y, num_steps=40, step_size=0.001)
        if val_hs > best_val:
            best_val = val_hs
            best_x = x_hs
            best_y = y_hs
            best_r = r_hs

    # Radius growth polish
    if elapsed() < DEADLINE - 2:
        grown_r, grown_val = grow_radii(best_x, best_y, best_r.copy())
        if grown_val > best_val:
            best_val = grown_val
            best_r = grown_r

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
