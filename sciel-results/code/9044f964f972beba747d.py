"""
Circle packing in unit square: pack 26 circles, maximize sum of radii.
Uses a smooth soft-min surrogate for the nonsmooth min objective,
with L-BFGS-B gradient polish, SA global search, and radius growth.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 140.0  # Safety margin from 150s limit


def construct():
    start_time = time.time()

    def _elapsed():
        return time.time() - start_time

    def _remaining():
        return DEADLINE - _elapsed()

    def _compute_radii(x, y):
        """Compute exact radii from centers."""
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
        r = np.maximum(r, 1e-9)
        return r

    def _is_valid(x, y, r, tol=1e-9):
        """Check if configuration is valid."""
        if np.any(r < -tol):
            return False
        if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
            return False
        if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
            return False
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(min_dist, 0)
        if np.any(dist < min_dist - tol):
            return False
        return True

    def _soft_objective(params, beta):
        """
        Smooth approximation of -sum(min_j c_ij) where c_ij are constraints.
        Uses log-sum-exp for soft-min. Returns (value, gradient).
        """
        x = params[:N]
        y = params[N:2*N]

        # Boundary constraints
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        # Pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-30)
        np.fill_diagonal(dist, 1e10)

        # Constraint matrix: c[i,j] = 0.5*dist(i,j) for i!=j, c[i,i] = r_bound[i]
        c = 0.5 * dist.copy()
        np.fill_diagonal(c, r_bound)

        # Soft-min: r_i = -1/beta * log(sum_j exp(-beta * c[i,j]))
        neg_beta_c = -beta * c
        max_neg = np.max(neg_beta_c, axis=1, keepdims=True)
        exp_terms = np.exp(neg_beta_c - max_neg)
        sum_exp = np.sum(exp_terms, axis=1, keepdims=True)
        log_sum = np.log(sum_exp)
        r = -(max_neg + log_sum) / beta
        r = r[:, 0]

        # Weights w[i,j] = exp(-beta*c[i,j]) / sum_j' exp(-beta*c[i,j'])
        w = exp_terms / sum_exp  # (N, N)

        # Gradient computation
        dist_safe = dist.copy()
        np.fill_diagonal(dist_safe, 1.0)

        # dc/dx_i for off-diagonal: 0.5 * (x_i - x_j) / dist
        dc_dx_i = 0.5 * dx / dist_safe
        dc_dy_i = 0.5 * dy / dist_safe
        dc_dx_j = -0.5 * dx / dist_safe
        dc_dy_j = -0.5 * dy / dist_safe

        # Diagonal: gradient of r_bound[i]
        bounds_4 = np.stack([x, 1.0 - x, y, 1.0 - y], axis=1)
        beta_b = min(beta * 2, 500.0)
        neg_bb = -beta_b * bounds_4
        max_bb = np.max(neg_bb, axis=1, keepdims=True)
        exp_bb = np.exp(neg_bb - max_bb)
        sum_bb = np.sum(exp_bb, axis=1, keepdims=True)
        w_b = exp_bb / sum_bb

        grad_rb_dx = w_b[:, 0] - w_b[:, 1]
        grad_rb_dy = w_b[:, 2] - w_b[:, 3]

        np.fill_diagonal(dc_dx_i, grad_rb_dx)
        np.fill_diagonal(dc_dy_i, grad_rb_dy)
        np.fill_diagonal(dc_dx_j, grad_rb_dx)
        np.fill_diagonal(dc_dy_j, grad_rb_dy)

        # Total gradient of sum(r) w.r.t. x_k
        dr_dx_self = np.sum(w * dc_dx_i, axis=1)
        dr_dy_self = np.sum(w * dc_dy_i, axis=1)
        dr_dx_cross = np.sum(w * dc_dx_j, axis=0)
        dr_dy_cross = np.sum(w * dc_dy_j, axis=0)

        dr_dx = dr_dx_self + dr_dx_cross - np.diag(w) * np.diag(dc_dx_i)
        dr_dy = dr_dy_self + dr_dy_cross - np.diag(w) * np.diag(dc_dy_i)

        # Objective is -sum(r), gradient is -dr/dparams
        grad = np.concatenate([-dr_dx, -dr_dy])
        obj_val = -np.sum(r)
        return obj_val, grad

    def _score(x, y):
        """Compute the score (sum of radii) for a configuration."""
        r = _compute_radii(x, y)
        if _is_valid(x, y, r):
            return np.sum(r), r
        # Try to fix by scaling
        for scale in [0.9999, 0.999, 0.99, 0.98, 0.95, 0.90]:
            r_test = r * scale
            if _is_valid(x, y, r_test):
                return np.sum(r_test), r_test
        return 0.0, r

    def _project_feasible(x, y, r, max_iters=200):
        """Project to feasible region by reducing overlapping radii."""
        x = x.copy()
        y = y.copy()
        r = r.copy()
        for _ in range(max_iters):
            if _remaining() < 3:
                break
            if _is_valid(x, y, r):
                break
            # Find the most violated pair
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            min_dist = r[:, None] + r[None, :]
            np.fill_diagonal(dist, np.inf)
            np.fill_diagonal(min_dist, 0)
            violation = min_dist - dist
            np.fill_diagonal(violation, 0)
            max_viol = np.max(violation)
            if max_viol < 1e-12:
                # Check boundaries
                if np.any(x - r < 0) or np.any(x + r > 1) or np.any(y - r < 0) or np.any(y + r > 1):
                    r = np.minimum(r, np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
                    continue
                break
            i, j = np.unravel_index(np.argmax(violation), (N, N))
            # Reduce the larger radius
            if r[i] >= r[j]:
                r[i] = max(0.0, r[i] - max_viol / 2)
            else:
                r[j] = max(0.0, r[j] - max_viol / 2)
        return x, y, r

    def _grow_radii(x, y, r, max_iters=500, step=0.0005):
        """Grow radii while maintaining feasibility."""
        x = x.copy()
        y = y.copy()
        r = r.copy()
        for _ in range(max_iters):
            if _remaining() < 3:
                break
            improved = False
            for i in range(N):
                old_r = r[i]
                r[i] = old_r + step
                if _is_valid(x, y, r):
                    improved = True
                else:
                    r[i] = old_r
            if not improved:
                # Try smaller step
                step *= 0.5
                if step < 1e-7:
                    break
        return x, y, r, np.sum(r)

    def _grid_init(seed):
        rng = np.random.RandomState(seed)
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
        x += rng.uniform(-0.05, 0.05, N)
        y += rng.uniform(-0.05, 0.05, N)
        return np.clip(x, 0.05, 0.95), np.clip(y, 0.05, 0.95)

    def _hex_init(seed):
        rng = np.random.RandomState(seed)
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
        x += rng.uniform(-0.03, 0.03, N)
        y += rng.uniform(-0.03, 0.03, N)
        return np.clip(x, 0.05, 0.95), np.clip(y, 0.05, 0.95)

    def _random_init(seed):
        rng = np.random.RandomState(seed)
        x = rng.uniform(0.1, 0.9, N)
        y = rng.uniform(0.1, 0.9, N)
        return x, y

    def _structured_init(seed):
        """Structured initialization with symmetry breaking."""
        rng = np.random.RandomState(seed)
        # Start with a 5x5 grid (25 points) + 1 extra
        x = np.zeros(N)
        y = np.zeros(N)
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx >= N:
                    break
                x[idx] = (i + 0.5) / 5.0
                y[idx] = (j + 0.5) / 5.0
                idx += 1
        # Add one more at center-ish
        if N > 25:
            x[25] = 0.5 + rng.uniform(-0.1, 0.1)
            y[25] = 0.5 + rng.uniform(-0.1, 0.1)
        # Jitter
        x += rng.uniform(-0.02, 0.02, N)
        y += rng.uniform(-0.02, 0.02, N)
        return np.clip(x, 0.05, 0.95), np.clip(y, 0.05, 0.95)

    def _repulsion_relax_init(seed):
        """Initialize via repulsion-based relaxation from random start."""
        rng = np.random.RandomState(seed)
        x = rng.uniform(0.1, 0.9, N)
        y = rng.uniform(0.1, 0.9, N)
        for it in range(200):
            if _remaining() < 5:
                break
            fx = np.zeros(N)
            fy = np.zeros(N)
            # Boundary
            fx += 10.0 * np.clip(0.08 - x, 0, None)
            fx -= 10.0 * np.clip(x - 0.92, 0, None)
            fy += 10.0 * np.clip(0.08 - y, 0, None)
            fy -= 10.0 * np.clip(y - 0.92, 0, None)
            # Repulsion
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(d, 1e10)
            f_mag = 1.0 / (d**2 + 0.01)
            np.fill_diagonal(f_mag, 0)
            fx += np.sum(-f_mag * dx / d, axis=1)
            fy += np.sum(-f_mag * dy / d, axis=1)
            # Centering
            fx += 0.1 * (0.5 - x)
            fy += 0.1 * (0.5 - y)
            alpha = 0.01 / (1 + it * 0.01)
            x += alpha * fx
            y += alpha * fy
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
        return x, y

    def _sa_optimize(x0, y0, seed, max_iters=2000):
        """Simulated annealing."""
        rng = np.random.RandomState(seed)
        x = x0.copy()
        y = y0.copy()
        val, r = _score(x, y)
        best_val = val
        best_x = x.copy()
        best_y = y.copy()

        T_start = 0.03
        T_end = 0.0003
        cool_rate = (T_end / T_start) ** (1.0 / max_iters)

        for it in range(max_iters):
            if _remaining() < 5:
                break
            T = T_start * (cool_rate ** it)

            # Compute forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            fx += 8.0 * np.clip(0.08 - x, 0, None)
            fx -= 8.0 * np.clip(x - 0.92, 0, None)
            fy += 8.0 * np.clip(0.08 - y, 0, None)
            fy -= 8.0 * np.clip(y - 0.92, 0, None)

            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(d, 1e10)
            f_mag = 1.5 / (d**2 + 0.01)
            np.fill_diagonal(f_mag, 0)
            fx += np.sum(-f_mag * dx / d, axis=1)
            fy += np.sum(-f_mag * dy / d, axis=1)

            scale = 0.015 * T / T_start + 0.003
            dx_prop = fx * scale + rng.normal(0, scale, N)
            dy_prop = fy * scale + rng.normal(0, scale, N)

            x_new = np.clip(x + dx_prop, 0.05, 0.95)
            y_new = np.clip(y + dy_prop, 0.05, 0.95)

            new_val, r_new = _score(x_new, y_new)
            delta = new_val - val
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                x = x_new
                y = y_new
                val = new_val
                if new_val > best_val:
                    best_val = new_val
                    best_x = x.copy()
                    best_y = y.copy()

        return best_x, best_y, _compute_radii(best_x, best_y), best_val

    def _best_fit_local(x, y, max_rounds=40):
        """Best-fit local search."""
        step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
        dirs = np.array([
            [1, 0], [-1, 0], [0, 1], [0, -1],
            [1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)],
            [-1/np.sqrt(2), 1/np.sqrt(2)], [-1/np.sqrt(2), -1/np.sqrt(2)]
        ])
        x = x.copy()
        y = y.copy()
        val, r = _score(x, y)

        for rnd in range(max_rounds):
            if _remaining() < 4:
                break
            best_imp = 1e-10
            best_i, best_d, best_s = -1, -1, 0

            for step in step_sizes:
                for i in range(N):
                    for d in range(8):
                        xn = x[i] + dirs[d, 0] * step
                        yn = y[i] + dirs[d, 1] * step
                        if xn < 0.01 or xn > 0.99 or yn < 0.01 or yn > 0.99:
                            continue
                        x[i] = xn
                        y[i] = yn
                        nv, _ = _score(x, y)
                        imp = nv - val
                        if imp > best_imp:
                            best_imp = imp
                            best_i, best_d, best_s = i, d, step
                        x[i] = x[i] - dirs[d, 0] * step
                        y[i] = y[i] - dirs[d, 1] * step

                if best_imp > step * 0.05:
                    break

            if best_i >= 0:
                x[best_i] += dirs[best_d, 0] * best_s
                y[best_i] += dirs[best_d, 1] * best_s
                val, r = _score(x, y)
            else:
                break

        return x, y, _compute_radii(x, y), val

    def _gradient_polish(x0, y0, beta_start=20.0, beta_end=500.0, num_steps=5, maxiter=200):
        """L-BFGS-B gradient polish with increasing beta."""
        params0 = np.concatenate([x0, y0])
        bounds = [(0.01, 0.99)] * (2 * N)
        betas = np.geomspace(beta_start, beta_end, num_steps)

        current = params0.copy()
        best_val = -np.inf
        best_result = None

        def update_best(params):
            nonlocal best_val, best_result
            x = params[:N]
            y = params[N:]
            val, r = _score(x, y)
            if val > best_val:
                best_val = val
                best_result = (x.copy(), y.copy(), r.copy(), val)

        update_best(current)

        for beta in betas:
            if _remaining() < 5:
                break
            try:
                res = minimize(_soft_objective, current,
                              method='L-BFGS-B', bounds=bounds,
                              args=(beta,), jac=True,
                              options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 40})
                current = res.x
                update_best(current)
            except Exception:
                pass

        if best_result is not None:
            return best_result
        return None

    def _hard_subgradient(x, y, num_steps=30, step_size=0.004):
        """Subgradient method on the exact min function."""
        x = x.copy()
        y = y.copy()
        for _ in range(num_steps):
            if _remaining() < 3:
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
                    mi = np.argmin(bounds)
                    if mi == 0:
                        grad_x[i] += 1.0
                    elif mi == 1:
                        grad_x[i] -= 1.0
                    elif mi == 2:
                        grad_y[i] += 1.0
                    elif mi == 3:
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
            if norm < 1e-12:
                break
            alpha = step_size / max(norm, 1.0)
            x_new = np.clip(x + alpha * grad_x, 0.05, 0.95)
            y_new = np.clip(y + alpha * grad_y, 0.05, 0.95)
            val_new, _ = _score(x_new, y_new)
            val_curr, _ = _score(x, y)
            if val_new > val_curr:
                x, y = x_new, y_new
            else:
                alpha *= 0.5
                x_new = np.clip(x + alpha * grad_x, 0.05, 0.95)
                y_new = np.clip(y + alpha * grad_y, 0.05, 0.95)
                val_new, _ = _score(x_new, y_new)
                if val_new > val_curr:
                    x, y = x_new, y_new
                else:
                    break

        return x, y, _compute_radii(x, y), _score(x, y)[0]

    # === Main optimization loop ===
    # Initial best
    best_x, best_y = _grid_init(0)
    best_val, best_r = _score(best_x, best_y)

    inits = [
        ('grid', _grid_init),
        ('hex', _hex_init),
        ('random', _random_init),
        ('structured', _structured_init),
        ('repulsion', _repulsion_relax_init),
    ]

    num_starts = 60
    for i in range(num_starts):
        if _remaining() < 20:
            break

        name, init_fn = inits[i % len(inits)]
        x0, y0 = init_fn(i)

        # SA
        x_sa, y_sa, r_sa, val_sa = _sa_optimize(x0, y0, seed=i, max_iters=1500)

        # Best-fit
        x_bf, y_bf, r_bf, val_bf = _best_fit_local(x_sa, y_sa)

        # Hard subgradient
        if _remaining() > 12:
            x_hs, y_hs, r_hs, val_hs = _hard_subgradient(x_bf, y_bf, num_steps=25, step_size=0.004)
            if val_hs > val_bf:
                x_bf, y_bf, r_bf, val_bf = x_hs, y_hs, r_hs, val_hs

        # Gradient polish
        if _remaining() > 10:
            polished = _gradient_polish(x_bf, y_bf, beta_start=20.0, beta_end=500.0, num_steps=4, maxiter=150)
            if polished is not None:
                xp, yp, rp, vp = polished
                if vp > val_bf:
                    x_bf, y_bf, r_bf, val_bf = xp, yp, rp, vp

        # Grow radii
        if _remaining() > 5:
            x_gr, y_gr, r_gr, val_gr = _grow_radii(x_bf, y_bf, r_bf, max_iters=200, step=0.0005)
            if val_gr > val_bf:
                x_bf, y_bf, r_bf, val_bf = x_gr, y_gr, r_gr, val_gr

        # Update best
        if val_bf > best_val:
            best_val = val_bf
            best_x = x_bf.copy()
            best_y = y_bf.copy()
            best_r = r_bf.copy()

    # Final high-precision polish on best solution
    if _remaining() > 10:
        x_bf, y_bf, r_bf, val_bf = _best_fit_local(best_x, best_y, max_rounds=30)
        if val_bf > best_val:
            best_val = val_bf
            best_x, best_y, best_r = x_bf, y_bf, r_bf

        if _remaining() > 8:
            x_hs, y_hs, r_hs, val_hs = _hard_subgradient(best_x, best_y, num_steps=40, step_size=0.002)
            if val_hs > best_val:
                best_val = val_hs
                best_x, best_y, best_r = x_hs, y_hs, r_hs

        # Multiple gradient polish passes
        for polish_pass in range(5):
            if _remaining() < 5:
                break
            beta_s = 50.0 * (10 ** polish_pass)
            beta_e = 1000.0 * (10 ** polish_pass)
            polished = _gradient_polish(best_x, best_y,
                                        beta_start=min(beta_s, 3000.0),
                                        beta_end=min(beta_e, 8000.0),
                                        num_steps=6, maxiter=250)
            if polished is not None:
                xp, yp, rp, vp = polished
                if vp > best_val:
                    best_val = vp
                    best_x, best_y, best_r = xp, yp, rp

            if _remaining() < 4:
                break
            x_bf2, y_bf2, r_bf2, val_bf2 = _best_fit_local(best_x, best_y, max_rounds=20)
            if val_bf2 > best_val:
                best_val = val_bf2
                best_x, best_y, best_r = x_bf2, y_bf2, r_bf2

            if _remaining() < 4:
                break
            x_hs2, y_hs2, r_hs2, val_hs2 = _hard_subgradient(best_x, best_y, num_steps=20, step_size=0.001)
            if val_hs2 > best_val:
                best_val = val_hs2
                best_x, best_y, best_r = x_hs2, y_hs2, r_hs2

            if _remaining() < 3:
                break
            x_gr, y_gr, r_gr, val_gr = _grow_radii(best_x, best_y, best_r, max_iters=300, step=0.0003)
            if val_gr > best_val:
                best_val = val_gr
                best_x, best_y, best_r = x_gr, y_gr, r_gr

    # Final validity check
    if not _is_valid(best_x, best_y, best_r):
        for scale in np.linspace(0.999, 0.90, 30):
            r_test = best_r * scale
            if _is_valid(best_x, best_y, r_test):
                best_r = r_test
                break
        else:
            best_r = _compute_radii(best_x, best_y)

    return np.column_stack((best_x, best_y, best_r))
