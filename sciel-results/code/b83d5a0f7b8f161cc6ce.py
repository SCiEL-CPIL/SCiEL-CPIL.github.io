"""
Circle packing in unit square using smooth soft-min objective with L-BFGS-B.
Multi-start with hexagonal, grid, and random initializations.
Iterative beta sharpening for convergence to hard constraints.
Final high-precision polish with exact radius computation.
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
        r = np.maximum(r, 1e-8)
        return r

    def soft_objective_and_grad(params, beta):
        """Smooth -sum(r) objective and its analytic gradient.
        params = [x_0..x_{N-1}, y_0..y_{N-1}]
        r_i = softmin_j(c_ij) where c_ij = 0.5*dist(i,j) for j!=i, c_ii = r_bound_i
        r_i = -(1/beta) * log(sum_j exp(-beta * c_ij))
        """
        x = params[:N]
        y = params[N:2*N]

        # Boundary distances
        b_x = np.minimum(x, 1.0 - x)
        b_y = np.minimum(y, 1.0 - y)
        r_bound = np.minimum(b_x, b_y)

        # Pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-30)
        np.fill_diagonal(dist, 1e10)

        # Constraint matrix c[i,j]
        c = 0.5 * dist
        np.fill_diagonal(c, r_bound)

        # Soft-min: r_i = -(1/beta) * log(sum_j exp(-beta * c[i,j]))
        neg_beta_c = -beta * c
        max_neg = np.max(neg_beta_c, axis=1, keepdims=True)
        exp_terms = np.exp(neg_beta_c - max_neg)
        sum_exp = np.sum(exp_terms, axis=1, keepdims=True)
        log_sum = np.log(sum_exp)
        r = -(max_neg + log_sum) / beta
        r = r[:, 0]

        # Weights w[i,j] = exp(-beta*c[i,j]) / sum_j' exp(-beta*c[i,j'])
        w = exp_terms / sum_exp  # (N, N)

        # Gradient of c[i,j] w.r.t. x_k
        # For i != j: dc_ij/dx_i = 0.5*(x_i-x_j)/dist_ij, dc_ij/dx_j = -0.5*(x_i-x_j)/dist_ij
        # For i == j: c_ii = r_bound_i = min(x_i, 1-x_i, y_i, 1-y_i)
        #   Use soft-min over 4 boundaries for smooth gradient

        dist_safe = dist.copy()
        np.fill_diagonal(dist_safe, 1.0)

        # Off-diagonal: dc_ij/dx_i = 0.5*dx[i,j]/dist[i,j]
        dc_dx_i = 0.5 * dx / dist_safe
        dc_dy_i = 0.5 * dy / dist_safe
        dc_dx_j = -0.5 * dx / dist_safe
        dc_dy_j = -0.5 * dy / dist_safe

        # Diagonal: soft-min gradient for r_bound_i
        # r_bound_i = softmin(x_i, 1-x_i, y_i, 1-y_i) with high beta
        beta_b = 200.0
        bounds_4 = np.stack([x, 1.0 - x, y, 1.0 - y], axis=1)  # (N, 4)
        neg_bb = -beta_b * bounds_4
        max_bb = np.max(neg_bb, axis=1, keepdims=True)
        exp_bb = np.exp(neg_bb - max_bb)
        sum_bb = np.sum(exp_bb, axis=1, keepdims=True)
        w_b = exp_bb / sum_bb  # (N, 4)

        # d(r_bound_i)/dx_i = w_b[i,0]*1 + w_b[i,1]*(-1)
        grad_rb_dx = w_b[:, 0] - w_b[:, 1]
        # d(r_bound_i)/dy_i = w_b[i,2]*1 + w_b[i,3]*(-1)
        grad_rb_dy = w_b[:, 2] - w_b[:, 3]

        # Set diagonal entries
        np.fill_diagonal(dc_dx_i, grad_rb_dx)
        np.fill_diagonal(dc_dy_i, grad_rb_dy)
        np.fill_diagonal(dc_dx_j, grad_rb_dx)
        np.fill_diagonal(dc_dy_j, grad_rb_dy)

        # dr_i/dx_k = sum_j w[i,j] * dc_ij/dx_k
        # For k = i: sum_j w[i,j] * dc_dx_i[i,j]
        # For k != i: w[i,k] * dc_dx_j[i,k]
        #
        # Total dr/dx_k = sum_i [w[i,k] * dc_dx_j[i,k]] + sum_i [w[i,i] * dc_dx_i[i,i]] (for i=k only in first sum)
        # More carefully:
        # dr_dx[k] = sum_i sum_j w[i,j] * dc_ij/dx_k
        # dc_ij/dx_k is nonzero when k==i (gives dc_dx_i[i,j]) or k==j (gives dc_dx_j[i,j])
        #
        # = sum_i w[i,k] * dc_dx_j[i,k]  (j=k term)
        #   + sum_i sum_{j != k} w[i,j] * [dc_ij/dx_k if i==k]
        # = sum_i w[i,k] * dc_dx_j[i,k]
        #   + sum_{j != k} w[k,j] * dc_dx_i[k,j]
        #
        # The first sum includes i=k: w[k,k]*dc_dx_j[k,k] = w[k,k]*grad_rb_dx[k]
        # The second sum excludes j=k, so no double count of the diagonal.
        # But wait, the first sum also has the i=k term which is the diagonal.
        # And the second sum has i=k, j!=k terms.
        # So total = sum_i w[i,k]*dc_dx_j[i,k] + sum_{j!=k} w[k,j]*dc_dx_i[k,j]
        #          = (column k of w*dc_dx_j summed over i) + (row k of w*dc_dx_i excluding j=k)
        #
        # Let's compute:
        # term1[k] = sum_i w[i,k] * dc_dx_j[i,k]  = sum of column k of (w * dc_dx_j)
        # term2[k] = sum_{j!=k} w[k,j] * dc_dx_i[k,j] = sum of row k of (w * dc_dx_i) minus w[k,k]*dc_dx_i[k,k]

        w_dc_dx_j = w * dc_dx_j  # (N, N)
        w_dc_dx_i = w * dc_dx_i  # (N, N)
        w_dc_dy_j = w * dc_dy_j
        w_dc_dy_i = w * dc_dy_i

        term1_dx = np.sum(w_dc_dx_j, axis=0)  # (N,)
        term2_dx = np.sum(w_dc_dx_i, axis=1) - np.diag(w) * np.diag(dc_dx_i)  # (N,)
        dr_dx = term1_dx + term2_dx

        term1_dy = np.sum(w_dc_dy_j, axis=0)
        term2_dy = np.sum(w_dc_dy_i, axis=1) - np.diag(w) * np.diag(dc_dy_i)
        dr_dy = term1_dy + term2_dy

        # Objective is -sum(r), gradient is -dr/dparams
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

    def hex_init(seed):
        """Hexagonal lattice initialization."""
        rng = np.random.RandomState(seed)
        # 5x5 grid = 25, add 1 more
        # Use a hexagonal pattern
        x = np.zeros(N)
        y = np.zeros(N)
        idx = 0
        for row in range(6):
            for col in range(5):
                if idx >= N:
                    break
                x[idx] = (col + 0.5) / 5.0 + (0.5 / 5.0) * (row % 2)
                y[idx] = (row + 0.5) / 6.0
                idx += 1
                if idx >= N:
                    break
        x += rng.uniform(-0.03, 0.03, N)
        y += rng.uniform(-0.03, 0.03, N)
        x = np.clip(x, 0.02, 0.98)
        y = np.clip(y, 0.02, 0.98)
        return x, y

    def grid_init(seed):
        """Grid initialization with jitter."""
        rng = np.random.RandomState(seed)
        side = 5  # 5x5 = 25, add 1
        x = np.zeros(N)
        y = np.zeros(N)
        idx = 0
        for i in range(6):
            for j in range(5):
                if idx >= N:
                    break
                x[idx] = (j + 0.5) / 5.0
                y[idx] = (i + 0.5) / 6.0
                idx += 1
                if idx >= N:
                    break
        x += rng.uniform(-0.04, 0.04, N)
        y += rng.uniform(-0.04, 0.04, N)
        x = np.clip(x, 0.02, 0.98)
        y = np.clip(y, 0.02, 0.98)
        return x, y

    def random_init(seed):
        """Random initialization."""
        rng = np.random.RandomState(seed)
        x = rng.uniform(0.1, 0.9, N)
        y = rng.uniform(0.1, 0.9, N)
        return x, y

    def gradient_polish(x0, y0, beta_start=10.0, beta_end=1000.0, num_steps=6):
        """Iterative gradient polish with increasing beta."""
        params0 = np.concatenate([x0, y0])
        bounds = [(0.005, 0.995)] * (2 * N)

        betas = np.geomspace(beta_start, beta_end, num_steps)
        current_params = params0.copy()
        best_val = -np.inf
        best_params = params0.copy()

        def get_val(params):
            x = params[:N]
            y = params[N:2*N]
            r = compute_radii(x, y)
            if check_validity(x, y, r):
                return np.sum(r)
            return -np.inf

        init_val = get_val(current_params)
        if init_val > best_val:
            best_val = init_val
            best_params = current_params.copy()

        for beta in betas:
            if time.time() - start_time > DEADLINE - 3:
                break
            try:
                res = minimize(soft_objective_and_grad, current_params,
                               method='L-BFGS-B', bounds=bounds,
                               args=(beta,),
                               jac=True,
                               options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50})
                current_params = res.x
                val = get_val(current_params)
                if val > best_val:
                    best_val = val
                    best_params = current_params.copy()
            except Exception:
                pass

        return best_params[:N], best_params[N:], best_val

    def best_fit_local_search(x, y, max_rounds=80):
        """Discrete best-fit local search."""
        step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005]
        dirs = np.array([
            [1, 0], [-1, 0], [0, 1], [0, -1],
            [1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)],
            [-1/np.sqrt(2), 1/np.sqrt(2)], [-1/np.sqrt(2), -1/np.sqrt(2)]
        ])

        x = x.copy()
        y = y.copy()
        r = compute_radii(x, y)
        curr_val = np.sum(r)

        for rnd in range(max_rounds):
            if time.time() - start_time > DEADLINE - 5:
                break

            best_improvement = 1e-12
            best_move = None

            for step in step_sizes:
                for i in range(N):
                    for d in range(8):
                        nx = x[i] + dirs[d, 0] * step
                        ny = y[i] + dirs[d, 1] * step
                        if nx < 0.01 or nx > 0.99 or ny < 0.01 or ny > 0.99:
                            continue
                        x_temp = x.copy()
                        y_temp = y.copy()
                        x_temp[i] = nx
                        y_temp[i] = ny
                        r_new = compute_radii(x_temp, y_temp)
                        new_val = np.sum(r_new)
                        improvement = new_val - curr_val
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_move = (i, d, step)

                if best_improvement > step * 0.05:
                    break

            if best_move is not None:
                i, d, step = best_move
                x[i] += dirs[d, 0] * step
                y[i] += dirs[d, 1] * step
                r = compute_radii(x, y)
                curr_val = np.sum(r)
            else:
                break

        return x, y, compute_radii(x, y), np.sum(compute_radii(x, y))

    # Multi-start optimization
    best_val = -np.inf
    best_x = None
    best_y = None
    best_r = None

    inits = [
        (hex_init, 0), (grid_init, 1), (random_init, 2),
        (hex_init, 10), (grid_init, 11), (random_init, 12),
        (hex_init, 20), (grid_init, 21), (random_init, 22),
        (hex_init, 30), (grid_init, 31), (random_init, 32),
    ]

    for init_func, seed in inits:
        if time.time() - start_time > DEADLINE - 15:
            break

        x0, y0 = init_func(seed)
        r0 = compute_radii(x0, y0)
        val0 = np.sum(r0)

        # Gradient polish
        if time.time() - start_time < DEADLINE - 10:
            xp, yp, vp = gradient_polish(x0, y0)
            if vp > val0:
                x0, y0, val0 = xp, yp, vp
                r0 = compute_radii(x0, y0)

        # Best-fit local search
        if time.time() - start_time < DEADLINE - 8:
            xb, yb, rb, vb = best_fit_local_search(x0, y0)
            if vb > val0:
                x0, y0, r0, val0 = xb, yb, rb, vb

        # Another gradient polish pass
        if time.time() - start_time < DEADLINE - 6:
            xp, yp, vp = gradient_polish(x0, y0, beta_start=50.0, beta_end=2000.0, num_steps=5)
            if vp > val0:
                x0, y0, val0 = xp, yp, vp
                r0 = compute_radii(x0, y0)

        if val0 > best_val:
            best_val = val0
            best_x = x0.copy()
            best_y = y0.copy()
            best_r = r0.copy()

    # Final high-precision polish
    if best_x is not None and time.time() - start_time < DEADLINE - 3:
        # Very high beta polish
        xp, yp, vp = gradient_polish(best_x, best_y, beta_start=200.0, beta_end=5000.0, num_steps=5)
        if vp > best_val:
            best_val = vp
            best_x = xp
            best_y = yp
            best_r = compute_radii(best_x, best_y)

        # Final best-fit
        xb, yb, rb, vb = best_fit_local_search(best_x, best_y, max_rounds=100)
        if vb > best_val:
            best_val = vb
            best_x = xb
            best_y = yb
            best_r = rb

    # Ensure validity
    if best_x is None:
        x, y = hex_init(0)
        best_r = compute_radii(x, y)
        best_x, best_y = x, y
    else:
        if not check_validity(best_x, best_y, best_r):
            for scale in np.linspace(0.999, 0.95, 20):
                r_test = best_r * scale
                if check_validity(best_x, best_y, r_test):
                    best_r = r_test
                    break
            else:
                best_r = compute_radii(best_x, best_y)

    return np.column_stack((best_x, best_y, best_r))
