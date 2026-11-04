"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    def _check_validity(w, h, x, y, r, tol=1e-7):
        """Check if solution is valid. Returns (is_valid, value)."""
        if np.any(r < 1e-8) or w <= 0 or h <= 0:
            return False, 0.0
        if w + h > 4.0 + tol:
            return False, 0.0
        if np.any(x < r - tol) or np.any(x > w - r + tol) or \
           np.any(y < r - tol) or np.any(y > h - r + tol):
            return False, 0.0
        diff_x = x[:, np.newaxis] - x[np.newaxis, :]
        diff_y = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(diff_x**2 + diff_y**2)
        np.fill_diagonal(dist, np.inf)
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        if np.any(r_sum > dist + tol):
            return False, 0.0
        return True, float(np.sum(r))

    def _hex_init(w, h, n=N):
        """Generate hexagonal initial layout. Returns (x, y, r)."""
        # Use rows pattern that sums to n
        if n == 21:
            rows = [4, 5, 4, 5, 3]
        else:
            # Fallback: square-ish grid
            side = int(np.ceil(np.sqrt(n)))
            rows = []
            remaining = n
            for i in range(side + 1):
                c = min(remaining, side)
                rows.append(c)
                remaining -= c
                if remaining == 0:
                    break

        max_row = max(rows)
        # Estimate radius from fitting hex pattern
        hex_w = max_row * 2.0
        hex_h = len(rows) * np.sqrt(3)
        r_est = min(w / (hex_w + 0.1), h / (hex_h + 0.1)) * 0.9
        r_est = max(r_est, 0.01)

        x_list, y_list, r_list = [], [], []
        y_offset = (h - hex_h * r_est) / 2.0
        for row_i, count in enumerate(rows):
            x_offset = (w - count * 2.0 * r_est) / 2.0
            for col in range(count):
                x = x_offset + col * 2.0 * r_est
                y = y_offset + row_i * np.sqrt(3) * r_est
                x_list.append(x)
                y_list.append(y)
                r_list.append(r_est)

        # Pad or trim to exactly n
        while len(x_list) < n:
            x_list.append(w / 2.0)
            y_list.append(h / 2.0)
            r_list.append(r_est * 0.5)
        x_list = x_list[:n]
        y_list = y_list[:n]
        r_list = r_list[:n]

        return np.array(x_list), np.array(y_list), np.array(r_list)

    def _grid_init(w, h, n=N):
        """Generate grid initial layout."""
        cols = int(np.ceil(np.sqrt(n * w / h)))
        rows = int(np.ceil(n / cols))
        while cols * rows < n:
            rows += 1
        x_list, y_list, r_list = [], [], []
        for i in range(rows):
            for j in range(cols):
                if len(x_list) >= n:
                    break
                x = (j + 0.5) * w / cols
                y = (i + 0.5) * h / rows
                r = min(w / cols, h / rows) * 0.45
                x_list.append(x)
                y_list.append(y)
                r_list.append(r)
        while len(x_list) < n:
            x_list.append(w / 2.0)
            y_list.append(h / 2.0)
            r_list.append(0.01)
        return np.array(x_list[:n]), np.array(y_list[:n]), np.array(r_list[:n])

    def _random_init(w, h, n=N, seed=42):
        """Generate random initial layout with non-overlapping circles."""
        rng = np.random.default_rng(seed)
        x_list, y_list, r_list = [], [], []
        r_max = min(w, h) / 2.0 * 0.4
        for i in range(n):
            for attempt in range(200):
                r = rng.uniform(0.02, r_max)
                x = rng.uniform(r, w - r)
                y = rng.uniform(r, h - r)
                # Check overlap with existing
                overlap = False
                for j in range(len(x_list)):
                    dx = x - x_list[j]
                    dy = y - y_list[j]
                    if dx*dx + dy*dy < (r + r_list[j])**2:
                        overlap = True
                        break
                if not overlap:
                    x_list.append(x)
                    y_list.append(y)
                    r_list.append(r)
                    break
            else:
                # Place at center with tiny radius
                x_list.append(w / 2.0)
                y_list.append(h / 2.0)
                r_list.append(0.01)
        return np.array(x_list), np.array(y_list), np.array(r_list)

    def _smooth_max_penalty(violations, beta=50.0):
        """Compute smooth max penalty: log-sum-exp approximation."""
        max_v = np.max(violations)
        if max_v < 1e-12:
            return 0.0
        scaled = violations * beta
        scaled -= np.max(scaled)  # for numerical stability
        smax = np.max(violations) + np.log(np.sum(np.exp(scaled))) / beta
        return smax

    def _objective_and_grad(v, w_fixed=None, beta=50.0, weight=100.0):
        """
        Objective: maximize sum(r) subject to constraints.
        v = [w, x1..xn, y1..yn, r1..rn]
        Returns (f, grad).
        """
        n = N
        w = v[0]
        h = 2.0 - w
        x = v[1:1+n]
        y = v[1+n:1+2*n]
        r = v[1+2*n:1+3*n]

        # Ensure positive values for computation
        r_pos = np.maximum(r, 1e-8)
        x_clamped = np.clip(x, 0, w)
        y_clamped = np.clip(y, 0, h)

        # Boundary violations
        bx_low = r_pos - x_clamped
        bx_high = x_clamped + r_pos - w
        by_low = r_pos - y_clamped
        by_high = y_clamped + r_pos - h
        bviol = np.concatenate([bx_low, bx_high, by_low, by_high])
        bviol_pos = np.maximum(bviol, 0.0)

        # Circle-circle violations
        diff_x = x_clamped[:, np.newaxis] - x_clamped[np.newaxis, :]
        diff_y = y_clamped[:, np.newaxis] - y_clamped[np.newaxis, :]
        dist_sq = diff_x**2 + diff_y**2
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(np.maximum(dist_sq, 1e-16))
        r_sum = r_pos[:, np.newaxis] + r_pos[np.newaxis, :]
        cviol = r_sum - dist
        np.fill_diagonal(cviol, -np.inf)
        cviol_pos = np.maximum(cviol, 0.0)
        # Only upper triangle to avoid double counting
        tri_i, tri_j = np.triu_indices(n, k=1)
        cviol_upper = cviol_pos[tri_i, tri_j]

        all_viol = np.concatenate([bviol_pos, cviol_upper])
        penalty = _smooth_max_penalty(all_viol, beta)

        sum_r = np.sum(r_pos)
        f = -sum_r + weight * penalty

        # Compute gradient
        grad = np.zeros_like(v)

        # Gradient of smooth max penalty w.r.t. violations
        if np.max(all_viol) > 1e-12:
            scaled = all_viol * beta
            scaled -= np.max(scaled)
            exp_vals = np.exp(scaled)
            weights_sm = exp_vals / np.sum(exp_vals)
        else:
            weights_sm = np.zeros_like(all_viol)

        # d(bviol)/d(x), d(bviol)/d(y), d(bviol)/d(r)
        # bx_low = r - x => d/dx = -1, d/dr = +1
        # bx_high = x + r - w => d/dx = +1, d/dr = +1
        # by_low = r - y => d/dy = -1, d/dr = +1
        # by_high = y + r - h => d/dy = +1, d/dr = +1

        wb = weights_sm[:4*n]  # boundary violation weights
        # bx_low: indices 0..n-1
        # bx_high: indices n..2n-1
        # by_low: indices 2n..3n-1
        # by_high: indices 3n..4n-1

        # d(penalty)/d(x_i) from boundary
        dx_b = -wb[0:n] + wb[n:2*n]
        # d(penalty)/d(y_i) from boundary
        dy_b = -wb[2*n:3*n] + wb[3*n:4*n]
        # d(penalty)/d(r_i) from boundary
        dr_b = wb[0:n] + wb[n:2*n] + wb[2*n:3*n] + wb[3*n:4*n]

        # d(cviol)/d(x_i), d(y_i), d(r_i) for circle-circle
        # cviol_ij = r_i + r_j - dist_ij
        # d/dx_i = (x_i - x_j)/dist_ij, d/dx_j = (x_j - x_i)/dist_ij
        # d/dr_i = 1, d/dr_j = 1

        wc = weights_sm[4*n:]  # circle-circle violation weights (upper triangle)
        dx_c = np.zeros(n)
        dy_c = np.zeros(n)
        dr_c = np.zeros(n)

        for k in range(len(tri_i)):
            i, j = tri_i[k], tri_j[k]
            w_k = wc[k]
            if w_k < 1e-14:
                continue
            dx = x_clamped[i] - x_clamped[j]
            dy = y_clamped[i] - y_clamped[j]
            d = dist[i, j]
            if d < 1e-12:
                d = 1e-12
            inv_d = 1.0 / d
            dx_c[i] += w_k * dx * inv_d
            dx_c[j] -= w_k * dx * inv_d
            dy_c[i] += w_k * dy * inv_d
            dy_c[j] -= w_k * dy * inv_d
            dr_c[i] += w_k
            dr_c[j] += w_k

        # Total gradient of penalty
        grad_pen = weight * np.concatenate([dx_b + dx_c, dy_b + dy_c, dr_b + dr_c])

        # Gradient of -sum_r
        grad[-1-2*n:] = -1.0  # d(-sum_r)/d(r_i) = -1
        grad[1:1+n] += grad_pen[:n]
        grad[1+n:1+2*n] += grad_pen[n:2*n]
        grad[1+2*n:] += grad_pen[2*n:]

        # Gradient w.r.t. w (h = 2 - w)
        # bx_high depends on w: d(bx_high)/dw = -1
        # by_high depends on h: d(by_high)/dh = -1, dh/dw = -1 => d(by_high)/dw = +1
        grad_w = -np.sum(wb[n:2*n]) + np.sum(wb[3*n:4*n])
        grad[0] += weight * grad_w

        return f, grad

    def _local_optimize(x0, w_fixed=None, beta=50.0, weight=100.0, maxiter=500, tol=1e-10):
        """Run L-BFGS-B optimization."""
        n = N
        bounds = [(0.05, 1.95)]  # w bounds
        for i in range(n):
            bounds.append((None, None))  # x free
        for i in range(n):
            bounds.append((None, None))  # y free
        for i in range(n):
            bounds.append((1e-6, None))  # r positive

        def obj_func(v):
            f, g = _objective_and_grad(v, w_fixed, beta, weight)
            return f

        def grad_func(v):
            f, g = _objective_and_grad(v, w_fixed, beta, weight)
            return g

        try:
            res = minimize(obj_func, x0, jac=grad_func, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': maxiter, 'ftol': tol, 'gtol': 1e-8})
            return res.x
        except Exception:
            return x0

    def _polish(v, beta=200.0, weight=500.0, maxiter=300):
        """High-precision polish with tighter tolerances."""
        return _local_optimize(v, beta=beta, weight=weight, maxiter=maxiter, tol=1e-12)

    def _separate_circles(w, h, x, y, r, max_iters=100):
        """Separate overlapping circles using iterative projection."""
        for it in range(max_iters):
            moved = False
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    d = np.sqrt(dx*dx + dy*dy)
                    min_d = r[i] + r[j]
                    if d < min_d - 1e-10:
                        if d < 1e-12:
                            dx, dy = 1e-6, 0
                            d = 1e-6
                        push = (min_d - d) / (2.0 * d)
                        x[i] -= push * dx
                        y[i] -= push * dy
                        x[j] += push * dx
                        y[j] += push * dy
                        moved = True
            # Clamp to bounds
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)
            if not moved:
                break
        return x, y

    best_value = 0.0
    best_solution = None

    def _update_best(v):
        nonlocal best_value, best_solution
        w = v[0]
        h = 2.0 - w
        x = v[1:1+N]
        y = v[1+N:1+2*N]
        r = v[1+2*N:]
        valid, val = _check_validity(w, h, x, y, r)
        if valid and val > best_value:
            best_value = val
            best_solution = np.column_stack((x, y, r))

    # Phase 1: Try multiple aspect ratios with hexagonal initialization
    aspect_ratios = np.linspace(0.6, 1.4, 13)
    inits = ['hex', 'grid', 'random']

    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)

        for init_type in inits:
            if time.time() > deadline:
                break
            if init_type == 'hex':
                x, y, r = _hex_init(w, h)
            elif init_type == 'grid':
                x, y, r = _grid_init(w, h)
            else:
                x, y, r = _random_init(w, h, seed=int(ar*1000))

            # Build initial vector
            v0 = np.concatenate(([w], x, y, r))

            # Coarse optimization
            v1 = _local_optimize(v0, beta=30.0, weight=50.0, maxiter=300)
            _update_best(v1)

            # Fine optimization
            if time.time() > deadline:
                break
            v2 = _local_optimize(v1, beta=80.0, weight=200.0, maxiter=500)
            _update_best(v2)

    # Phase 2: Polish the best solution found
    if best_solution is not None and time.time() < deadline - 5.0:
        # Reconstruct v from best_solution
        x, y, r = best_solution[:, 0], best_solution[:, 1], best_solution[:, 2]
        # Find w, h from the solution
        # We need to recover w and h; use the aspect that was used
        # Actually, let's just try polishing with the current w from the solution
        # The solution doesn't store w,h directly, so let's try a few aspect ratios
        for ar in [0.8, 1.0, 1.2]:
            if time.time() > deadline:
                break
            w = 2.0 * ar / (1.0 + ar)
            h = 2.0 / (1.0 + ar)
            v0 = np.concatenate(([w], x, y, r))
            v_polished = _polish(v0, beta=200.0, weight=500.0, maxiter=200)
            _update_best(v_polished)

    # Phase 3: If still not good, try a broader search with different seeds
    if best_value < 0.5 and time.time() < deadline - 10.0:
        for seed in range(5):
            if time.time() > deadline:
                break
            w = 1.0 + 0.1 * seed
            h = 2.0 - w
            x, y, r = _random_init(w, h, seed=100 + seed)
            v0 = np.concatenate(([w], x, y, r))
            v1 = _local_optimize(v0, beta=30.0, weight=50.0, maxiter=400)
            _update_best(v1)
            v2 = _local_optimize(v1, beta=100.0, weight=300.0, maxiter=400)
            _update_best(v2)

    # Fallback
    if best_solution is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_solution

# EVOLVE-BLOCK-END
