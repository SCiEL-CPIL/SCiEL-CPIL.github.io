import numpy as np
import time
from scipy.optimize import minimize

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def is_valid(x, y, r, w, tol=1e-8):
        h = 2.0 - w
        if w <= tol or h <= tol:
            return False
        for i in range(N):
            if r[i] <= tol:
                return False
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                    return False
        return True

    def is_valid_equal_r(x, y, r, w, tol=1e-8):
        h = 2.0 - w
        if w <= tol or h <= tol or r <= tol:
            return False
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        iu = np.triu_indices(N, k=1)
        if np.any(d_sq[iu] < (2 * r) ** 2 - tol):
            return False
        return True

    def sum_radii(r):
        return float(np.sum(r))

    def equal_r_objective(z, r, w):
        x = z[:N]
        y = z[N:2*N]
        h = 2.0 - w
        pen = 0.0
        x_low = np.maximum(r - x, 0.0)
        x_high = np.maximum(x - (w - r), 0.0)
        y_low = np.maximum(r - y, 0.0)
        y_high = np.maximum(y - (h - r), 0.0)
        pen += np.sum(x_low ** 2) + np.sum(x_high ** 2)
        pen += np.sum(y_low ** 2) + np.sum(y_high ** 2)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        iu = np.triu_indices(N, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-15))
        overlap = np.maximum(2 * r - d_pairs, 0.0)
        pen += np.sum(overlap ** 2)
        return pen

    def clearance_maximization_objective(z, r, w, beta=80.0):
        x = z[:N]
        y = z[N:2*N]
        h_ = 2.0 - w
        clearances = []
        for i in range(N):
            clearances.append(x[i] - r[i])
            clearances.append(w - x[i] - r[i])
            clearances.append(y[i] - r[i])
            clearances.append(h_ - y[i] - r[i])
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                clearances.append(d - r[i] - r[j])
        c = np.array(clearances)
        max_c = np.max(c)
        lse = max_c + np.log(np.sum(np.exp(c - max_c))) / beta
        neg_c = np.maximum(-c, 0.0)
        pen = 1000.0 * np.sum(neg_c ** 2)
        return -lse + pen

    def newton_contact_solve(x, y, r, w, max_iter=300, tol=1e-13, damp=0.5):
        """
        Newton solver for the contact system with equal radii:
          active circle-circle contacts: |p_i - p_j| = 2r
          active wall contacts: x_i=r, x_i=w-r, y_i=r, y_i=h-r
          plus: w + h = 2  (h = 2 - w)
        Variables: [x, y, r, w]  (2N+2 unknowns).
        Active set = contacts within a small band (touching or slightly overlapping).
        Returns (x, y, r, w, converged, max_residual).
        """
        h = 2.0 - w
        x = np.array(x, dtype=float).copy()
        y = np.array(y, dtype=float).copy()
        r = float(r)
        w = float(w)

        band = 0.005  # contact detection band (fraction of 2r scale)
        band_abs = band * max(r, 0.01)

        for it in range(max_iter):
            if time.time() > deadline - 6:
                break
            h = 2.0 - w

            # Build active contact list
            active = []  # (type, i, j, sign)
            for i in range(N):
                # wall x_low: x_i = r
                c = x[i] - r
                if abs(c) < band_abs:
                    active.append((0, i, -1, 1))  # x_i - r
                # wall x_high: x_i = w - r
                c = x[i] - (w - r)
                if abs(c) < band_abs:
                    active.append((1, i, -1, 1))  # x_i - w + r
                # wall y_low: y_i = r
                c = y[i] - r
                if abs(c) < band_abs:
                    active.append((2, i, -1, 1))  # y_i - r
                # wall y_high: y_i = h - r
                c = y[i] - (h - r)
                if abs(c) < band_abs:
                    active.append((3, i, -1, 1))  # y_i - h + r
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    c = d - 2.0 * r
                    if abs(c) < band_abs:
                        active.append((4, i, j, 1))  # d - 2r

            n_act = len(active)
            nvar = 2 * N + 2
            # We need a square system; if n_act + 2 (r,w) < nvar, use least-squares
            # Build J (n_act x nvar) and residual (n_act,)
            J = np.zeros((n_act, nvar))
            res = np.zeros(n_act)

            for k, (typ, i, j, sgn) in enumerate(active):
                if typ == 0:  # x_i - r
                    J[k, i] = 1.0
                    J[k, 2*N] = -1.0
                    res[k] = x[i] - r
                elif typ == 1:  # x_i - w + r
                    J[k, i] = 1.0
                    J[k, 2*N] = 1.0
                    J[k, 2*N+1] = -1.0
                    res[k] = x[i] - (w - r)
                elif typ == 2:  # y_i - r
                    J[k, N+i] = 1.0
                    J[k, 2*N] = -1.0
                    res[k] = y[i] - r
                elif typ == 3:  # y_i - h + r
                    J[k, N+i] = 1.0
                    J[k, 2*N] = 1.0
                    J[k, 2*N+1] = 1.0
                    res[k] = y[i] - (h - r)
                else:  # circle-circle: d - 2r
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx*dx + dy*dy) + 1e-15
                    ux = dx / d
                    uy = dy / d
                    J[k, i] = ux
                    J[k, N+i] = uy
                    J[k, j] = -ux
                    J[k, N+j] = -uy
                    J[k, 2*N] = -2.0
                    res[k] = d - 2.0 * r

            # Add perimeter constraint: w + h = 2  ->  w - (2-w) = 0 ... use w + (2-w) - 2 = 0
            # Actually h = 2 - w is enforced by construction; instead add a regularizer
            # to fix the under-determined system: we pin the rectangle via w.
            # Use least-squares on [J; w_row] with target [res; 0]
            J_aug = np.vstack([J, np.zeros((1, nvar))])
            r_aug = np.concatenate([res, [0.0]])
            # The last row is all zeros -> gives no info; instead regularize w
            # Add a weak constraint pulling w to current w to keep aspect fixed
            J_aug[0, 2*N+1] = 1e-3  # weak w regularization
            r_aug[0] = 1e-3 * (w - w)  # 0

            # Solve least squares
            try:
                delta, *_ = np.linalg.lstsq(J_aug, r_aug, rcond=None)
            except np.linalg.LinAlgError:
                break

            # Damped step
            step = delta
            step[2*N] = np.clip(step[2*N], -0.02, 0.02)  # limit r change
            step[2*N+1] = np.clip(step[2*N+1], -0.05, 0.05)  # limit w change
            lam = damp
            for _ in range(10):
                x_new = x + lam * step[:N]
                y_new = y + lam * step[N:2*N]
                r_new = r + lam * step[2*N]
                w_new = w + lam * step[2*N+1]
                h_new = 2.0 - w_new
                if r_new <= 0.005 or w_new <= 0.2 or h_new <= 0.2:
                    lam *= 0.5
                    continue
                # Check no circle went outside
                if np.any(x_new < r_new - 0.02) or np.any(x_new > w_new - r_new + 0.02):
                    lam *= 0.5
                    continue
                if np.any(y_new < r_new - 0.02) or np.any(y_new > h_new - r_new + 0.02):
                    lam *= 0.5
                    continue
                break
            if lam < 1e-6:
                break

            x = x + lam * step[:N]
            y = y + lam * step[N:2*N]
            r = r + lam * step[2*N]
            w = w + lam * step[2*N+1]

            # Compute max residual
            h = 2.0 - w
            max_res = 0.0
            for i in range(N):
                max_res = max(max_res, abs(x[i] - r), abs(x[i] - (w - r)))
                max_res = max(max_res, abs(y[i] - r), abs(y[i] - (h - r)))
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i]-x[j]; dy = y[i]-y[j]
                    d = np.sqrt(dx*dx+dy*dy)
                    max_res = max(max_res, abs(d - 2*r))
            if max_res < tol:
                break
            damp = min(1.0, damp * 1.1)

        h = 2.0 - w
        converged = True
        # Final residual check
        max_res = 0.0
        for i in range(N):
            max_res = max(max_res, abs(x[i] - r), abs(x[i] - (w - r)))
            max_res = max(max_res, abs(y[i] - r), abs(y[i] - (h - r)))
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i]-x[j]; dy = y[i]-y[j]
                d = np.sqrt(dx*dx+dy*dy)
                max_res = max(max_res, abs(d - 2*r))
        if max_res > 1e-6:
            converged = False
        return x, y, r, w, converged, max_res

    def newton_polish(x, y, r, w, max_iter=400, tol=1e-14):
        """Run Newton contact solve; return improved (x,y,r,w) if valid & better radius."""
        if time.time() > deadline - 5:
            return None
        x_n, y_n, r_n, w_n, conv, max_res = newton_contact_solve(x, y, r, w,
                                                                max_iter=max_iter, tol=tol)
        if r_n <= 0.005 or not is_valid_equal_r(x_n, y_n, r_n, w_n):
            return None
        if r_n <= r * (1 - 1e-9):
            return None
        return x_n, y_n, r_n, w_n

    def grid_init(w0, n=N):
        h0 = 2.0 - w0
        best_val = -1
        best_x = best_y = best_r = None
        for rows in range(3, 9):
            for cols in range(3, 9):
                if rows * cols < n:
                    continue
                if rows * cols > n + 8:
                    continue
                cell_w = w0 / cols
                cell_h = h0 / rows
                r_cell = min(cell_w, cell_h) / 2.0 * 0.98
                if r_cell < 0.015:
                    continue
                x = np.zeros(n)
                y = np.zeros(n)
                r = np.full(n, r_cell)
                idx = 0
                for ri in range(rows):
                    for ci in range(cols):
                        if idx >= n:
                            break
                        x[idx] = (ci + 0.5) * cell_w
                        y[idx] = (ri + 0.5) * cell_h
                        idx += 1
                if idx < n:
                    for i in range(idx, n):
                        x[i] = w0 / 2.0
                        y[i] = h0 / 2.0
                        r[i] = r_cell * 0.5
                val = sum_radii(r)
                if val > best_val:
                    best_val = val
                    best_x = x.copy()
                    best_y = y.copy()
                    best_r = r.copy()
        if best_x is None:
            cols = 5
            rows = 5
            cell_w = w0 / cols
            cell_h = h0 / rows
            r0 = min(cell_w, cell_h) / 2.0 * 0.9
            x = np.zeros(n)
            y = np.zeros(n)
            r = np.full(n, r0)
            for i in range(n):
                ri, ci = divmod(i, cols)
                x[i] = (ci + 0.5) * cell_w
                y[i] = (ri + 0.5) * cell_h
            best_x, best_y, best_r = x, y, r
        return best_x, best_y, best_r

    def hex_init(w0, n=N):
        h0 = 2.0 - w0
        area = w0 * h0
        r_est = np.sqrt(area / (n * np.pi / 0.9069))
        r_est = min(r_est, 0.15)
        x = np.zeros(n)
        y = np.zeros(n)
        r = np.full(n, r_est)
        dx = 2.0 * r_est * 1.002
        dy = np.sqrt(3) * r_est * 1.002
        row = 0
        col = 0
        idx = 0
        while idx < n:
            cx = dx * (col + 0.5 * (row % 2)) + r_est
            cy = dy * row + r_est
            if cx > w0 - r_est:
                row += 1
                col = 0
                cx = dx * (col + 0.5 * (row % 2)) + r_est
                cy = dy * row + r_est
            if cy > h0 - r_est:
                break
            x[idx] = cx
            y[idx] = cy
            idx += 1
            col += 1
        for i in range(idx, n):
            x[i] = w0 / 2.0
            y[i] = h0 / 2.0
            r[i] = r_est * 0.6
        return x, y, r

    def find_positions_for_radius(r, w, x_init, y_init, max_iter=200):
        h = 2.0 - w
        z0 = np.concatenate([x_init, y_init])
        bounds = [(r, w - r)] * N + [(r, h - r)] * N
        try:
            res = minimize(equal_r_objective, z0, args=(r, w), method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-14, 'maxls': 50})
            return res.x[:N], res.x[N:2*N], res.fun
        except Exception:
            return x_init.copy(), y_init.copy(), np.inf

    def find_positions_maximize_clearance(r, w, x_init, y_init, max_iter=200, beta=80.0):
        h = 2.0 - w
        z0 = np.concatenate([x_init, y_init])
        bounds = [(r, w - r)] * N + [(r, h - r)] * N
        try:
            res = minimize(clearance_maximization_objective, z0, args=(r, w, beta),
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': 1e-14, 'maxls': 50})
            return res.x[:N], res.x[N:2*N], res.fun
        except Exception:
            return x_init.copy(), y_init.copy(), np.inf

    def force_relax(x, y, r, w, max_iter=800, tol=1e-10):
        h = 2.0 - w
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        dt = 0.003
        min_dt = 0.00005
        for it in range(max_iter):
            fx = np.zeros(N)
            fy = np.zeros(N)
            for i in range(N):
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i]) * 200
                if x[i] > w - r[i]:
                    fx[i] -= (x[i] - (w - r[i])) * 200
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i]) * 200
                if y[i] > h - r[i]:
                    fy[i] -= (y[i] - (h - r[i])) * 200
            max_overlap = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy) + 1e-12
                    r_sum = r[i] + r[j]
                    overlap = r_sum - d
                    if overlap > 0:
                        ux = dx / d
                        uy = dy / d
                        f = overlap * 30.0
                        fx[i] += ux * f
                        fy[i] += uy * f
                        fx[j] -= ux * f
                        fy[j] -= uy * f
                        max_overlap = max(max_overlap, overlap)
            for i in range(N):
                if x[i] < r[i]:
                    max_overlap = max(max_overlap, r[i] - x[i])
                if x[i] > w - r[i]:
                    max_overlap = max(max_overlap, x[i] - (w - r[i]))
                if y[i] < r[i]:
                    max_overlap = max(max_overlap, r[i] - y[i])
                if y[i] > h - r[i]:
                    max_overlap = max(max_overlap, y[i] - (h - r[i]))
            if max_overlap < tol:
                break
            force_mag = np.max(np.sqrt(fx * fx + fy * fy))
            if force_mag > 1.0:
                dt = max(min_dt, dt * 0.5)
            elif force_mag < 0.05:
                dt = min(0.008, dt * 1.15)
            x += dt * fx
            y += dt * fy
            for i in range(N):
                x[i] = np.clip(x[i], r[i], w - r[i])
                y[i] = np.clip(y[i], r[i], h - r[i])
        return x, y

    def binary_search_max_radius(w, x_init, y_init, r_low, r_high, max_iter=12, tol=1e-6):
        best_x, best_y, best_r = x_init.copy(), y_init.copy(), r_low
        for _ in range(max_iter):
            if time.time() > deadline - 10:
                break
            r_mid = (r_low + r_high) / 2.0
            if r_mid - r_low < tol:
                break
            x_opt, y_opt, obj_val = find_positions_for_radius(r_mid, w, x_init, y_init, max_iter=150)
            if obj_val < 1e-8 and is_valid_equal_r(x_opt, y_opt, r_mid, w):
                best_x, best_y, best_r = x_opt.copy(), y_opt.copy(), r_mid
                r_low = r_mid
            else:
                r_high = r_mid
        return best_x, best_y, best_r

    def binary_search_max_radius_clearance(w, x_init, y_init, r_low, r_high, max_iter=12, tol=1e-6):
        best_x, best_y, best_r = x_init.copy(), y_init.copy(), r_low
        for _ in range(max_iter):
            if time.time() > deadline - 10:
                break
            r_mid = (r_low + r_high) / 2.0
            if r_mid - r_low < tol:
                break
            x_opt, y_opt, obj_val = find_positions_maximize_clearance(r_mid, w, x_init, y_init, max_iter=150, beta=80.0)
            if is_valid_equal_r(x_opt, y_opt, r_mid, w):
                best_x, best_y, best_r = x_opt.copy(), y_opt.copy(), r_mid
                r_low = r_mid
            else:
                r_high = r_mid
        return best_x, best_y, best_r

    def grow_equal_radius(x, y, r, w, max_rounds=10):
        r_cur = r[0]
        for _ in range(max_rounds):
            if time.time() > deadline - 8:
                break
            r_new = r_cur * 1.005
            x_opt, y_opt, obj_val = find_positions_for_radius(r_new, w, x, y, max_iter=100)
            if obj_val < 1e-8 and is_valid_equal_r(x_opt, y_opt, r_new, w):
                x, y, r_cur = x_opt, y_opt, r_new
            else:
                break
        r_arr = np.full(N, r_cur)
        return x, y, r_arr, w

    def grow_equal_radius_clearance(x, y, r, w, max_rounds=10):
        r_cur = r[0]
        for _ in range(max_rounds):
            if time.time() > deadline - 8:
                break
            r_new = r_cur * 1.005
            x_opt, y_opt, obj_val = find_positions_maximize_clearance(r_new, w, x, y, max_iter=100, beta=100.0)
            if is_valid_equal_r(x_opt, y_opt, r_new, w):
                x, y, r_cur = x_opt, y_opt, r_new
            else:
                break
        r_arr = np.full(N, r_cur)
        return x, y, r_arr, w

    def cooperative_grow(x, y, r, w, max_rounds=8):
        r_cur = r[0] if np.allclose(r, r[0]) else min(r)
        w_cur = w
        for _ in range(max_rounds):
            if time.time() > deadline - 8:
                break
            r_new = r_cur * 1.003
            x_opt, y_opt, obj_val = find_positions_for_radius(r_new, w_cur, x, y, max_iter=80)
            if obj_val < 1e-8 and is_valid_equal_r(x_opt, y_opt, r_new, w_cur):
                x, y, r_cur = x_opt, y_opt, r_new
                continue
            improved = False
            for dw in [-0.02, -0.01, 0.01, 0.02]:
                w_try = w_cur + dw
                h_try = 2.0 - w_try
                if w_try < 0.4 or h_try < 0.4:
                    continue
                x_opt, y_opt, obj_val = find_positions_for_radius(r_new, w_try, x, y, max_iter=60)
                if obj_val < 1e-8 and is_valid_equal_r(x_opt, y_opt, r_new, w_try):
                    x, y, w_cur, r_cur = x_opt, y_opt, w_try, r_new
                    improved = True
                    break
            if not improved:
                break
        r_arr = np.full(N, r_cur)
        return x, y, r_arr, w_cur

    def polish_clearance(x, y, r, w, beta=100.0):
        h = 2.0 - w
        z0 = np.concatenate([x, y])
        bounds = [(r[i], w - r[i]) for i in range(N)] + \
                 [(r[i], h - r[i]) for i in range(N)]
        try:
            res = minimize(clearance_maximization_objective, z0, args=(r, w, beta),
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 300, 'ftol': 1e-14, 'maxls': 60})
            x_t, y_t = res.x[:N], res.x[N:2*N]
            if is_valid(x_t, y_t, r, w):
                return x_t, y_t
        except Exception:
            pass
        return x, y

    def equal_r_slsqp(x_init, y_init, r_init, w_init, time_limit=15.0):
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            return None
        r_target = r_init[0]
        h_init = 2.0 - w_init

        def obj(z):
            x = z[:N]
            y = z[N:2*N]
            w = z[2*N]
            h = 2.0 - w
            if w < 0.3 or h < 0.3:
                return 1e8
            x_low = np.maximum(r_target - x, 0.0)
            x_high = np.maximum(x - (w - r_target), 0.0)
            y_low = np.maximum(r_target - y, 0.0)
            y_high = np.maximum(y - (h - r_target), 0.0)
            wall_pen = 500.0 * (np.sum(x_low ** 2) + np.sum(x_high ** 2) +
                               np.sum(y_low ** 2) + np.sum(y_high ** 2))
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d_sq = dx * dx + dy * dy
            iu = np.triu_indices(N, k=1)
            d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-15))
            overlap = np.maximum(2 * r_target - d_pairs, 0.0)
            overlap_pen = 500.0 * np.sum(overlap ** 2)
            space_reward = 0.1 * (w * (2.0 - w))
            return -space_reward + wall_pen + overlap_pen

        z0 = np.concatenate([x_init, y_init, [w_init]])
        bounds = [(r_target, w_init - r_target)] * N + \
                 [(r_target, h_init - r_target)] * N + [(0.3, 1.7)]
        try:
            maxiter = int(min(300, max(50, remaining * 20)))
            res = minimize(obj, z0, method='SLSQP', bounds=bounds,
                          options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 80})
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            w_opt = float(np.clip(res.x[2*N], 0.3, 1.7))
            if is_valid_equal_r(x_opt, y_opt, r_target, w_opt):
                r_arr = np.full(N, r_target)
                return x_opt, y_opt, r_arr, w_opt
            x_f, y_f = force_relax(x_opt, y_opt, np.full(N, r_target), w_opt, max_iter=100)
            if is_valid_equal_r(x_f, y_f, r_target, w_opt):
                r_arr = np.full(N, r_target)
                return x_f, y_f, r_arr, w_opt
        except Exception:
            pass
        return None

    def smooth_objective(z, penalty_weight=500.0):
        x = z[:N]
        y = z[N:2*N]
        r = z[2*N:3*N]
        w = z[3*N]
        h = 2.0 - w
        if w < 0.2 or h < 0.2:
            return 1e8
        r_clipped = np.clip(r, 0.001, min(w, h) / 2.0 - 0.001)
        obj = -np.sum(r_clipped)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r_clipped[:, None] + r_clipped[None, :]
        iu = np.triu_indices(N, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-15))
        overlap = np.maximum(r_sum[iu] - d_pairs, 0.0)
        overlap_penalty = penalty_weight * np.sum(overlap ** 2)
        x_low_viol = np.maximum(r_clipped - x, 0.0)
        x_high_viol = np.maximum(x - (w - r_clipped), 0.0)
        y_low_viol = np.maximum(r_clipped - y, 0.0)
        y_high_viol = np.maximum(y - (h - r_clipped), 0.0)
        wall_penalty = penalty_weight * (np.sum(x_low_viol ** 2) + np.sum(x_high_viol ** 2) +
                                         np.sum(y_low_viol ** 2) + np.sum(y_high_viol ** 2))
        small_r_penalty = penalty_weight * np.sum(np.maximum(0.01 - r_clipped, 0.0) ** 2)
        return obj + overlap_penalty + wall_penalty + small_r_penalty

    def smooth_slsqp(x_init, y_init, r_init, w_init, time_limit=18.0):
        z0 = np.concatenate([x_init, y_init, r_init, [w_init]])
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            return None
        maxiter = int(min(500, max(100, remaining * 25)))
        h_init = 2.0 - w_init
        bounds = []
        for i in range(N):
            bounds.append((r_init[i] * 0.3, w_init - r_init[i] * 0.3))
        for i in range(N):
            bounds.append((r_init[i] * 0.3, max(r_init[i] * 0.3, h_init - r_init[i] * 0.3)))
        for i in range(N):
            bounds.append((0.01, 0.45))
        bounds.append((0.35, 1.65))
        try:
            res = minimize(smooth_objective, z0, method='SLSQP', bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            w_opt = res.x[3*N]
            if np.any(r_opt < 0.005):
                return None
            if is_valid(x_opt, y_opt, r_opt, w_opt):
                return x_opt, y_opt, r_opt, w_opt
            x_f, y_f = force_relax(x_opt, y_opt, r_opt, w_opt, max_iter=200)
            if is_valid(x_f, y_f, r_opt, w_opt):
                return x_f, y_f, r_opt, w_opt
        except Exception:
            pass
        return None

    def try_grow(x, y, r, w, max_rounds=5):
        h = 2.0 - w
        for round_idx in range(max_rounds):
            if (time.time() - start_time) > deadline - 8:
                break
            improved = False
            for i in range(N):
                if (time.time() - start_time) > deadline - 8:
                    break
                for _ in range(50):
                    r_new = r.copy()
                    r_new[i] *= 1.003
                    # relax positions via L-BFGS-B
                    def obj(z):
                        xz = z[:N]
                        yz = z[N:2*N]
                        pen = 0.0
                        for k in range(N):
                            if xz[k] < r_new[k]:
                                pen += (r_new[k] - xz[k]) ** 2
                            if xz[k] > w - r_new[k]:
                                pen += (xz[k] - (w - r_new[k])) ** 2
                            if yz[k] < r_new[k]:
                                pen += (r_new[k] - yz[k]) ** 2
                            if yz[k] > h - r_new[k]:
                                pen += (yz[k] - (h - r_new[k])) ** 2
                        for a in range(N):
                            for b in range(a + 1, N):
                                ddx = xz[a] - xz[b]
                                ddy = yz[a] - yz[b]
                                d = np.sqrt(ddx * ddx + ddy * ddy) + 1e-12
                                ov = r_new[a] + r_new[b] - d
                                if ov > 0:
                                    pen += ov * ov
                        return pen
                    z0 = np.concatenate([x, y])
                    bounds = [(r_new[k], w - r_new[k]) for k in range(N)] + \
                             [(r_new[k], h - r_new[k]) for k in range(N)]
                    try:
                        res = minimize(obj, z0, method='L-BFGS-B', bounds=bounds,
                                       options={'maxiter': 40, 'ftol': 1e-12, 'maxls': 50})
                        x_t, y_t = res.x[:N], res.x[N:2*N]
                        if is_valid(x_t, y_t, r_new, w):
                            x, y, r = x_t, y_t, r_new
                            improved = True
                        else:
                            break
                    except Exception:
                        break
            if not improved:
                break
        return x, y, r, w

    def polish(x, y, r, w):
        h = 2.0 - w

        def obj(z):
            xz = z[:N]
            yz = z[N:2*N]
            pen = 0.0
            for i in range(N):
                if xz[i] < r[i]:
                    pen += (r[i] - xz[i]) ** 2
                if xz[i] > w - r[i]:
                    pen += (xz[i] - (w - r[i])) ** 2
                if yz[i] < r[i]:
                    pen += (r[i] - yz[i]) ** 2
                if yz[i] > h - r[i]:
                    pen += (yz[i] - (h - r[i])) ** 2
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xz[i] - xz[j]
                    dy = yz[i] - yz[j]
                    d = np.sqrt(dx * dx + dy * dy) + 1e-12
                    ov = r[i] + r[j] - d
                    if ov > 0:
                        pen += ov * ov
            return pen

        z0 = np.concatenate([x, y])
        try:
            res = minimize(obj, z0, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14})
            x_t, y_t = res.x[:N], res.x[N:2*N]
            if is_valid(x_t, y_t, r, w):
                return x_t, y_t, r, w
        except Exception:
            pass
        return x, y, r, w

    def logsumexp_objective(z, r_fixed, w, weight=100.0):
        x = z[:N]
        y = z[N:2*N]
        h = 2.0 - w
        x_low = np.maximum(r_fixed - x, 0.0)
        x_high = np.maximum(x - (w - r_fixed), 0.0)
        y_low = np.maximum(r_fixed - y, 0.0)
        y_high = np.maximum(y - (h - r_fixed), 0.0)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        iu = np.triu_indices(N, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-15))
        overlap = np.maximum(2 * r_fixed - d_pairs, 0.0)
        all_viol = np.concatenate([x_low**2, x_high**2, y_low**2, y_high**2, overlap**2])
        max_v = np.max(all_viol)
        if max_v < 1e-14:
            return 0.0
        lse = max_v + np.log(np.sum(np.exp(all_viol - max_v)))
        return weight * lse

    def logsumexp_optimize(x_init, y_init, r, w, time_limit=12.0):
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            return None
        h = 2.0 - w
        z0 = np.concatenate([x_init, y_init])
        bounds = [(r, w - r)] * N + [(r, h - r)] * N
        try:
            maxiter = int(min(400, max(80, remaining * 30)))
            res = minimize(logsumexp_objective, z0, args=(r, w, 100.0), method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 60})
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            if is_valid_equal_r(x_opt, y_opt, r, w):
                return x_opt, y_opt, np.full(N, r), w
        except Exception:
            pass
        return None

    def multi_start_perturb(x, y, r, w, rng, n_perturb=3, scale=0.01):
        best_val = sum_radii(r)
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        for _ in range(n_perturb):
            if time.time() > deadline - 10:
                break
            x_p = x + rng.normal(0, scale, N)
            y_p = y + rng.normal(0, scale, N)
            x_p = np.clip(x_p, r, w - r)
            h = 2.0 - w
            y_p = np.clip(y_p, r, h - r)
            x_t, y_t = force_relax(x_p, y_p, r, w, max_iter=300)
            if is_valid(x_t, y_t, r, w):
                val = sum_radii(r)
                if val > best_val:
                    best_val = val
                    best_x, best_y, best_r = x_t.copy(), y_t.copy(), r.copy()
        return best_x, best_y, best_r

    best_value = 0.0
    best_result = None
    rng = np.random.default_rng(42)

    # Phase 1: multiple aspect ratios with grid + hex init + binary search
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.7, 0.55, 2.0, 0.5, 1.1, 0.9, 1.25, 0.8, 1.4, 0.7, 1.6, 0.65]
    for w0 in aspect_ratios:
        if time.time() > deadline - 50:
            break
        h0 = 2.0 - w0
        if h0 < 0.35:
            continue
        x0, y0, r0 = grid_init(w0)
        x_t, y_t = force_relax(x0, y0, r0, w0, max_iter=500)
        if is_valid(x_t, y_t, r0, w0):
            val = sum_radii(r0)
            if val > best_value:
                best_value = val
                best_result = (x_t.copy(), y_t.copy(), r0.copy(), w0)
        x0h, y0h, r0h = hex_init(w0)
        x_th, y_th = force_relax(x0h, y0h, r0h, w0, max_iter=500)
        if is_valid(x_th, y_th, r0h, w0):
            val_h = sum_radii(r0h)
            if val_h > best_value:
                best_value = val_h
                best_result = (x_th.copy(), y_th.copy(), r0h.copy(), w0)
        if time.time() < deadline - 40:
            r_init_val = r0[0] if np.allclose(r0, r0[0]) else min(r0)
            x_eq, y_eq, r_eq = binary_search_max_radius(w0, x_t, y_t, r_init_val * 0.9, r_init_val * 1.15, max_iter=10)
            if r_eq > 0.01:
                r_arr = np.full(N, r_eq)
                val_eq = sum_radii(r_arr)
                if val_eq > best_value:
                    best_value = val_eq
                    best_result = (x_eq.copy(), y_eq.copy(), r_arr.copy(), w0)
            if time.time() < deadline - 35:
                x_eq2, y_eq2, r_eq2 = binary_search_max_radius_clearance(w0, x_t, y_t, r_init_val * 0.95, r_init_val * 1.12, max_iter=8)
                if r_eq2 > 0.01:
                    r_arr2 = np.full(N, r_eq2)
                    val_eq2 = sum_radii(r_arr2)
                    if val_eq2 > best_value:
                        best_value = val_eq2
                        best_result = (x_eq2.copy(), y_eq2.copy(), r_arr2.copy(), w0)

    # Phase 2: SLSQP + clearance polish + log-sum-exp
    if best_result is not None and time.time() < deadline - 35:
        x_cur, y_cur, r_cur, w_cur = best_result
        result_eq = equal_r_slsqp(x_cur, y_cur, r_cur, w_cur, time_limit=18.0)
        if result_eq is not None:
            x_s, y_s, r_s, w_s = result_eq
            val_s = sum_radii(r_s)
            if val_s > best_value:
                best_value = val_s
                best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)
        if time.time() < deadline - 25:
            x_cur, y_cur, r_cur, w_cur = best_result
            r_eq = r_cur[0] if np.allclose(r_cur, r_cur[0]) else min(r_cur)
            x_cp, y_cp = polish_clearance(x_cur, y_cur, np.full(N, r_eq), w_cur, beta=100.0)
            if is_valid_equal_r(x_cp, y_cp, r_eq, w_cur):
                r_arr_cp = np.full(N, r_eq)
                val_cp = sum_radii(r_arr_cp)
                if val_cp > best_value:
                    best_value = val_cp
                    best_result = (x_cp.copy(), y_cp.copy(), r_arr_cp.copy(), w_cur)
        if time.time() < deadline - 20:
            x_cur, y_cur, r_cur, w_cur = best_result
            r_eq = r_cur[0] if np.allclose(r_cur, r_cur[0]) else min(r_cur)
            result_lse = logsumexp_optimize(x_cur, y_cur, r_eq, w_cur, time_limit=10.0)
            if result_lse is not None:
                x_s, y_s, r_s, w_s = result_lse
                val_s = sum_radii(r_s)
                if val_s > best_value:
                    best_value = val_s
                    best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)
        if time.time() < deadline - 15:
            x_cur, y_cur, r_cur, w_cur = best_result
            result = smooth_slsqp(x_cur, y_cur, r_cur, w_cur, time_limit=12.0)
            if result is not None:
                x_s, y_s, r_s, w_s = result
                val_s = sum_radii(r_s)
                if val_s > best_value:
                    best_value = val_s
                    best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)

    # Phase 3: grow radii
    if best_result is not None and time.time() < deadline - 25:
        x_cur, y_cur, r_cur, w_cur = best_result
        if np.std(r_cur) < 0.005:
            x_eq, y_eq, r_eq_arr, w_eq = grow_equal_radius(x_cur, y_cur, r_cur, w_cur, max_rounds=8)
            if is_valid_equal_r(x_eq, y_eq, r_eq_arr[0], w_eq):
                val_eq = sum_radii(r_eq_arr)
                if val_eq > best_value:
                    best_value = val_eq
                    best_result = (x_eq.copy(), y_eq.copy(), r_eq_arr.copy(), w_eq)
            if time.time() < deadline - 18:
                x_cur, y_cur, r_cur, w_cur = best_result
                x_eq2, y_eq2, r_eq2_arr, w_eq2 = grow_equal_radius_clearance(x_cur, y_cur, r_cur, w_cur, max_rounds=8)
                if is_valid_equal_r(x_eq2, y_eq2, r_eq2_arr[0], w_eq2):
                    val_eq2 = sum_radii(r_eq2_arr)
                    if val_eq2 > best_value:
                        best_value = val_eq2
                        best_result = (x_eq2.copy(), y_eq2.copy(), r_eq2_arr.copy(), w_eq2)
        if time.time() < deadline - 15:
            x_cur, y_cur, r_cur, w_cur = best_result
            x_cur, y_cur, r_cur, w_cur = try_grow(x_cur, y_cur, r_cur, w_cur)
            if is_valid(x_cur, y_cur, r_cur, w_cur):
                val = sum_radii(r_cur)
                if val > best_value:
                    best_value = val
                    best_result = (x_cur.copy(), y_cur.copy(), r_cur.copy(), w_cur)

    # Phase 4: cooperative growth
    if best_result is not None and time.time() < deadline - 20:
        x_cur, y_cur, r_cur, w_cur = best_result
        if np.std(r_cur) < 0.005:
            x_cg, y_cg, r_cg, w_cg = cooperative_grow(x_cur, y_cur, r_cur, w_cur, max_rounds=10)
            if is_valid_equal_r(x_cg, y_cg, r_cg[0], w_cg):
                val_cg = sum_radii(r_cg)
                if val_cg > best_value:
                    best_value = val_cg
                    best_result = (x_cg.copy(), y_cg.copy(), r_cg.copy(), w_cg)

    # Phase 5: multi-start perturbation
    if best_result is not None and time.time() < deadline - 15:
        x_cur, y_cur, r_cur, w_cur = best_result
        x_p, y_p, r_p = multi_start_perturb(x_cur, y_cur, r_cur, w_cur, rng, n_perturb=4, scale=0.012)
        if is_valid(x_p, y_p, r_p, w_cur):
            val_p = sum_radii(r_p)
            if val_p > best_value:
                best_value = val_p
                best_result = (x_p.copy(), y_p.copy(), r_p.copy(), w_cur)

    # Phase 6: polish
    if best_result is not None and time.time() < deadline - 15:
        x_cur, y_cur, r_cur, w_cur = best_result
        x_cur, y_cur, r_cur, w_cur = polish(x_cur, y_cur, r_cur, w_cur)
        if is_valid(x_cur, y_cur, r_cur, w_cur):
            val = sum_radii(r_cur)
            if val > best_value:
                best_value = val
                best_result = (x_cur.copy(), y_cur.copy(), r_cur.copy(), w_cur)
        if time.time() < deadline - 10:
            x_cur, y_cur, r_cur, w_cur = best_result
            r_eq = r_cur[0] if np.allclose(r_cur, r_cur[0]) else min(r_cur)
            x_cp, y_cp = polish_clearance(x_cur, y_cur, np.full(N, r_eq), w_cur, beta=150.0)
            if is_valid_equal_r(x_cp, y_cp, r_eq, w_cur):
                r_arr_cp = np.full(N, r_eq)
                val_cp = sum_radii(r_arr_cp)
                if val_cp > best_value:
                    best_value = val_cp
                    best_result = (x_cp.copy(), y_cp.copy(), r_arr_cp.copy(), w_cur)

    # Phase 7: second optimization pass
    if best_result is not None and time.time() < deadline - 25:
        x_cur, y_cur, r_cur, w_cur = best_result
        result_eq = equal_r_slsqp(x_cur, y_cur, r_cur, w_cur, time_limit=15.0)
        if result_eq is not None:
            x_s, y_s, r_s, w_s = result_eq
            val_s = sum_radii(r_s)
            if val_s > best_value:
                best_value = val_s
                best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)
        if time.time() < deadline - 12:
            x_cur, y_cur, r_cur, w_cur = best_result
            result = smooth_slsqp(x_cur, y_cur, r_cur, w_cur, time_limit=10.0)
            if result is not None:
                x_s, y_s, r_s, w_s = result
                val_s = sum_radii(r_s)
                if val_s > best_value:
                    best_value = val_s
                    best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)
        if time.time() < deadline - 8:
            x_cur, y_cur, r_cur, w_cur = best_result
            r_eq = r_cur[0] if np.allclose(r_cur, r_cur[0]) else min(r_cur)
            result_lse = logsumexp_optimize(x_cur, y_cur, r_eq, w_cur, time_limit=8.0)
            if result_lse is not None:
                x_s, y_s, r_s, w_s = result_lse
                val_s = sum_radii(r_s)
                if val_s > best_value:
                    best_value = val_s
                    best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)

    # Phase 8: random restarts
    if time.time() < deadline - 30:
        for _ in range(12):
            if time.time() > deadline - 25:
                break
            w0 = rng.uniform(0.55, 1.45)
            h0 = 2.0 - w0
            if h0 < 0.35:
                continue
            if rng.random() < 0.5:
                x0, y0, r0 = grid_init(w0)
            else:
                x0, y0, r0 = hex_init(w0)
            x0 += rng.normal(0, 0.008, N)
            y0 += rng.normal(0, 0.008, N)
            x_t, y_t = force_relax(x0, y0, r0, w0, max_iter=300)
            if not is_valid(x_t, y_t, r0, w0):
                continue
            val = sum_radii(r0)
            if val > best_value * 0.92:
                if time.time() < deadline - 22:
                    r_init_val = r0[0] if np.allclose(r0, r0[0]) else min(r0)
                    x_eq, y_eq, r_eq = binary_search_max_radius(w0, x_t, y_t, r_init_val * 0.95, r_init_val * 1.1, max_iter=8)
                    if r_eq > 0.01:
                        r_arr = np.full(N, r_eq)
                        val_eq = sum_radii(r_arr)
                        if val_eq > best_value:
                            best_value = val_eq
                            best_result = (x_eq.copy(), y_eq.copy(), r_arr.copy(), w0)
                            continue
                    if time.time() < deadline - 20:
                        x_eq2, y_eq2, r_eq2 = binary_search_max_radius_clearance(w0, x_t, y_t, r_init_val * 0.95, r_init_val * 1.08, max_iter=6)
                        if r_eq2 > 0.01:
                            r_arr2 = np.full(N, r_eq2)
                            val_eq2 = sum_radii(r_arr2)
                            if val_eq2 > best_value:
                                best_value = val_eq2
                                best_result = (x_eq2.copy(), y_eq2.copy(), r_arr2.copy(), w0)
                                continue
                if time.time() < deadline - 15:
                    result = smooth_slsqp(x_t, y_t, r0, w0, time_limit=8.0)
                    if result is not None:
                        x_s, y_s, r_s, w_s = result
                        val_s = sum_radii(r_s)
                        if val_s > best_value:
                            best_value = val_s
                            best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)
                            continue
                if time.time() < deadline - 10:
                    x_g, y_g, r_g, w_g = try_grow(x_t, y_t, r0, w0, max_rounds=3)
                    if is_valid(x_g, y_g, r_g, w_g):
                        val_g = sum_radii(r_g)
                        if val_g > best_value:
                            best_value = val_g
                            best_result = (x_g.copy(), y_g.copy(), r_g.copy(), w_g)

    # Phase 9: NEWTON CONTACT SOLVER - the key refinement
    # At the optimum, circles form a rigid contact graph. The Newton solver
    # directly solves the equality system and can push r higher than the
    # penalty-based optimizers manage, because it finds the exact contact
    # configuration for a given active set.
    if best_result is not None and time.time() < deadline - 15:
        x_cur, y_cur, r_cur, w_cur = best_result
        r_eq = r_cur[0] if np.allclose(r_cur, r_cur[0]) else min(r_cur)
        # Try Newton polish at current radius, then grow
        for attempt in range(6):
            if time.time() > deadline - 10:
                break
            np_res = newton_polish(x_cur, y_cur, r_eq, w_cur, max_iter=400, tol=1e-14)
            if np_res is not None:
                x_n, y_n, r_n, w_n = np_res
                if r_n > r_eq:
                    x_cur, y_cur, r_eq, w_cur = x_n, y_n, r_n, w_n
                else:
                    x_cur, y_cur, w_cur = x_n, y_n, w_n
            # Try to grow r slightly and Newton-solve
            r_try = r_eq * 1.002
            # First get a good starting position via L-BFGS-B
            x_opt, y_opt, obj_val = find_positions_for_radius(r_try, w_cur, x_cur, y_cur, max_iter=80)
            if is_valid_equal_r(x_opt, y_opt, r_try, w_cur):
                x_cur, y_cur, r_eq = x_opt, y_opt, r_try
                # Newton polish
                np_res = newton_polish(x_cur, y_cur, r_eq, w_cur, max_iter=300, tol=1e-14)
                if np_res is not None:
                    x_n, y_n, r_n, w_n = np_res
                    x_cur, y_cur, r_eq, w_cur = x_n, y_n, r_n, w_n
            else:
                # Try Newton directly at r_try (it may find a valid config)
                x_n, y_n, r_n, w_n, conv, max_res = newton_contact_solve(
                    x_cur, y_cur, r_try, w_cur, max_iter=300, tol=1e-13)
                if is_valid_equal_r(x_n, y_n, r_n, w_n) and r_n >= r_try * 0.999:
                    x_cur, y_cur, r_eq, w_cur = x_n, y_n, r_n, w_n
                else:
                    break

        r_arr_final = np.full(N, r_eq)
        val_n = sum_radii(r_arr_final)
        if val_n > best_value:
            best_value = val_n
            best_result = (x_cur.copy(), y_cur.copy(), r_arr_final.copy(), w_cur)

    # Phase 10: Final polish + Newton
    if best_result is not None and time.time() < deadline - 8:
        x_cur, y_cur, r_cur, w_cur = best_result
        x_cur, y_cur, r_cur, w_cur = polish(x_cur, y_cur, r_cur, w_cur)
        if is_valid(x_cur, y_cur, r_cur, w_cur):
            val = sum_radii(r_cur)
            if val > best_value:
                best_value = val
                best_result = (x_cur.copy(), y_cur.copy(), r_cur.copy(), w_cur)
        if time.time() < deadline - 5:
            x_cur, y_cur, r_cur, w_cur = best_result
            r_eq = r_cur[0] if np.allclose(r_cur, r_cur[0]) else min(r_cur)
            x_cp, y_cp = polish_clearance(x_cur, y_cur, np.full(N, r_eq), w_cur, beta=200.0)
            if is_valid_equal_r(x_cp, y_cp, r_eq, w_cur):
                r_arr_cp = np.full(N, r_eq)
                val_cp = sum_radii(r_arr_cp)
                if val_cp > best_value:
                    best_value = val_cp
                    best_result = (x_cp.copy(), y_cp.copy(), r_arr_cp.copy(), w_cur)
        if time.time() < deadline - 4:
            x_cur, y_cur, r_cur, w_cur = best_result
            if np.std(r_cur) < 0.005:
                # Final Newton polish
                r_eq = r_cur[0]
                np_res = newton_polish(x_cur, y_cur, r_eq, w_cur, max_iter=500, tol=1e-15)
                if np_res is not None:
                    x_n, y_n, r_n, w_n = np_res
                    r_arr_n = np.full(N, r_n)
                    val_n = sum_radii(r_arr_n)
                    if val_n > best_value:
                        best_value = val_n
                        best_result = (x_n.copy(), y_n.copy(), r_arr_n.copy(), w_n)

    if best_result is None:
        w0 = 1.0
        cols = 5
        rows = 5
        cell_w = w0 / cols
        cell_h = (2.0 - w0) / rows
        r0 = min(cell_w, cell_h) / 2.0 * 0.45
        cells = []
        for i in range(N):
            row, col = divmod(i, cols)
            cx = (col + 0.5) * cell_w
            cy = (row + 0.5) * cell_h
            cells.append((cx, cy, r0))
        return np.array(cells)

    x_final, y_final, r_final, w_final = best_result

    # Safety shrink
    for _ in range(10):
        r_trial = r_final * (1 - 1e-10)
        if is_valid(x_final, y_final, r_trial, w_final):
            r_final = r_trial
        else:
            break

    # Final clearance-based polish
    if time.time() < deadline - 2:
        x_cp, y_cp = polish_clearance(x_final, y_final, r_final, w_final, beta=200.0)
        if is_valid(x_cp, y_cp, r_final, w_final):
            x_final, y_final = x_cp, y_cp

    # Final Newton polish
    if time.time() < deadline - 1:
        r_eq = r_final[0] if np.allclose(r_final, r_final[0]) else min(r_final)
        np_res = newton_polish(x_final, y_final, r_eq, w_final, max_iter=300, tol=1e-15)
        if np_res is not None:
            x_n, y_n, r_n, w_n = np_res
            r_arr_n = np.full(N, r_n)
            if is_valid(x_n, y_n, r_arr_n, w_n):
                val_n = sum_radii(r_arr_n)
                if val_n > best_value:
                    x_final, y_final, r_final, w_final = x_n, y_n, r_arr_n, w_n

    return np.column_stack((x_final, y_final, r_final))
