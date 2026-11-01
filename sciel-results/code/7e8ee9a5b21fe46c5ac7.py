import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def is_valid(x, y, r, w, tol=1e-8):
        h = 2.0 - w
        if w <= tol or h <= tol:
            return False
        if np.any(r <= tol):
            return False
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(N, k=1)
        if np.any(d_sq[iu] < r_sum[iu] ** 2 - tol):
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
                        cx = (ci + 0.5) * cell_w
                        cy = (ri + 0.5) * cell_h
                        x[idx] = cx
                        y[idx] = cy
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

    def contact_graph_init(w0, n=N):
        """Initialize using contact-graph structure: start from a hexagonal packing 
        and build the contact graph, then use it to guide optimization."""
        h0 = 2.0 - w0
        # Estimate radius from area
        r_est = np.sqrt(h0 * w0 / (n * np.pi / 0.9069))
        r_est = min(r_est, min(w0, h0) / 2.0 * 0.45)
        
        # Place circles in a hexagonal pattern
        dx = 2.0 * r_est * 1.001
        dy = np.sqrt(3) * r_est * 1.001
        
        x = np.zeros(n)
        y = np.zeros(n)
        r = np.full(n, r_est)
        
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
        
        # Fill remaining circles in gaps
        for i in range(idx, n):
            # Place in center of available space
            x[i] = w0 / 2.0 + (i - idx) * 0.01
            y[i] = h0 / 2.0 + (i - idx) * 0.01
            r[i] = r_est * 0.7
        
        return x, y, r

    def find_positions_for_radius(r, w, x_init, y_init, max_iter=200):
        """Find positions for N circles of equal radius r in rectangle w x (2-w)."""
        h = 2.0 - w
        z0 = np.concatenate([x_init, y_init])
        bounds = [(r, w - r)] * N + [(r, h - r)] * N
        
        try:
            res = minimize(equal_r_objective, z0, args=(r, w), method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-14, 'maxls': 50})
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            return x_opt, y_opt, res.fun
        except Exception:
            return x_init.copy(), y_init.copy(), np.inf

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

    def build_contact_graph(x, y, r, w, tol=1e-6):
        """Build contact graph: edges where circles are in contact (distance ≈ r_i + r_j)."""
        h = 2.0 - w
        contacts = []
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_sum = r[i] + r[j]
                if abs(d - r_sum) < tol:
                    contacts.append((i, j))
        return contacts

    def binary_search_max_radius(w, x_init, y_init, r_low, r_high, max_iter=12, tol=1e-6):
        """Binary search for maximum feasible equal radius in rectangle w x (2-w)."""
        h = 2.0 - w
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

    def force_relax(x, y, r, w, max_iter=800, tol=1e-10):
        """Force-directed relaxation to resolve overlaps and wall violations."""
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

    def optimize_aspect_ratio(x_init, y_init, r_init, w_init, max_iter=100):
        """Optimize aspect ratio w while keeping radii equal."""
        r = r_init[0]  # assume equal radii
        
        def obj(w):
            h = 2.0 - w
            if w < 0.3 or h < 0.3:
                return 1e8
            x_opt, y_opt, val = find_positions_for_radius(r, w, x_init, y_init, max_iter=80)
            return val + 0.01 * abs(w - w_init)  # mild regularization
        
        try:
            res = minimize(obj, w_init, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12})
            w_opt = float(np.clip(res.x, 0.3, 1.7))
            x_opt, y_opt, _ = find_positions_for_radius(r, w_opt, x_init, y_init, max_iter=150)
            return w_opt, x_opt, y_opt
        except Exception:
            return w_init, x_init.copy(), y_init.copy()

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
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-15))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        overlap_penalty = penalty_weight * np.sum(overlap ** 2)

        x_low_viol = np.maximum(r_clipped - x, 0.0)
        x_high_viol = np.maximum(x - (w - r_clipped), 0.0)
        y_low_viol = np.maximum(r_clipped - y, 0.0)
        y_high_viol = np.maximum(y - (h - r_clipped), 0.0)

        wall_penalty = penalty_weight * (np.sum(x_low_viol ** 2) + np.sum(x_high_viol ** 2) +
                                         np.sum(y_low_viol ** 2) + np.sum(y_high_viol ** 2))

        small_r_penalty = penalty_weight * np.sum(np.maximum(0.01 - r_clipped, 0.0) ** 2)

        return obj + overlap_penalty + wall_penalty + small_r_penalty

    def contact_graph_objective(z, r_target, w, contact_pairs):
        """Objective that only enforces constraints for contact pairs, 
        allowing non-contact pairs to have more freedom."""
        x = z[:N]
        y = z[N:2*N]
        h = 2.0 - w
        
        if w < 0.3 or h < 0.3:
            return 1e8
            
        pen = 0.0
        
        # Wall violations for all circles
        x_low = np.maximum(r_target - x, 0.0)
        x_high = np.maximum(x - (w - r_target), 0.0)
        y_low = np.maximum(r_target - y, 0.0)
        y_high = np.maximum(y - (h - r_target), 0.0)
        pen += 100.0 * (np.sum(x_low ** 2) + np.sum(x_high ** 2) +
                        np.sum(y_low ** 2) + np.sum(y_high ** 2))
        
        # Only enforce separation for contact pairs
        for i, j in contact_pairs:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy) + 1e-12
            overlap = 2 * r_target - d
            if overlap > 0:
                pen += 50.0 * overlap ** 2
        
        return pen

    def equal_r_slsqp(x_init, y_init, r_init, w_init, time_limit=15.0):
        """SLSQP optimization with equal radii constraint."""
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            return None
        
        # Use r_init as target radius, optimize positions and w
        r_target = r_init[0]
        h_init = 2.0 - w_init
        
        def obj(z):
            x = z[:N]
            y = z[N:2*N]
            w = z[2*N]
            h = 2.0 - w
            
            if w < 0.3 or h < 0.3:
                return 1e8
            
            # Wall penalties
            x_low = np.maximum(r_target - x, 0.0)
            x_high = np.maximum(x - (w - r_target), 0.0)
            y_low = np.maximum(r_target - y, 0.0)
            y_high = np.maximum(y - (h - r_target), 0.0)
            wall_pen = 500.0 * (np.sum(x_low ** 2) + np.sum(x_high ** 2) +
                               np.sum(y_low ** 2) + np.sum(y_high ** 2))
            
            # Overlap penalties
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d_sq = dx * dx + dy * dy
            iu = np.triu_indices(N, k=1)
            d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-15))
            overlap = np.maximum(2 * r_target - d_pairs, 0.0)
            overlap_pen = 500.0 * np.sum(overlap ** 2)
            
            # Reward larger w that allows more space
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
            
            # Try to fix
            x_f, y_f = force_relax(x_opt, y_opt, np.full(N, r_target), w_opt, max_iter=100)
            if is_valid_equal_r(x_f, y_f, r_target, w_opt):
                r_arr = np.full(N, r_target)
                return x_f, y_f, r_arr, w_opt
                
        except Exception:
            pass
        
        return None

    def smooth_slsqp(x_init, y_init, r_init, w_init, time_limit=18.0):
        """SLSQP with smooth vectorized objective and explicit bounds."""
        z0 = np.concatenate([x_init, y_init, r_init, [w_init]])

        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            return None

        maxiter = int(min(500, max(100, remaining * 25)))

        h_init = 2.0 - w_init
        bounds = []
        for i in range(N):
            bounds.append((r_init[i] * 0.3, w_init - r_init[i] * 0.3))  # x
        for i in range(N):
            bounds.append((r_init[i] * 0.3, max(r_init[i] * 0.3, h_init - r_init[i] * 0.3)))  # y
        for i in range(N):
            bounds.append((0.01, 0.45))  # r
        bounds.append((0.35, 1.65))  # w

        try:
            res = minimize(smooth_objective, z0, method='SLSQP', bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})

            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            w_opt = res.x[3*N]

            if np.any(r_opt < 0.005):
                return None

            # Validate
            if is_valid(x_opt, y_opt, r_opt, w_opt):
                return x_opt, y_opt, r_opt, w_opt

            # Try to fix small violations
            x_f, y_f = force_relax(x_opt, y_opt, r_opt, w_opt, max_iter=200)
            if is_valid(x_f, y_f, r_opt, w_opt):
                return x_f, y_f, r_opt, w_opt

        except Exception:
            pass

        return None

    def grow_equal_radius(x, y, r, w, max_rounds=10):
        """Grow equal radius and re-optimize positions."""
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

    def relax_positions(x, y, r, w, max_iter=300, ftol=1e-14):
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
        bounds = [(r[i], w - r[i]) for i in range(N)] + \
                 [(r[i], h - r[i]) for i in range(N)]

        try:
            res = minimize(obj, z0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': max_iter, 'ftol': ftol, 'maxls': 50})
            return res.x[:N], res.x[N:2*N]
        except Exception:
            return x.copy(), y.copy()

    def contact_graph_relax(x, y, r_target, w, max_iter=200, ftol=1e-14):
        """Relax positions using contact graph structure - only enforce constraints 
        for pairs that should be in contact."""
        h = 2.0 - w
        
        # Build contact graph from current configuration
        contacts = build_contact_graph(x, y, np.full(N, r_target), w, tol=0.01)
        
        if len(contacts) == 0:
            # No contacts found, fall back to full relaxation
            return relax_positions(x, y, np.full(N, r_target), w, max_iter=max_iter)
        
        def obj(z):
            xz = z[:N]
            yz = z[N:2*N]
            pen = 0.0
            
            # Wall constraints
            x_low = np.maximum(r_target - xz, 0.0)
            x_high = np.maximum(xz - (w - r_target), 0.0)
            y_low = np.maximum(r_target - yz, 0.0)
            y_high = np.maximum(yz - (h - r_target), 0.0)
            pen += 100.0 * (np.sum(x_low ** 2) + np.sum(x_high ** 2) +
                           np.sum(y_low ** 2) + np.sum(y_high ** 2))
            
            # Only enforce separation for contact pairs
            for i, j in contacts:
                dx = xz[i] - xz[j]
                dy = yz[i] - yz[j]
                d = np.sqrt(dx * dx + dy * dy) + 1e-12
                overlap = 2 * r_target - d
                if overlap > 0:
                    pen += 50.0 * overlap ** 2
            
            return pen

        z0 = np.concatenate([x, y])
        bounds = [(r_target, w - r_target)] * N + [(r_target, h - r_target)] * N

        try:
            res = minimize(obj, z0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': max_iter, 'ftol': ftol, 'maxls': 50})
            return res.x[:N], res.x[N:2*N]
        except Exception:
            return x.copy(), y.copy()

    def try_grow(x, y, r, w, max_rounds=5):
        """Iteratively grow radii and relax positions."""
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
                    x_t, y_t = relax_positions(x, y, r_new, w, max_iter=40, ftol=1e-12)
                    if is_valid(x_t, y_t, r_new, w):
                        x, y, r = x_t, y_t, r_new
                        improved = True
                    else:
                        break
            if not improved:
                break
        return x, y, r, w

    def polish(x, y, r, w):
        """Nelder-Mead polish on positions only."""
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

    best_value = 0.0
    best_result = None
    rng = np.random.default_rng(42)

    # Phase 1: Try multiple aspect ratios with grid, hex, and contact-graph initialization
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.7, 0.55, 2.0, 0.5, 1.1, 0.9, 1.25, 0.8, 1.4, 0.7, 1.6, 0.65]

    for w0 in aspect_ratios:
        if time.time() > deadline - 40:
            break
        h0 = 2.0 - w0
        if h0 < 0.35:
            continue

        # Grid init
        x0, y0, r0 = grid_init(w0)
        x_t, y_t = force_relax(x0, y0, r0, w0, max_iter=500)
        if is_valid(x_t, y_t, r0, w0):
            val = sum_radii(r0)
            if val > best_value:
                best_value = val
                best_result = (x_t.copy(), y_t.copy(), r0.copy(), w0)

        # Hex init
        x0h, y0h, r0h = hex_init(w0)
        x_th, y_th = force_relax(x0h, y0h, r0h, w0, max_iter=500)
        if is_valid(x_th, y_th, r0h, w0):
            val_h = sum_radii(r0h)
            if val_h > best_value:
                best_value = val_h
                best_result = (x_th.copy(), y_th.copy(), r0h.copy(), w0)

        # Contact graph init
        x0c, y0c, r0c = contact_graph_init(w0)
        x_tc, y_tc = force_relax(x0c, y0c, r0c, w0, max_iter=500)
        if is_valid(x_tc, y_tc, r0c, w0):
            val_c = sum_radii(r0c)
            if val_c > best_value:
                best_value = val_c
                best_result = (x_tc.copy(), y_tc.copy(), r0c.copy(), w0)

        # Equal-radius binary search from this init
        if time.time() < deadline - 35:
            r_init_val = r0[0] if np.allclose(r0, r0[0]) else min(r0)
            x_eq, y_eq, r_eq = binary_search_max_radius(w0, x_t, y_t, r_init_val * 0.9, r_init_val * 1.15, max_iter=10)
            if r_eq > 0.01:
                r_arr = np.full(N, r_eq)
                val_eq = sum_radii(r_arr)
                if val_eq > best_value:
                    best_value = val_eq
                    best_result = (x_eq.copy(), y_eq.copy(), r_arr.copy(), w0)

    # Phase 2: Equal-radius SLSQP from best
    if best_result is not None and time.time() < deadline - 30:
        x_cur, y_cur, r_cur, w_cur = best_result
        result_eq = equal_r_slsqp(x_cur, y_cur, r_cur, w_cur, time_limit=18.0)
        if result_eq is not None:
            x_s, y_s, r_s, w_s = result_eq
            val_s = sum_radii(r_s)
            if val_s > best_value:
                best_value = val_s
                best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)
        
        # Also try standard SLSQP
        if time.time() < deadline - 15:
            result = smooth_slsqp(x_cur, y_cur, r_cur, w_cur, time_limit=12.0)
            if result is not None:
                x_s, y_s, r_s, w_s = result
                val_s = sum_radii(r_s)
                if val_s > best_value:
                    best_value = val_s
                    best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)
        
        # Try contact-graph-based relaxation
        if time.time() < deadline - 12 and np.std(r_cur) < 0.005:
            r_target = r_cur[0]
            x_cg, y_cg = contact_graph_relax(x_cur, y_cur, r_target, w_cur, max_iter=150)
            if is_valid_equal_r(x_cg, y_cg, r_target, w_cur):
                r_arr = np.full(N, r_target)
                val_cg = sum_radii(r_arr)
                if val_cg > best_value:
                    best_value = val_cg
                    best_result = (x_cg.copy(), y_cg.copy(), r_arr.copy(), w_cur)

    # Phase 3: Grow radii from best
    if best_result is not None and time.time() < deadline - 20:
        x_cur, y_cur, r_cur, w_cur = best_result
        
        # Try equal-radius growth if radii are close to equal
        if np.std(r_cur) < 0.005:
            x_eq, y_eq, r_eq_arr, w_eq = grow_equal_radius(x_cur, y_cur, r_cur, w_cur, max_rounds=8)
            if is_valid_equal_r(x_eq, y_eq, r_eq_arr[0], w_eq):
                val_eq = sum_radii(r_eq_arr)
                if val_eq > best_value:
                    best_value = val_eq
                    best_result = (x_eq.copy(), y_eq.copy(), r_eq_arr.copy(), w_eq)
        
        # Standard growth
        if time.time() < deadline - 12:
            x_cur, y_cur, r_cur, w_cur = best_result
            x_cur, y_cur, r_cur, w_cur = try_grow(x_cur, y_cur, r_cur, w_cur)
            if is_valid(x_cur, y_cur, r_cur, w_cur):
                val = sum_radii(r_cur)
                if val > best_value:
                    best_value = val
                    best_result = (x_cur.copy(), y_cur.copy(), r_cur.copy(), w_cur)
        
        # Contact-graph-guided growth
        if time.time() < deadline - 10 and np.std(r_cur) < 0.005:
            r_target = r_cur[0]
            for _ in range(10):
                if time.time() > deadline - 8:
                    break
                r_new = r_target * 1.005
                x_cg, y_cg = contact_graph_relax(x_cur, y_cur, r_new, w_cur, max_iter=100)
                if is_valid_equal_r(x_cg, y_cg, r_new, w_cur):
                    x_cur, y_cur = x_cg, y_cg
                    r_target = r_new
                else:
                    break
            r_arr = np.full(N, r_target)
            val_cg = sum_radii(r_arr)
            if val_cg > best_value:
                best_value = val_cg
                best_result = (x_cur.copy(), y_cur.copy(), r_arr.copy(), w_cur)

    # Phase 4: Polish
    if best_result is not None and time.time() < deadline - 15:
        x_cur, y_cur, r_cur, w_cur = best_result
        x_cur, y_cur, r_cur, w_cur = polish(x_cur, y_cur, r_cur, w_cur)
        if is_valid(x_cur, y_cur, r_cur, w_cur):
            val = sum_radii(r_cur)
            if val > best_value:
                best_value = val
                best_result = (x_cur.copy(), y_cur.copy(), r_cur.copy(), w_cur)

    # Phase 5: Second optimization pass after growth and polish
    if best_result is not None and time.time() < deadline - 25:
        x_cur, y_cur, r_cur, w_cur = best_result
        
        # Equal-radius SLSQP
        result_eq = equal_r_slsqp(x_cur, y_cur, r_cur, w_cur, time_limit=15.0)
        if result_eq is not None:
            x_s, y_s, r_s, w_s = result_eq
            val_s = sum_radii(r_s)
            if val_s > best_value:
                best_value = val_s
                best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)
        
        # Standard SLSQP
        if time.time() < deadline - 10:
            result = smooth_slsqp(x_cur, y_cur, r_cur, w_cur, time_limit=10.0)
            if result is not None:
                x_s, y_s, r_s, w_s = result
                val_s = sum_radii(r_s)
                if val_s > best_value:
                    best_value = val_s
                    best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)

    # Phase 6: Additional random restarts if time permits
    if time.time() < deadline - 25:
        for _ in range(10):
            if time.time() > deadline - 20:
                break
            w0 = rng.uniform(0.55, 1.45)
            h0 = 2.0 - w0
            if h0 < 0.35:
                continue

            choice = rng.random()
            if choice < 0.33:
                x0, y0, r0 = grid_init(w0)
            elif choice < 0.66:
                x0, y0, r0 = hex_init(w0)
            else:
                x0, y0, r0 = contact_graph_init(w0)

            x0 += rng.normal(0, 0.008, N)
            y0 += rng.normal(0, 0.008, N)

            x_t, y_t = force_relax(x0, y0, r0, w0, max_iter=300)
            if not is_valid(x_t, y_t, r0, w0):
                continue
            val = sum_radii(r0)

            if val > best_value * 0.92:
                # Try equal-radius binary search
                if time.time() < deadline - 18:
                    r_init_val = r0[0] if np.allclose(r0, r0[0]) else min(r0)
                    x_eq, y_eq, r_eq = binary_search_max_radius(w0, x_t, y_t, r_init_val * 0.95, r_init_val * 1.1, max_iter=8)
                    if r_eq > 0.01:
                        r_arr = np.full(N, r_eq)
                        val_eq = sum_radii(r_arr)
                        if val_eq > best_value:
                            best_value = val_eq
                            best_result = (x_eq.copy(), y_eq.copy(), r_arr.copy(), w0)
                            continue
                
                if time.time() < deadline - 12:
                    result = smooth_slsqp(x_t, y_t, r0, w0, time_limit=8.0)
                    if result is not None:
                        x_s, y_s, r_s, w_s = result
                        val_s = sum_radii(r_s)
                        if val_s > best_value:
                            best_value = val_s
                            best_result = (x_s.copy(), y_s.copy(), r_s.copy(), w_s)
                            continue

                if time.time() < deadline - 8:
                    x_g, y_g, r_g, w_g = try_grow(x_t, y_t, r0, w0, max_rounds=3)
                    if is_valid(x_g, y_g, r_g, w_g):
                        val_g = sum_radii(r_g)
                        if val_g > best_value:
                            best_value = val_g
                            best_result = (x_g.copy(), y_g.copy(), r_g.copy(), w_g)

    # Phase 7: Final polish
    if best_result is not None and time.time() < deadline - 8:
        x_cur, y_cur, r_cur, w_cur = best_result
        x_cur, y_cur, r_cur, w_cur = polish(x_cur, y_cur, r_cur, w_cur)
        if is_valid(x_cur, y_cur, r_cur, w_cur):
            val = sum_radii(r_cur)
            if val > best_value:
                best_value = val
                best_result = (x_cur.copy(), y_cur.copy(), r_cur.copy(), w_cur)

        # Final grow
        if time.time() < deadline - 4:
            x_cur, y_cur, r_cur, w_cur = best_result
            if np.std(r_cur) < 0.005:
                x_eq, y_eq, r_eq_arr, w_eq = grow_equal_radius(x_cur, y_cur, r_cur, w_cur, max_rounds=5)
                if is_valid_equal_r(x_eq, y_eq, r_eq_arr[0], w_eq):
                    val_eq = sum_radii(r_eq_arr)
                    if val_eq > best_value:
                        best_value = val_eq
                        best_result = (x_eq.copy(), y_eq.copy(), r_eq_arr.copy(), w_eq)
            else:
                x_cur, y_cur, r_cur, w_cur = try_grow(x_cur, y_cur, r_cur, w_cur, max_rounds=3)
                if is_valid(x_cur, y_cur, r_cur, w_cur):
                    val = sum_radii(r_cur)
                    if val > best_value:
                        best_value = val
                        best_result = (x_cur.copy(), y_cur.copy(), r_cur.copy(), w_cur)

    if best_result is None:
        # Fallback: simple grid
        w0 = 1.0
        h0 = 1.0
        cols = 5
        rows = 5
        cell_w = w0 / cols
        cell_h = h0 / rows
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

    return np.column_stack((x_final, y_final, r_final))
