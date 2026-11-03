import numpy as np
import time
import random
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    best_value = -np.inf
    best_result = None

    def _strictly_valid(circles, w, h, tol=1e-8):
        """Full-pair strict validity check. circles: (N,3)."""
        n = circles.shape[0]
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        if np.any(rs < 1e-9):
            return False
        if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
            return False
        if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
            return False
        for i in range(n):
            dx = xs[i] - xs[i + 1:]
            dy = ys[i] - ys[i + 1:]
            d2 = dx * dx + dy * dy
            need = rs[i] + rs[i + 1:] - tol
            if np.any(d2 < need * need):
                return False
        return True

    def _evaluate(circles, w, h, tol=1e-7):
        """Return sum of radii if valid, else -inf."""
        if not _strictly_valid(circles, w, h, tol):
            return -np.inf
        return float(np.sum(circles[:, 2]))

    def check_and_update(circles, w, h):
        nonlocal best_value, best_result
        if time_left() < 1.0:
            return
        val = _evaluate(circles, w, h)
        if val > best_value:
            best_value = val
            best_result = circles.copy()

    def _softplus(x, beta=100.0):
        xb = beta * x
        return np.where(xb > 0,
                        x + np.log1p(np.exp(-xb)) / beta,
                        np.log1p(np.exp(xb)) / beta)

    def _softplus_grad(x, beta=100.0):
        return 1.0 / (1.0 + np.exp(-beta * x))

    def _objective_and_grad(x, w, h, beta=100.0, eq_weight=0.05):
        """Softplus-penalized objective with analytical gradients (vectorized)."""
        c = x.reshape(-1, 3)
        xs = c[:, 0]
        ys = c[:, 1]
        rs = c[:, 2]

        obj = -np.sum(rs)
        grad = np.zeros_like(x)

        # Equalization pressure
        if eq_weight > 0:
            r_mean = np.mean(rs)
            r_dev = rs - r_mean
            obj += eq_weight * np.sum(r_dev * r_dev)
            grad[2::3] += -2.0 * eq_weight * r_dev

        # Boundary penalties
        v1 = rs - xs
        obj += np.sum(_softplus(v1, beta)); g1 = _softplus_grad(v1, beta)
        grad[0::3] -= g1; grad[2::3] += g1
        v2 = xs + rs - w
        obj += np.sum(_softplus(v2, beta)); g2 = _softplus_grad(v2, beta)
        grad[0::3] += g2; grad[2::3] += g2
        v3 = rs - ys
        obj += np.sum(_softplus(v3, beta)); g3 = _softplus_grad(v3, beta)
        grad[1::3] -= g3; grad[2::3] += g3
        v4 = ys + rs - h
        obj += np.sum(_softplus(v4, beta)); g4 = _softplus_grad(v4, beta)
        grad[1::3] += g4; grad[2::3] += g4
        v5 = -rs
        obj += np.sum(_softplus(v5, beta)); g5 = _softplus_grad(v5, beta)
        grad[2::3] -= g5

        # Pairwise overlap penalties
        for i in range(N):
            dx = xs[i] - xs[i + 1:]
            dy = ys[i] - ys[i + 1:]
            d = np.sqrt(dx * dx + dy * dy + 1e-12)
            v = rs[i] + rs[i + 1:] - d
            obj += np.sum(_softplus(v, beta))
            g = _softplus_grad(v, beta)
            grad[3 * i]     += -np.sum(g * dx / d)
            grad[3 * i + 1] += -np.sum(g * dy / d)
            grad[3 * i + 2] += np.sum(g)
            idx = np.arange(i + 1, N)
            grad[3 * idx]     += g * dx / d
            grad[3 * idx + 1] += g * dy / d
            grad[3 * idx + 2] += g

        grad[2::3] += -1.0
        return obj, grad

    def _delaunay_neighbors(centers):
        try:
            tri = Delaunay(centers)
            nbrs = [set() for _ in range(N)]
            for i, j, k in tri.simplices:
                nbrs[i].add(j); nbrs[i].add(k)
                nbrs[j].add(i); nbrs[j].add(k)
                nbrs[k].add(i); nbrs[k].add(j)
            return [sorted(s) for s in nbrs]
        except Exception:
            return [list(range(N)) for _ in range(N)]

    def _fast_valid(circles, w, h, neighbors, tol=1e-9):
        n = circles.shape[0]
        xs = circles[:, 0]; ys = circles[:, 1]; rs = circles[:, 2]
        if np.any(rs < 1e-9):
            return False
        if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
            return False
        if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
            return False
        for i in range(n):
            for j in neighbors[i]:
                if j <= i:
                    continue
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d2 = dx * dx + dy * dy
                need = rs[i] + rs[j] - tol
                if d2 < need * need:
                    return False
        return True

    def _hex_init(w, h, seed):
        rng = random.Random(seed)
        row_configs = [
            [3, 4, 4, 4, 3, 3],
            [4, 4, 4, 4, 3, 2],
            [2, 4, 4, 4, 4, 3],
            [3, 4, 4, 4, 4, 2],
            [4, 3, 4, 4, 4, 2],
            [2, 3, 4, 4, 4, 4],
            [3, 3, 4, 4, 4, 3],
            [4, 4, 3, 4, 4, 2],
            [2, 4, 3, 4, 4, 4],
            [3, 4, 3, 4, 4, 3],
            [5, 4, 4, 4, 4],
            [4, 5, 4, 4, 4],
            [3, 5, 4, 4, 5],
            [4, 4, 5, 4, 4],
            [2, 5, 4, 5, 5],
            [5, 5, 5, 5, 1],
            [1, 5, 5, 5, 5],
            [3, 4, 5, 4, 5],
            [5, 4, 5, 4, 3],
            [4, 4, 4, 4, 4, 1],
            [1, 4, 4, 4, 4, 4],
            [2, 3, 5, 4, 4, 3],
            [3, 4, 4, 5, 3, 2],
        ]
        best = None
        best_val = -np.inf
        for row_counts in row_configs:
            if sum(row_counts) != N:
                continue
            n_rows = len(row_counts)
            max_cols = max(row_counts)
            est_r_h = w / (max_cols + 0.5)
            est_r_v = h / (n_rows * np.sqrt(3) / 2 + 1.0)
            r_est = min(est_r_h, est_r_v) * 0.95
            if r_est < 0.01:
                continue
            circles = []
            for row_idx, n_cols in enumerate(row_counts):
                y = (row_idx + 0.5) * r_est * np.sqrt(3)
                x_offset = r_est if row_idx % 2 == 1 else 0.0
                row_width = n_cols * 2 * r_est
                center_x = w / 2
                for col_idx in range(n_cols):
                    x = center_x - row_width / 2 + x_offset + (col_idx + 0.5) * 2 * r_est
                    circles.append([x, y, r_est])
            circles = np.array(circles)
            valid = True
            for i in range(N):
                xi, yi, ri = circles[i]
                if ri < 1e-9 or xi - ri < -1e-6 or xi + ri > w + 1e-6 or yi - ri < -1e-6 or yi + ri > h + 1e-6:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i + 1, N):
                        xi, yi, ri = circles[i]
                        xj, yj, rj = circles[j]
                        dist_sq = (xi - xj) ** 2 + (yi - yj) ** 2
                        min_dist = ri + rj
                        if dist_sq < (min_dist - 1e-7) ** 2:
                            valid = False
                            break
                    if not valid:
                        break
            if valid:
                val = float(np.sum(circles[:, 2]))
                if val > best_val:
                    best_val = val
                    best = circles.copy()
        if best is None:
            # Fallback: simple grid
            side = int(np.ceil(np.sqrt(N)))
            circles = []
            for i in range(N):
                row = i // side
                col = i % side
                x = (col + 0.5) * w / side
                y = (row + 0.5) * h / side
                r = 0.5 * min(x, w - x, y, h - y) * 0.95
                circles.append([x, y, r])
            best = np.array(circles)
        return best

    def _grid_init(w, h, seed):
        rng = random.Random(seed)
        best = None
        best_val = -np.inf
        for rows in range(3, 8):
            cols = int(np.ceil(N / rows))
            if cols < 2:
                continue
            circles = []
            for i in range(N):
                row = i // cols
                col = i % cols
                x = (col + 0.5) * (w / cols)
                y = (row + 0.5) * (h / rows)
                r = 0.5 * min(x, w - x, y, h - y) * 0.95
                circles.append([x, y, r])
            circles = np.array(circles)
            if _strictly_valid(circles, w, h, tol=1e-6):
                val = float(np.sum(circles[:, 2]))
                if val > best_val:
                    best_val = val
                    best = circles.copy()
        if best is None:
            side = int(np.ceil(np.sqrt(N)))
            circles = []
            for i in range(N):
                row = i // side
                col = i % side
                x = (col + 0.5) * w / side
                y = (row + 0.5) * h / side
                r = 0.5 * min(x, w - x, y, h - y) * 0.9
                circles.append([x, y, r])
            best = np.array(circles)
        return best

    def _random_init(w, h, seed):
        rng = random.Random(seed)
        circles = []
        for _ in range(N):
            x = rng.uniform(0.1, w - 0.1)
            y = rng.uniform(0.1, h - 0.1)
            r = rng.uniform(0.05, 0.2)
            circles.append([x, y, r])
        return np.array(circles)

    def _solve_lp_radii(xs, ys, w, h):
        """
        Given fixed centers (xs, ys) and rectangle (w, h), solve for the maximum
        sum of radii such that circles do not overlap and stay within bounds.
        """
        n = N
        c = -np.ones(n)
        A_ub_list = []
        b_ub_list = []
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_ub_list.append(row)
                b_ub_list.append(dist)
                
        for i in range(n):
            row = np.zeros(n); row[i] = 1.0
            A_ub_list.append(row); b_ub_list.append(xs[i])
            row = np.zeros(n); row[i] = 1.0
            A_ub_list.append(row); b_ub_list.append(w - xs[i])
            row = np.zeros(n); row[i] = 1.0
            A_ub_list.append(row); b_ub_list.append(ys[i])
            row = np.zeros(n); row[i] = 1.0
            A_ub_list.append(row); b_ub_list.append(h - ys[i])
            
        A_ub = np.array(A_ub_list)
        b_ub = np.array(b_ub_list)
        bounds = [(0, None)] * n
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                return res.x
        except Exception:
            pass
        return None

    def _polish(circles, w, h, max_iter=300):
        """Polish a valid configuration using gradient descent + LP radius fix."""
        if time_left() < 2.0:
            return circles
        
        x = circles.flatten().copy()
        
        # Phase 1: Gradient optimization
        try:
            max_r = min(w, h) / 2.0
            bounds = []
            for i in range(N):
                bounds.append((0.0, w))
                bounds.append((0.0, h))
                bounds.append((0.0, max_r))
            
            res = minimize(_objective_and_grad, x, args=(w, h),
                           jac=True, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
            
            if res.success:
                x = res.x
        except Exception:
            pass
        
        # Phase 2: LP radius polishing
        xs = x[0::3].copy()
        ys = x[1::3].copy()
        xs = np.clip(xs, 1e-6, w - 1e-6)
        ys = np.clip(ys, 1e-6, h - 1e-6)
        
        rs_lp = _solve_lp_radii(xs, ys, w, h)
        if rs_lp is not None:
            new_circles = np.column_stack([xs, ys, rs_lp])
            if _strictly_valid(new_circles, w, h, tol=1e-7):
                # Phase 3: Re-optimize centers
                x2 = new_circles.flatten().copy()
                try:
                    res2 = minimize(_objective_and_grad, x2, args=(w, h),
                                   jac=True, method='L-BFGS-B',
                                   bounds=bounds,
                                   options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
                    if res2.success:
                        x2 = res2.x
                        xs2 = x2[0::3].copy()
                        ys2 = x2[1::3].copy()
                        xs2 = np.clip(xs2, 1e-6, w - 1e-6)
                        ys2 = np.clip(ys2, 1e-6, h - 1e-6)
                        rs_lp2 = _solve_lp_radii(xs2, ys2, w, h)
                        if rs_lp2 is not None:
                            new_circles2 = np.column_stack([xs2, ys2, rs_lp2])
                            if _strictly_valid(new_circles2, w, h, tol=1e-7):
                                if np.sum(new_circles2[:, 2]) > np.sum(new_circles[:, 2]):
                                    new_circles = new_circles2
                except Exception:
                    pass
                return new_circles
        
        return circles

    # Main search loop
    w_values = np.linspace(0.6, 1.4, 15)
    rng = random.Random(42)
    
    for w in w_values:
        if time_left() < 25.0:
            break
        h = 2.0 - w
        
        n_restarts = 8
        for k in range(n_restarts):
            if time_left() < 15.0:
                break
            
            seed = rng.randint(0, 1000000)
            
            if k % 3 == 0:
                init = _hex_init(w, h, seed)
            elif k % 3 == 1:
                init = _grid_init(w, h, seed)
            else:
                init = _random_init(w, h, seed)
            
            if not _strictly_valid(init, w, h, tol=1e-6):
                # Try to fix by shrinking radii
                init[:, 2] *= 0.9
                if not _strictly_valid(init, w, h, tol=1e-6):
                    init[:, 2] *= 0.9
                    if not _strictly_valid(init, w, h, tol=1e-6):
                        continue
            
            polished = _polish(init, w, h, max_iter=200)
            check_and_update(polished, w, h)
            
            # Additional polish if time allows
            if time_left() > 30.0:
                polished2 = _polish(polished, w, h, max_iter=300)
                check_and_update(polished2, w, h)

    # Final high-precision polish on best result
    if best_result is not None and time_left() > 5.0:
        # Determine w and h from the best result's bounding box
        # Actually, we need to track which w/h was used
        # For now, just re-polish with the best w/h we can infer
        # We'll just do a final LP polish
        xs = best_result[:, 0]
        ys = best_result[:, 1]
        # Infer w, h from the context - we'll use the last w_values value that was best
        # Actually, let's just do a generic final check
        for w in w_values:
            h = 2.0 - w
            if _strictly_valid(best_result, w, h, tol=1e-8):
                polished = _polish(best_result, w, h, max_iter=500)
                check_and_update(polished, w, h)
                break

    if best_result is not None:
        best_result[:, 2] = np.maximum(0, best_result[:, 2])
        return best_result
    else:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
