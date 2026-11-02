import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay

N = 21

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s (evaluator kills at 150s, but keep conservative)

    def evaluate(circles, w, h):
        """Return sum of radii if valid, else -inf."""
        n = len(circles)
        for i in range(n):
            x, y, r = circles[i]
            if x - r < -1e-6 or x + r > w + 1e-6 or y - r < -1e-6 or y + r > h + 1e-6:
                return -np.inf
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                d = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                if d < ri + rj - 1e-6:
                    return -np.inf
        return float(np.sum(circles[:, 2]))

    def delaunay_edges(xs, ys):
        """Return list of (i, j) edges from Delaunay triangulation of centers."""
        points = np.column_stack((xs, ys))
        try:
            tri = Delaunay(points)
        except Exception:
            return None
        edges = set()
        for simplex in tri.simplices:
            edges.add(tuple(sorted((simplex[0], simplex[1]))))
            edges.add(tuple(sorted((simplex[1], simplex[2]))))
            edges.add(tuple(sorted((simplex[0], simplex[2]))))
        return list(edges)

    def solve_lp_radii(xs, ys, w, h):
        """
        Solve LP for per-circle radii given fixed centers and rectangle.
        Maximize sum(r) s.t. r_i + r_j <= d_ij (Delaunay edges)
                             r_i <= min(x_i, w-x_i, y_i, h-y_i)
                             r_i >= 0
        Returns array of radii.
        """
        limits = np.minimum(np.minimum(xs, w - xs), np.minimum(ys, h - ys))
        if np.any(limits < 0):
            return np.zeros(N)

        A_list = []
        b_list = []
        edges = delaunay_edges(xs, ys)
        if edges is not None:
            for i, j in edges:
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                row = np.zeros(N)
                row[i] = 1
                row[j] = 1
                A_list.append(row)
                b_list.append(dist)

        A_bound = np.eye(N)
        if A_list:
            A_ub = np.vstack([A_list, A_bound])
            b_ub = np.concatenate([b_list, limits])
        else:
            A_ub = A_bound
            b_ub = limits

        c = -np.ones(N)
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
            if res.success:
                return np.maximum(0.0, res.x)
        except Exception:
            pass
        return np.maximum(0.0, limits * 0.9)

    def make_hex(rows, w, h):
        """Hexagonal layout of given per-row counts, scaled to fit w x h."""
        pts = []
        for k, n in enumerate(rows):
            y = k * np.sqrt(3.0)
            off = 1.0 if k % 2 == 1 else 0.0
            for i in range(n):
                pts.append((i * 2.0 + off, y))
        pts = np.array(pts[:N])
        if len(pts) < N:
            return None
        xs = pts[:, 0]
        ys = pts[:, 1]
        cx = 0.5 * (xs.max() + xs.min())
        cy = 0.5 * (ys.max() + ys.min())
        xs = xs - cx
        ys = ys - cy
        wc = xs.max() - xs.min()
        hc = ys.max() - ys.min()
        wt = wc + 2.0
        ht = hc + 2.0
        s = 2.0 / (wt + ht)
        xs_f = xs * s + 0.5 * (s * wt)
        ys_f = ys * s + 0.5 * (s * ht)
        return xs_f, ys_f, s * wt, s * ht

    def make_grid(rows, cols, w, h):
        """Grid layout, scaled to fit w x h."""
        pts = []
        for j in range(rows):
            for i in range(cols):
                if len(pts) >= N:
                    break
                pts.append((i * 2.0, j * 2.0))
        pts = np.array(pts[:N])
        if len(pts) < N:
            return None
        xs = pts[:, 0]
        ys = pts[:, 1]
        cx = 0.5 * (xs.max() + xs.min())
        cy = 0.5 * (ys.max() + ys.min())
        xs = xs - cx
        ys = ys - cy
        wt = (xs.max() - xs.min()) + 2.0
        ht = (ys.max() - ys.min()) + 2.0
        s = 2.0 / (wt + ht)
        xs_f = xs * s + 0.5 * (s * wt)
        ys_f = ys * s + 0.5 * (s * ht)
        return xs_f, ys_f, s * wt, s * ht

    def make_random(w, h, seed):
        rng = np.random.RandomState(seed)
        xs = rng.uniform(0.1, 0.9, N) * w
        ys = rng.uniform(0.1, 0.9, N) * h
        return xs, ys, w, h

    def physics_settle(xs, ys, w, h, n_steps=500, dt=0.002, deadline=None):
        """
        Physics-based settling using Lennard-Jones-like repulsion and soft boundary.
        Returns settled positions.
        """
        xs = xs.copy().astype(float)
        ys = ys.copy().astype(float)
        vx = np.zeros(N)
        vy = np.zeros(N)
        
        sigma = 0.15  # interaction range scale
        k_repel = 50.0
        k_boundary = 100.0
        damping = 0.95
        
        for step in range(n_steps):
            if deadline is not None and time.time() > deadline - 2.0:
                break
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Pairwise repulsion (short-range)
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    d2 = dx*dx + dy*dy
                    d = np.sqrt(d2 + 1e-10)
                    if d < 2.0 * sigma:
                        # Repulsive force
                        force = k_repel * (sigma / d) ** 6 / d
                        fx[i] += force * dx / d
                        fy[i] += force * dy / d
                        fx[j] -= force * dx / d
                        fy[j] -= force * dy / d
            
            # Boundary forces
            for i in range(N):
                if xs[i] < sigma:
                    fx[i] += k_boundary * (sigma - xs[i])
                if xs[i] > w - sigma:
                    fx[i] -= k_boundary * (xs[i] - (w - sigma))
                if ys[i] < sigma:
                    fy[i] += k_boundary * (sigma - ys[i])
                if ys[i] > h - sigma:
                    fy[i] -= k_boundary * (ys[i] - (h - sigma))
            
            # Velocity Verlet
            vx = (vx + fx * dt) * damping
            vy = (vy + fy * dt) * damping
            xs += vx * dt
            ys += vy * dt
            
            # Clamp to bounds
            xs = np.clip(xs, 0.01, w - 0.01)
            ys = np.clip(ys, 0.01, h - 0.01)
        
        return xs, ys

    def optimize_positions(xs0, ys0, w, h, max_iter=200):
        """
        Optimize center positions to maximize sum of LP-solvable radii.
        Uses smooth surrogate: minimize negative soft-min of constraints.
        """
        x0 = np.concatenate([xs0, ys0])
        
        def objective(x):
            xs = x[:N]
            ys = x[N:]
            
            # Soft penalty for being outside bounds
            penalty = 0.0
            for i in range(N):
                if xs[i] < 0:
                    penalty += (xs[i]) ** 2 * 100
                if xs[i] > w:
                    penalty += (xs[i] - w) ** 2 * 100
                if ys[i] < 0:
                    penalty += (ys[i]) ** 2 * 100
                if ys[i] > h:
                    penalty += (ys[i] - h) ** 2 * 100
            
            # Estimate radii via LP
            limits = np.minimum(np.minimum(xs, w - xs), np.minimum(ys, h - ys))
            if np.any(limits < -1e-6):
                return 1e6 + penalty
            
            limits = np.maximum(limits, 0.0)
            
            # Use Delaunay edges for constraint estimation
            edges = delaunay_edges(xs, ys)
            if edges is None:
                return 1e6 + penalty
            
            # Soft constraint: for each edge, r_i + r_j <= d_ij
            # Approximate r_i by limits[i], check if sum exceeds distance
            # Use log-sum-exp for smooth max of violations
            violations = []
            for i, j in edges:
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                # Estimated max sum of radii for this pair
                est_sum = limits[i] + limits[j]
                viol = est_sum - dist
                if viol > 0:
                    violations.append(viol)
            
            if violations:
                # Smooth max via log-sum-exp
                viol_arr = np.array(violations)
                v_max = np.max(viol_arr)
                if v_max > 1e-6:
                    smooth_max = v_max + np.log(np.sum(np.exp(viol_arr - v_max)))
                    penalty += smooth_max ** 2 * 50
            
            # Objective: maximize sum of limits (proxy for sum of radii)
            return -np.sum(limits) + penalty
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B',
                             options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-6})
            xs_opt = result.x[:N]
            ys_opt = result.x[N:]
        except Exception:
            xs_opt = xs0
            ys_opt = ys0
        
        return xs_opt, ys_opt

    def make_result_from_lp(w, h, xs, ys):
        """Build a valid (N,3) result using per-circle LP radii."""
        rs = solve_lp_radii(xs, ys, w, h)
        result = np.zeros((N, 3))
        result[:, 0] = xs
        result[:, 1] = ys
        result[:, 2] = rs
        result[:, 2] = np.maximum(result[:, 2], 0.0)
        
        # Scale to perimeter <= 4
        xs_arr = result[:, 0]
        ys_arr = result[:, 1]
        rs_arr = result[:, 2]
        min_x = np.min(xs_arr - rs_arr)
        max_x = np.max(xs_arr + rs_arr)
        min_y = np.min(ys_arr - rs_arr)
        max_y = np.max(ys_arr + rs_arr)
        bw = max_x - min_x
        bh = max_y - min_y
        perim = 2.0 * (bw + bh)
        if perim > 4.0 + 1e-9 and perim > 0:
            scale = 4.0 / perim
            cx = 0.5 * (min_x + max_x)
            cy = 0.5 * (min_y + max_y)
            for i in range(N):
                result[i, 0] = cx + (result[i, 0] - cx) * scale
                result[i, 1] = cy + (result[i, 1] - cy) * scale
                result[i, 2] = rs_arr[i] * scale
        
        return result

    def refine_slsqp_equal(w, h, xs, ys, r, deadline):
        """
        Equal-radius SLSQP refinement jointly optimizing w and r.
        h = 2 - w.
        """
        if time.time() > deadline - 3.0:
            return w, h, r, xs, ys

        v0 = np.zeros(2 + 2 * N)
        v0[0] = w
        v0[1] = r
        v0[2:2 + N] = xs
        v0[2 + N:2 + 2 * N] = ys

        def objective(v):
            if v[0] <= 0.2 or (2.0 - v[0]) <= 0.2 or v[1] <= 0:
                return 1e6
            return -v[1]

        def make_cons():
            cons = [
                {'type': 'ineq', 'fun': lambda v: v[0] - 0.2},
                {'type': 'ineq', 'fun': lambda v: 2.0 - v[0] - 0.2},
            ]
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2 + i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[0] - v[2 + i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2 + N + i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: (2.0 - v[0]) - v[2 + N + i] - v[1]})
            for i in range(N):
                for j in range(i + 1, N):
                    cons.append({
                        'type': 'ineq',
                        'fun': lambda v, i=i, j=j: np.sqrt((v[2 + i] - v[2 + j]) ** 2
                                                           + (v[2 + N + i] - v[2 + N + j]) ** 2) - 2.0 * v[1]
                    })
            return cons

        bounds = [(0.2, 1.8), (0.001, 1.0)] + [(0.0, 2.0)] * (2 * N)

        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=make_cons(),
                           options={'maxiter': 100, 'ftol': 1e-9, 'disp': False})
        except Exception:
            return w, h, r, xs, ys

        if res.success:
            w_r = res.x[0]
            r_r = res.x[1]
            h_r = 2.0 - w_r
            xs_r = res.x[2:2 + N]
            ys_r = res.x[2 + N:2 + 2 * N]
            if w_r > 0 and h_r > 0 and r_r > 0:
                return w_r, h_r, r_r, xs_r, ys_r
        return w, h, r, xs, ys

    best_value = -np.inf
    best_result = None

    # Generate initial configurations
    configs = []
    
    # Grid configurations
    for aspect in [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.5, 0.5, 2.0, 0.4, 2.5, 0.3, 3.0, 4.0, 0.25]:
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        for cols in range(1, N+1):
            rows = int(np.ceil(N / cols))
            if cols * rows < N:
                continue
            cfg = make_grid(rows, cols, w, h)
            if cfg is not None:
                configs.append(cfg)
    
    # Hexagonal configurations
    hex_rows_list = [
        [7, 7, 7],
        [5, 6, 5, 5],
        [4, 5, 4, 4, 4],
        [3, 4, 3, 4, 3, 4],
        [6, 7, 8],
        [8, 7, 6],
        [5, 5, 5, 6],
        [6, 6, 6, 3],
    ]
    for rows in hex_rows_list:
        if sum(rows) >= N:
            for aspect in [1.0, 0.8, 1.25, 0.7, 1.4]:
                w = 2.0 * aspect / (1.0 + aspect)
                h = 2.0 / (1.0 + aspect)
                cfg = make_hex(rows, w, h)
                if cfg is not None:
                    configs.append(cfg)
    
    # Random configurations
    rng = np.random.RandomState(42)
    for seed in range(5):
        aspect = rng.uniform(0.5, 2.0)
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        cfg = make_random(w, h, seed)
        configs.append(cfg)

    # Process each configuration
    for idx, (xs, ys, w, h) in enumerate(configs):
        if time.time() - start_time > deadline - 5:
            break
        
        # Physics settle
        xs_set, ys_set = physics_settle(xs, ys, w, h, n_steps=300, deadline=deadline)
        
        # LP radii on settled positions
        result = make_result_from_lp(w, h, xs_set, ys_set)
        val = evaluate(result, w, h)
        
        if val > best_value:
            best_value = val
            best_result = result.copy()
        
        # Optimize positions
        xs_opt, ys_opt = optimize_positions(xs_set, ys_set, w, h, max_iter=100)
        result_opt = make_result_from_lp(w, h, xs_opt, ys_opt)
        val_opt = evaluate(result_opt, w, h)
        
        if val_opt > best_value:
            best_value = val_opt
            best_result = result_opt.copy()
        
        # Equal-radius SLSQP refinement on best candidates
        if idx < 20 and time.time() - start_time < deadline - 10:
            # Estimate equal radius
            limits = np.minimum(np.minimum(xs_opt, w - xs_opt), np.minimum(ys_opt, h - ys_opt))
            r_est = np.min(limits)
            if r_est > 0:
                w_r, h_r, r_r, xs_r, ys_r = refine_slsqp_equal(w, h, xs_opt, ys_opt, r_est, deadline)
                result_eq = np.zeros((N, 3))
                result_eq[:, 0] = xs_r
                result_eq[:, 1] = ys_r
                result_eq[:, 2] = r_r
                
                # Scale to perimeter
                xs_arr = result_eq[:, 0]
                ys_arr = result_eq[:, 1]
                rs_arr = result_eq[:, 2]
                min_x = np.min(xs_arr - rs_arr)
                max_x = np.max(xs_arr + rs_arr)
                min_y = np.min(ys_arr - rs_arr)
                max_y = np.max(ys_arr + rs_arr)
                bw = max_x - min_x
                bh = max_y - min_y
                perim = 2.0 * (bw + bh)
                if perim > 4.0 + 1e-9 and perim > 0:
                    scale = 4.0 / perim
                    cx = 0.5 * (min_x + max_x)
                    cy = 0.5 * (min_y + max_y)
                    for i in range(N):
                        result_eq[i, 0] = cx + (result_eq[i, 0] - cx) * scale
                        result_eq[i, 1] = cy + (result_eq[i, 1] - cy) * scale
                        result_eq[i, 2] = rs_arr[i] * scale
                
                val_eq = evaluate(result_eq, w_r, h_r)
                if val_eq > best_value:
                    best_value = val_eq
                    best_result = result_eq.copy()

    # Final polish: if we have a good result, try to refine with tighter tolerance
    if best_result is not None and time.time() - start_time < deadline - 5:
        # Extract w, h from best result bounding box
        xs = best_result[:, 0]
        ys = best_result[:, 1]
        rs = best_result[:, 2]
        min_x = np.min(xs - rs)
        max_x = np.max(xs + rs)
        min_y = np.min(ys - rs)
        max_y = np.max(ys + rs)
        w = max_x - min_x
        h = max_y - min_y
        
        # Try one more LP solve with slightly perturbed positions
        for _ in range(3):
            if time.time() - start_time > deadline - 2:
                break
            perturbation = np.random.randn(N, 2) * 0.01
            xs_p = xs + perturbation[:, 0]
            ys_p = ys + perturbation[:, 1]
            result_p = make_result_from_lp(w, h, xs_p, ys_p)
            val_p = evaluate(result_p, w, h)
            if val_p > best_value:
                best_value = val_p
                best_result = result_p.copy()

    if best_result is None:
        # Fallback: simple grid
        w, h = 1.0, 1.0
        xs, ys, w, h = make_grid(3, 7, w, h)
        best_result = make_result_from_lp(w, h, xs, ys)

    return best_result
