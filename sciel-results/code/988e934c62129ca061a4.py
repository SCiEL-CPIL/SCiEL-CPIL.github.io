import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

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

    def sum_radii(r):
        return float(np.sum(r))

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
                r_cell = min(cell_w, cell_h) / 2.0 * 0.95

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

    def force_relax(x, y, r, w, max_iter=500, tol=1e-9):
        h = 2.0 - w
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        dt = 0.005
        min_dt = 0.0001

        for it in range(max_iter):
            fx = np.zeros(N)
            fy = np.zeros(N)

            for i in range(N):
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i]) * 100
                if x[i] > w - r[i]:
                    fx[i] -= (x[i] - (w - r[i])) * 100
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i]) * 100
                if y[i] > h - r[i]:
                    fy[i] -= (y[i] - (h - r[i])) * 100

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
                        f = overlap * 20.0
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
            else:
                dt = min(0.01, dt * 1.1)

            x += dt * fx
            y += dt * fy

            for i in range(N):
                x[i] = np.clip(x[i], r[i], w - r[i])
                y[i] = np.clip(y[i], r[i], h - r[i])

        return x, y

    def voronoi_max_radius(x, y, w, h, indices=None):
        if indices is None:
            indices = list(range(N))
        
        if len(indices) == 0:
            return np.zeros(0)
        
        # Add boundary points to handle edges correctly
        # Create a grid of boundary points
        n_bnd = 10
        bnd_x = np.concatenate([
            np.linspace(0, w, n_bnd),
            np.linspace(0, w, n_bnd),
            np.linspace(0, h, n_bnd),
            np.linspace(0, h, n_bnd)
        ])
        bnd_y = np.concatenate([
            np.zeros(n_bnd),
            np.full(n_bnd, h),
            np.zeros(n_bnd),
            np.full(n_bnd, h)
        ])
        
        # Shift boundary points slightly outside to ensure they are "outside" the domain
        # Actually, for Voronoi, we just need to make sure the region is bounded.
        # A better way: add points far away.
        # But standard Voronoi in scipy works on the convex hull.
        # To handle walls, we can reflect or just add points at -large and +large.
        
        # Let's use a simpler approach: 
        # The max radius for circle i is half the distance to the nearest other circle or wall.
        # We can compute this directly without Voronoi if N is small, but Voronoi is O(N log N).
        # For N=21, direct O(N^2) is fine and more robust to numerical issues.
        
        radii = np.zeros(len(indices))
        for k, i in enumerate(indices):
            min_dist = np.inf
            # Walls
            min_dist = min(min_dist, x[i], w - x[i], y[i], h - y[i])
            
            # Other circles
            for j in range(N):
                if j == i:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                # Distance to edge of circle j
                min_dist = min(min_dist, d - 0) # We want distance to center, then subtract r[j] later?
                # Actually, if we want to grow r[i] keeping others fixed:
                # Constraint: dist(i,j) >= r[i] + r[j]
                # So r[i] <= dist(i,j) - r[j]
                # And r[i] <= dist_to_wall(i)
                
                # Wait, the "Voronoi" idea usually implies growing all simultaneously or finding the "space" available.
                # If we want to maximize sum of radii, we can't just take min(dist - r_j) for each independently because they might conflict.
                # However, a common heuristic is to set r[i] = 0.5 * min(dist_to_nearest_neighbor_or_wall).
                # Let's stick to the safe independent bound for initialization or local improvement.
                
                pass
            
            # Let's do the proper calculation:
            # Max possible r[i] if others are fixed is:
            # min( x[i], w-x[i], y[i], h-y[i], min_{j!=i} (dist(i,j) - r[j]) )
            
            min_wall = min(x[i], w - x[i], y[i], h - y[i])
            min_circ = np.inf
            for j in range(N):
                if j == i:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_circ = min(min_circ, d - 0) # This is distance to center. We need d - r[j].
                # But we don't have r[j] here in this function signature if we want it generic.
                # Let's pass r as well.
            
            # Revised: This function should take r as well to be useful for growing.
            # But for now, let's just use it to estimate a safe radius for a new circle or to reset.
            # For the main algorithm, we will use a direct "grow" loop.
            
            radii[k] = min_wall * 0.9 # Conservative
        return radii

    def nlp_objective(z, w, h, r_init):
        # z contains [x_0...x_N, y_0...y_N, r_0...r_N]
        # We want to maximize sum(r) -> minimize -sum(r)
        # Constraints:
        # 1. x_i - r_i >= 0
        # 2. w - x_i - r_i >= 0
        # 3. y_i - r_i >= 0
        # 4. h - y_i - r_i >= 0
        # 5. dist(i,j) - r_i - r_j >= 0
        
        x = z[:N]
        y = z[N:2*N]
        r = z[2*N:3*N]
        
        obj = -np.sum(r)
        
        # Constraints function returns array of values that should be >= 0
        cons = []
        
        for i in range(N):
            cons.append(x[i] - r[i])
            cons.append(w - x[i] - r[i])
            cons.append(y[i] - r[i])
            cons.append(h - y[i] - r[i])
            
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                cons.append(d - r[i] - r[j])
                
        return obj, np.array(cons)

    def nlp_jacobian(z, w, h):
        x = z[:N]
        y = z[N:2*N]
        r = z[2*N:3*N]
        
        # Objective gradient: [0...0, 0...0, -1...-1]
        grad_obj = np.zeros(3*N)
        grad_obj[2*N:3*N] = -1.0
        
        # Constraint Jacobian
        # Number of constraints: 4*N + N*(N-1)/2
        n_cons = 4*N + N*(N-1)/2
        jac = np.zeros((n_cons, 3*N))
        
        idx = 0
        for i in range(N):
            # x_i - r_i >= 0
            jac[idx, i] = 1.0
            jac[idx, 2*N+i] = -1.0
            idx += 1
            # w - x_i - r_i >= 0
            jac[idx, i] = -1.0
            jac[idx, 2*N+i] = -1.0
            idx += 1
            # y_i - r_i >= 0
            jac[idx, N+i] = 1.0
            jac[idx, 2*N+i] = -1.0
            idx += 1
            # h - y_i - r_i >= 0
            jac[idx, N+i] = -1.0
            jac[idx, 2*N+i] = -1.0
            idx += 1
            
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy) + 1e-12
                ux = dx / d
                uy = dy / d
                # dist(i,j) - r_i - r_j >= 0
                jac[idx, i] = ux
                jac[idx, N+j] = -uy # Wait, derivative of dist wrt x_j is -ux
                # d/dx_j (sqrt((x_i-x_j)^2 + ...)) = (x_i-x_j)/dist * (-1) = -ux
                # So jac[idx, j] = -ux
                jac[idx, j] = -ux
                jac[idx, N+i] = uy
                jac[idx, N+j] = -uy
                jac[idx, 2*N+i] = -1.0
                jac[idx, 2*N+j] = -1.0
                idx += 1
                
        return grad_obj, jac

    def run_nlp(x0, y0, r0, w, h, max_iter=100):
        z0 = np.concatenate([x0, y0, r0])
        
        # Bounds:
        # x in [0, w], y in [0, h], r in [0, min(w,h)/2]
        bounds = []
        for i in range(N):
            bounds.append((0.0, w))
        for i in range(N):
            bounds.append((0.0, h))
        r_max = min(w, h) / 2.0
        for i in range(N):
            bounds.append((0.0, r_max))
            
        # Constraints
        # We will define them as functions for scipy
        def cons_func(z):
            x = z[:N]
            y = z[N:2*N]
            r = z[2*N:3*N]
            vals = []
            for i in range(N):
                vals.append(x[i] - r[i])
                vals.append(w - x[i] - r[i])
                vals.append(y[i] - r[i])
                vals.append(h - y[i] - r[i])
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    vals.append(d - r[i] - r[j])
            return np.array(vals)
            
        # This is slow, but trust-constr uses numerical diff if not provided?
        # No, better to provide analytical Jacobian.
        
        res = minimize(
            lambda z: nlp_objective(z, w, h, r0)[0],
            z0,
            method='trust-constr',
            jac=lambda z: nlp_jacobian(z, w, h)[0],
            bounds=bounds,
            constraints=[
                {
                    'type': 'ineq',
                    'fun': cons_func,
                    # 'jac': lambda z: nlp_jacobian(z, w, h)[1] # This is expensive to construct every time
                }
            ],
            options={'maxiter': max_iter, 'gtol': 1e-5}
        )
        
        if res.success or res.fun < 1e-6:
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            return x_opt, y_opt, r_opt
        return None

    best_value = 0.0
    best_result = None
    rng = np.random.default_rng(42)

    # Phase 1: Initialization and Force Relax
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.7, 0.55, 2.0, 0.5, 1.1, 0.9, 1.25, 0.8, 1.4, 0.7, 1.6, 0.65]

    for w0 in aspect_ratios:
        if time.time() > deadline - 50:
            break
        h0 = 2.0 - w0
        if h0 < 0.35:
            continue

        x0, y0, r0 = grid_init(w0)
        x_t, y_t = force_relax(x0, y0, r0, w0, max_iter=300)
        if is_valid(x_t, y_t, r0, w0):
            val = sum_radii(r0)
            if val > best_value:
                best_value = val
                best_result = (x_t.copy(), y_t.copy(), r0.copy(), w0)

    if best_result is None:
        # Fallback
        w0 = 1.0
        h0 = 1.0
        cols = 5
        rows = 5
        cell_w = w0 / cols
        cell_h = h0 / rows
        r0 = min(cell_w, cell_h) / 2.0 * 0.45
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.full(N, r0)
        for i in range(N):
            row, col = divmod(i, cols)
            x[i] = (col + 0.5) * cell_w
            y[i] = (row + 0.5) * cell_h
        return np.column_stack((x, y, r))

    # Phase 2: NLP Refinement
    if best_result is not None and time.time() < deadline - 40:
        x_cur, y_cur, r_cur, w_cur = best_result
        h_cur = 2.0 - w_cur
        
        # Try NLP on the current best
        res_nlp = run_nlp(x_cur, y_cur, r_cur, w_cur, h_cur, max_iter=50)
        if res_nlp is not None:
            x_nlp, y_nlp, r_nlp = res_nlp
            if is_valid(x_nlp, y_nlp, r_nlp, w_cur):
                val_nlp = sum_radii(r_nlp)
                if val_nlp > best_value:
                    best_value = val_nlp
                    best_result = (x_nlp.copy(), y_nlp.copy(), r_nlp.copy(), w_cur)

    # Phase 3: Iterative Growth with NLP
    if best_result is not None:
        x_cur, y_cur, r_cur, w_cur = best_result
        h_cur = 2.0 - w_cur
        
        # Attempt to grow radii by solving NLP with increased lower bounds on r?
        # Or just run NLP again with higher precision?
        
        # Alternative: "Jiggle" and re-optimize
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            
            # Perturb
            scale = 0.005
            x_pert = x_cur + rng.normal(0, scale, N)
            y_pert = y_cur + rng.normal(0, scale, N)
            x_pert = np.clip(x_pert, r_cur, w_cur - r_cur)
            y_pert = np.clip(y_pert, r_cur, h_cur - r_cur)
            
            x_rel, y_rel = force_relax(x_pert, y_pert, r_cur, w_cur, max_iter=200)
            if not is_valid(x_rel, y_rel, r_cur, w_cur):
                continue
                
            res_nlp = run_nlp(x_rel, y_rel, r_cur, w_cur, h_cur, max_iter=50)
            if res_nlp is not None:
                x_new, y_new, r_new = res_nlp
                if is_valid(x_new, y_new, r_new, w_cur):
                    val_new = sum_radii(r_new)
                    if val_new > best_value:
                        best_value = val_new
                        best_result = (x_new.copy(), y_new.copy(), r_new.copy(), w_cur)
                        x_cur, y_cur, r_cur = x_new, y_new, r_new

    # Phase 4: Aspect Ratio Optimization
    if best_result is not None and time.time() < deadline - 20:
        x_cur, y_cur, r_cur, w_cur = best_result
        
        # Try a few nearby aspect ratios
        for dw in [-0.1, -0.05, 0.05, 0.1]:
            if time.time() > deadline - 10:
                break
            w_try = w_cur + dw
            h_try = 2.0 - w_try
            if w_try < 0.4 or h_try < 0.4:
                continue
            
            # Scale positions to new aspect ratio?
            # Simple: just take current positions and relax in new box
            x_try = x_cur.copy()
            y_try = y_cur.copy()
            r_try = r_cur.copy()
            
            # Clip to new bounds
            for i in range(N):
                x_try[i] = np.clip(x_try[i], r_try[i], w_try - r_try[i])
                y_try[i] = np.clip(y_try[i], r_try[i], h_try - r_try[i])
                
            x_rel, y_rel = force_relax(x_try, y_try, r_try, w_try, max_iter=200)
            if is_valid(x_rel, y_rel, r_try, w_try):
                res_nlp = run_nlp(x_rel, y_rel, r_try, w_try, h_try, max_iter=50)
                if res_nlp is not None:
                    x_new, y_new, r_new = res_nlp
                    if is_valid(x_new, y_new, r_new, w_try):
                        val_new = sum_radii(r_new)
                        if val_new > best_value:
                            best_value = val_new
                            best_result = (x_new.copy(), y_new.copy(), r_new.copy(), w_try)

    # Final Polish: Force relax to ensure validity
    if best_result is not None:
        x_final, y_final, r_final, w_final = best_result
        x_final, y_final = force_relax(x_final, y_final, r_final, w_final, max_iter=100)
        
        # Safety shrink if still invalid (shouldn't happen)
        if not is_valid(x_final, y_final, r_final, w_final):
            for _ in range(10):
                r_final = r_final * 0.99
                if is_valid(x_final, y_final, r_final, w_final):
                    break

    return np.column_stack((x_final, y_final, r_final))

if __name__ == "__main__":
    res = construct()
    print(res.shape)
