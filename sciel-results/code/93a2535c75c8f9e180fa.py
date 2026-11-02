"""Circle packing in perimeter-4 rectangle: SOCP-style alternating optimization."""
import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 21


def _solve_radii_lp(x, y, W, H):
    """Solve LP: max sum(r) s.t. r_i + r_j <= d_ij, r_i <= wall_dist_i, r_i >= 0"""
    n = N
    c = -np.ones(n)
    
    A_ub_rows = []
    b_ub_vals = []
    
    # Pairwise constraints: r_i + r_j <= d_ij
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(d)
    
    # Wall constraints: r_i <= x_i, W-x_i, y_i, H-y_i
    for i in range(n):
        for bound in [x[i], W - x[i], y[i], H - y[i]]:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(bound)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    bounds = [(0, None)] * n
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    
    # Fallback: greedy estimate
    r = np.zeros(n)
    for i in range(n):
        r_max = min(x[i], W - x[i], y[i], H - y[i])
        for j in range(n):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            r_max = min(r_max, d - r[j])
        r[i] = max(0.0, r_max)
    return r


def _optimize_centers(x0, y0, r, W, H, time_budget):
    """Optimize centers given fixed radii using SLSQP with constraints."""
    n = N
    z0 = np.concatenate([x0, y0])
    
    def objective(z):
        # We want to find a feasible point; use a small regularization
        # to encourage spread. Actually, just return 0 for feasibility.
        # But SLSQP needs some objective. Use sum of squared violations?
        # Better: minimize a small penalty for being too close to boundaries
        x = z[:n]
        y = z[n:2*n]
        # Regularization: encourage centers to be in valid region
        pen = 0.0
        for i in range(n):
            pen += max(0, r[i] - x[i])**2
            pen += max(0, r[i] - (W - x[i]))**2
            pen += max(0, r[i] - y[i])**2
            pen += max(0, r[i] - (H - y[i]))**2
        return pen
    
    constraints = []
    bounds = []
    
    for i in range(n):
        bounds.extend([(r[i], W - r[i]), (r[i], H - r[i])])
    
    for i in range(n):
        for j in range(i + 1, n):
            def make_con(i, j):
                return lambda z: (z[i] - z[n+j])**2 + (z[n+i] - z[j])**2 - (r[i] + r[j])**2
            constraints.append({'type': 'ineq', 'fun': make_con(i, j)})
    
    maxiter = int(min(300, max(50, time_budget * 100)))
    
    try:
        res = minimize(objective, z0, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 50})
        if res.success or res.status == 0:
            return res.x[:n], res.x[n:2*n]
    except Exception:
        pass
    
    return x0, y0


def _pack_for_aspect(W, H, time_budget):
    """Pack N circles in rectangle W x H. Returns (x, y, r, value) or None."""
    n = N
    
    # Initialize with a grid
    side = int(np.ceil(np.sqrt(n)))
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(n):
        row, col = divmod(i, side)
        x[i] = (col + 0.5) / side * W
        y[i] = (row + 0.5) / side * H
    
    # Solve for initial radii
    r = _solve_radii_lp(x, y, W, H)
    
    best_val = np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    # Alternating optimization
    max_rounds = int(min(40, max(5, time_budget * 3)))
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        
        # Step 1: Optimize centers given radii
        t1 = time.time()
        x_new, y_new = _optimize_centers(x, y, r, W, H, time_budget - t1 - 5)
        x, y = x_new, y_new
        
        # Step 2: Solve LP for radii given centers
        t2 = time.time()
        r_new = _solve_radii_lp(x, y, W, H)
        r = r_new
        
        # Update best
        val = np.sum(r)
        if val > best_val:
            best_val = val
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        # Check convergence
        if rnd > 0 and abs(val - best_val) < 1e-8:
            # Try one more round
            pass
    
    # Final: try to scale up radii slightly
    scale = 1.0
    for _ in range(50):
        r_trial = best_r * (1 + scale * 0.001)
        # Check if valid
        valid = True
        for i in range(n):
            if r_trial[i] <= 0:
                valid = False
                break
            if best_x[i] - r_trial[i] < -1e-9 or best_x[i] + r_trial[i] > W + 1e-9:
                valid = False
                break
            if best_y[i] - r_trial[i] < -1e-9 or best_y[i] + r_trial[i] > H + 1e-9:
                valid = False
                break
        if valid:
            for i in range(n):
                for j in range(i+1, n):
                    dx = best_x[i] - best_x[j]
                    dy = best_y[i] - best_y[j]
                    if dx*dx + dy*dy < (r_trial[i] + r_trial[j])**2 - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
        if valid:
            best_r = r_trial
            best_val = np.sum(best_r)
        else:
            break
    
    return best_x, best_y, best_r, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_value = 0.0
    best_result = None
    
    # Try multiple aspect ratios
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.33, 0.7, 1.5, 0.65, 1.6, 0.6, 1.75, 0.55, 1.9, 0.5, 2.0]
    
    for w0 in aspect_ratios:
        if time.time() > deadline - 20:
            break
        h0 = 2.0 - w0
        if h0 < 0.35 or w0 < 0.35:
            continue
        
        remaining = deadline - (time.time() - start_time) - 10
        if remaining < 5:
            break
        
        try:
            x_opt, y_opt, r_opt, val = _pack_for_aspect(w0, h0, time.time() + remaining)
            
            # Validate
            valid = True
            for i in range(N):
                if r_opt[i] <= 0:
                    valid = False
                    break
                if x_opt[i] - r_opt[i] < -1e-7 or x_opt[i] + r_opt[i] > w0 + 1e-7:
                    valid = False
                    break
                if y_opt[i] - r_opt[i] < -1e-7 or y_opt[i] + r_opt[i] > h0 + 1e-7:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x_opt[i] - x_opt[j]
                        dy = y_opt[i] - y_opt[j]
                        if dx*dx + dy*dy < (r_opt[i] + r_opt[j])**2 - 1e-7:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid and val > best_value:
                best_value = val
                best_result = np.column_stack((x_opt, y_opt, r_opt))
        except Exception:
            continue
    
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row, col = divmod(i, side)
            cx = (col + 0.5) / side
            cy = (row + 0.5) / side
            r0 = 0.5 / side - 1e-3
            cells.append([cx, cy, r0])
        return np.array(cells)
    
    return best_result
