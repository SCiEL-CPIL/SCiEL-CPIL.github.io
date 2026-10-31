"""Circle packing in perimeter-4 rectangle: Advanced Force-Directed Optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    best_value = 0.0
    best_result = None

    def check_validity(x, y, r, w, h, tol=1e-9):
        if w <= 0.05 or h <= 0.05:
            return False, 0.0
        if np.any(r <= 0):
            return False, 0.0
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False, 0.0
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False, 0.0
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d2 = dx * dx + dy * dy
        rs = r[:, None] + r[None, :]
        iu = np.triu_indices(N, k=1)
        if np.any(d2[iu] < rs[iu] ** 2 - tol):
            return False, 0.0
        return True, float(np.sum(r))

    def objective(params, w, h, inflate_factor):
        """
        Objective function: Minimize negative sum of radii + penalty for violations.
        params: [x1, y1, r1, x2, y2, r2, ...]
        """
        x = params[0::3]
        y = params[1::3]
        r = params[2::3] * inflate_factor
        
        penalty = 0.0
        # Wall constraints
        # If x < r, penalty is (r - x)^2
        # If x > w - r, penalty is (x - (w - r))^2
        # We use a soft barrier
        wall_pen = 0.0
        # Left wall
        d_left = x - r
        wall_pen += np.sum(np.maximum(-d_left, 0.0) ** 2)
        # Right wall
        d_right = (w - r) - x
        wall_pen += np.sum(np.maximum(-d_right, 0.0) ** 2)
        # Bottom wall
        d_bot = y - r
        wall_pen += np.sum(np.maximum(-d_bot, 0.0) ** 2)
        # Top wall
        d_top = (h - r) - y
        wall_pen += np.sum(np.maximum(-d_top, 0.0) ** 2)
        
        # Pairwise constraints
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d2 = dx * dx + dy * dy
        rs = r[:, None] + r[None, :]
        iu = np.triu_indices(N, k=1)
        d = np.sqrt(np.maximum(d2[iu], 1e-12))
        overlap = rs[iu] - d
        pair_pen = np.sum(np.maximum(overlap, 0.0) ** 2)
        
        total_penalty = wall_pen + pair_pen
        sum_r = np.sum(r)
        
        # We want to maximize sum_r, so minimize -sum_r + penalty
        return -sum_r + 100.0 * total_penalty

    def simulate(w0, h0, time_budget, seed=0):
        rng = np.random.default_rng(seed)
        
        # Grid initialization
        # Estimate grid size
        side = int(np.ceil(np.sqrt(N)))
        # Create a grid of N points
        xs = []
        ys = []
        count = 0
        # Fill a grid of side x side, take first N
        # To ensure they are spread out, we use a jittered grid
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                cx = (i + 0.5) * w0 / side
                cy = (j + 0.5) * h0 / side
                # Add jitter
                cx += rng.uniform(-0.1, 0.1)
                cy += rng.uniform(-0.1, 0.1)
                xs.append(cx)
                ys.append(cy)
                count += 1
        
        x = np.array(xs)
        y = np.array(ys)
        
        # Initial radii
        # Estimate max possible radius
        r_init = 0.04
        r = np.full(N, r_init)
        
        # Parameters vector
        params = np.zeros(3 * N)
        params[0::3] = x
        params[1::3] = y
        params[2::3] = r
        
        # Optimization using L-BFGS-B
        # We will run this in phases with increasing inflation factors
        
        best_params = params.copy()
        best_val = 0.0
        
        # Phase 1: Relax positions with small radii
        # Phase 2: Inflate and relax
        
        # We'll do a manual loop of optimization steps
        # To save time, we'll use a few restarts with different jitter seeds
        
        for inflate_factor in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
            # Define objective for this inflate factor
            def obj(p):
                return objective(p, w0, h0, inflate_factor)
            
            # Use minimize
            try:
                res = minimize(obj, best_params, method='L-BFGS-B', 
                               options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-4})
                
                # Extract result
                p_opt = res.x
                x_opt = p_opt[0::3]
                y_opt = p_opt[1::3]
                r_opt = p_opt[2::3] * inflate_factor
                
                # Check validity
                valid, val = check_validity(x_opt, y_opt, r_opt, w0, h0)
                if valid and val > best_val:
                    best_val = val
                    best_params = p_opt.copy()
                    # Update best_params to be the base radii for next inflation?
                    # Actually, the optimizer optimizes the base radii in params.
                    # The inflate_factor is applied in the objective.
                    # So best_params holds the base radii.
                    # We should keep the base radii from the last valid solution.
                    
            except Exception:
                pass
            
            if time.time() > start_time + time_budget:
                break

        # Final extraction
        x_final = best_params[0::3]
        y_final = best_params[1::3]
        r_final = best_params[2::3] * 1.5 # Use the last inflate factor for final check?
        # Wait, the objective uses inflate_factor. The params are the "base" radii.
        # The actual radii in the solution are params[2::3] * inflate_factor.
        # But the validity check needs the actual radii.
        # Let's re-evaluate the best valid configuration found during the loop.
        
        # Actually, the loop above updates best_params which are the base radii.
        # But the "value" is calculated with the specific inflate_factor.
        # So we need to track the best (x, y, r_actual) tuple.
        
        # Let's redo the loop logic to track actual radii.
        
        return best_params[0::3], best_params[1::3], best_params[2::3]

    # The above simulate function is a bit messy with the inflation tracking.
    # Let's write a cleaner simulate function.
    
    def simulate_clean(w0, h0, time_budget, seed=0):
        rng = np.random.default_rng(seed)
        
        # Grid initialization
        side = int(np.ceil(np.sqrt(N)))
        xs = []
        ys = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                cx = (i + 0.5) * w0 / side
                cy = (j + 0.5) * h0 / side
                cx += rng.uniform(-0.05, 0.05)
                cy += rng.uniform(-0.05, 0.05)
                xs.append(cx)
                ys.append(cy)
                count += 1
        
        x = np.array(xs)
        y = np.array(ys)
        r = np.full(N, 0.04)
        
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        best_val = 0.0
        
        # Optimization loop
        # We optimize positions and radii simultaneously.
        # Variables: x, y, r (all free, constrained by validity/penalty)
        
        def obj_full(params):
            px = params[0:N]
            py = params[N:2*N]
            pr = params[2*N:3*N]
            
            # Penalties
            pen = 0.0
            # Walls
            pen += np.sum(np.maximum(pr - px, 0.0)**2)
            pen += np.sum(np.maximum(px - (w0 - pr), 0.0)**2)
            pen += np.sum(np.maximum(pr - py, 0.0)**2)
            pen += np.sum(np.maximum(py - (h0 - pr), 0.0)**2)
            
            # Pairs
            dx = px[:, None] - px[None, :]
            dy = py[:, None] - py[None, :]
            d2 = dx*dx + dy*dy
            rs = pr[:, None] + pr[None, :]
            iu = np.triu_indices(N, k=1)
            d = np.sqrt(np.maximum(d2[iu], 1e-12))
            overlap = rs[iu] - d
            pen += np.sum(np.maximum(overlap, 0.0)**2)
            
            sum_r = np.sum(pr)
            return -sum_r + 50.0 * pen

        params = np.concatenate([x, y, r])
        
        # Run L-BFGS-B
        # We do this in a few stages to help convergence
        for stage in range(3):
            try:
                res = minimize(obj_full, params, method='L-BFGS-B', 
                               options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-6})
                params = res.x
            except Exception:
                pass
            
            if time.time() > start_time + time_budget:
                break
        
        # Extract
        x_opt = params[0:N]
        y_opt = params[N:2*N]
        r_opt = params[2*N:3*N]
        
        valid, val = check_validity(x_opt, y_opt, r_opt, w0, h0)
        if valid:
            return x_opt, y_opt, r_opt
        else:
            return None

    # Sweep aspect ratios
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.33, 0.7, 1.5, 0.65, 1.6, 0.6, 1.75, 0.55]
    
    for k, w0 in enumerate(aspect_ratios):
        h0 = 2.0 - w0
        if h0 < 0.4 or w0 < 0.4:
            continue
            
        remaining = deadline - (time.time() - start_time)
        if remaining < 5.0:
            break
        
        budget = min(8.0, remaining - 1.0)
        
        # Try a few seeds
        for seed in [42 + k, 123 + k, 999 + k]:
            if time.time() > deadline - 2.0:
                break
            
            try:
                res = simulate_clean(w0, h0, budget, seed)
                if res is not None:
                    x, y, r = res
                    valid, val = check_validity(x, y, r, w0, h0)
                    if valid and val > best_value:
                        best_value = val
                        best_result = (x.copy(), y.copy(), r.copy(), w0, h0)
            except Exception:
                continue

    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, w, h = best_result
    
    # Final safety shrink
    for _ in range(8):
        rt = r * (1 - 1e-9)
        if check_validity(x, y, rt, w, h)[0]:
            r = rt
        else:
            break

    return np.column_stack((x, y, r))
