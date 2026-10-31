"""Circle packing in perimeter-4 rectangle: SLSQP + High-Precision Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
_IU = np.triu_indices(N, k=1)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    def get_bounds(w, h):
        """Calculate bounds for each circle's center and radius."""
        # z = [x0, y0, r0, ..., x20, y20, r20, w]
        bounds = []
        for i in range(N):
            # x_i in [0, w], y_i in [0, h], r_i in [0, min(w,h)/2]
            # We will refine these in the objective/constraints based on current w
            # But SLSQP needs fixed bounds per run. We will run SLSQP with w fixed 
            # or include w in optimization? 
            # Including w makes it a non-convex problem. 
            # Strategy: Optimize w in a coarse loop, then optimize circles for fixed w.
            # Or optimize all together with loose bounds.
            # Let's optimize all together. w in [0.5, 1.5] (since perimeter=4, w+h=2, w=1 is square)
            # If w is too small, h is large, and vice versa.
            # To keep bounds valid, we must ensure r_i <= min(x_i, w-x_i, y_i, h-y_i).
            # This is handled by constraints. Bounds can be loose.
            bounds.append((0.0, 2.0)) # x
            bounds.append((0.0, 2.0)) # y
            bounds.append((0.0, 1.0)) # r
        bounds.append((0.4, 1.6)) # w
        return bounds

    def objective(z):
        """Maximize sum of radii (minimize negative sum)."""
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        # Penalize invalid w/h slightly to guide optimizer
        if w < 0.3 or h < 0.3:
            return 1e6
        return -np.sum(r)

    def constraints(z):
        """Define inequality constraints: g(z) >= 0."""
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        
        cons = []
        
        # Boundary constraints:
        # x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda z, x=x, r=r: x - r})
        # w - x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda z, x=x, r=r, w=w: w - x - r})
        # y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda z, y=y, r=r: y - r})
        # h - y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda z, y=y, r=r, h=h: h - y - r})
        
        # Pairwise constraints:
        # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        # This is non-smooth at contact. SLSQP handles it okay but might struggle.
        # We can use the smooth version: d - (r_i + r_j) >= 0
        dx = x[_IU[0]] - x[_IU[1]]
        dy = y[_IU[0]] - y[_IU[1]]
        d = np.sqrt(dx*dx + dy*dy + 1e-12)
        r_sum = r[_IU[0]] + r[_IU[1]]
        cons.append({'type': 'ineq', 'fun': lambda z, d=d, r_sum=r_sum: d - r_sum})
        
        # Radii must be positive
        cons.append({'type': 'ineq', 'fun': lambda z, r=r: r})
        
        # w and h must be positive
        cons.append({'type': 'ineq', 'fun': lambda z, w=w: w})
        cons.append({'type': 'ineq', 'fun': lambda z, h=h: h})
        
        return cons

    def run_slsqp(z0, time_budget):
        """Run SLSQP optimization."""
        bounds = get_bounds(z0[-1], 2.0-z0[-1])
        
        # SLSQP can be slow. Use maxiter and time check.
        # We cannot easily pass a time limit to minimize, so we rely on maxiter 
        # and checking time in the outer loop.
        # However, SLSQP can hang. We'll set a reasonable maxiter.
        
        try:
            res = minimize(
                objective, 
                z0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
            )
            return res.x, res.fun
        except Exception as e:
            return z0, objective(z0)

    def polish_slack(z, time_budget):
        """
        Iterative polish: 
        1. Check current validity.
        2. For each circle, try to increase radius slightly.
        3. Run a short SLSQP to adjust positions to accommodate the larger radius.
        4. Repeat.
        """
        z = z.copy()
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        
        # Ensure we start valid
        if not is_valid(z):
            return z, 0.0
            
        best_val = np.sum(r)
        end_time = time.time() + time_budget
        
        # Adaptive step size for radius expansion
        step_r = 0.005
        
        while time.time() < end_time:
            improved = False
            # Try expanding each circle
            for i in range(N):
                if time.time() > end_time:
                    break
                    
                # Calculate max possible radius based on current position
                # limited by walls
                r_wall = min(x[i], w - x[i], y[i], h - y[i])
                
                # limited by neighbors
                min_dist = np.inf
                for j in range(N):
                    if i == j: continue
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    min_dist = min(min_dist, d - r[j])
                
                r_max = min(r_wall, min_dist)
                
                if r_max > r[i] + 1e-6:
                    # We can expand. Let's try a small step
                    new_r = min(r_max, r[i] + step_r)
                    
                    # Create new z
                    z_new = z.copy()
                    z_new[i*3 + 2] = new_r
                    
                    # Run a very short SLSQP to adjust positions to make it valid
                    # We only need to move positions, not w or other r's ideally, 
                    # but SLSQP will move what it needs to.
                    # To save time, we can fix w and other r's? 
                    # SLSQP doesn't support fixing variables easily without bounds.
                    # We'll just run a quick SLSQP with tight bounds on others.
                    
                    bounds_tight = get_bounds(w, h)
                    # Tighten bounds for other radii to prevent them from shrinking too much
                    for j in range(N):
                        if j != i:
                            # Allow small shrinkage? No, we want to keep them.
                            # Actually, if we expand i, we might need to move j.
                            # We don't want j to shrink. 
                            # SLSQP minimizes -sum(r). If we fix others, it might fail.
                            # Let's just run standard SLSQP but with a very low maxiter.
                            pass
                    
                    try:
                        res = minimize(
                            objective, 
                            z_new, 
                            method='SLSQP', 
                            bounds=bounds_tight, 
                            constraints=constraints,
                            options={'maxiter': 20, 'ftol': 1e-8, 'disp': False}
                        )
                        
                        if res.success and is_valid(res.x):
                            val = np.sum(res.x[2::3][:N])
                            if val > best_val:
                                z = res.x
                                best_val = val
                                improved = True
                                # Update local views
                                x = z[0::3][:N]
                                y = z[1::3][:N]
                                r = z[2::3][:N]
                    except:
                        pass
            
            if not improved:
                # No single circle could be expanded easily.
                # Maybe we need to adjust positions globally to create space?
                # Run a standard SLSQP to "settle" the current configuration
                try:
                    bounds_tight = get_bounds(w, h)
                    res = minimize(
                        objective, 
                        z, 
                        method='SLSQP', 
                        bounds=bounds_tight, 
                        constraints=constraints,
                        options={'maxiter': 50, 'ftol': 1e-9, 'disp': False}
                    )
                    if res.success and is_valid(res.x):
                        val = np.sum(res.x[2::3][:N])
                        if val > best_val:
                            z = res.x
                            best_val = val
                            improved = True
                            x = z[0::3][:N]
                            y = z[1::3][:N]
                            r = z[2::3][:N]
                except:
                    pass
                    
            if not improved:
                break
                
        return z, best_val

    def is_valid(z, tol=1e-7):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        
        if w <= 0.1 or h <= 0.1: return False
        if np.any(r < -tol): return False
        if np.any(x < r - tol) or np.any(x > w - r + tol): return False
        if np.any(y < r - tol) or np.any(y > h - r + tol): return False
        
        dx = x[_IU[0]] - x[_IU[1]]
        dy = y[_IU[0]] - y[_IU[1]]
        d2 = dx*dx + dy*dy
        r_sum = r[_IU[0]] + r[_IU[1]]
        if np.any(d2 < r_sum*r_sum - tol): return False
        
        return True

    # Generate initial guesses
    def get_initial_guesses():
        guesses = []
        rng = np.random.default_rng(123)
        
        # 1. Grid-based starts with various aspect ratios
        aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.4: continue
            
            z = np.zeros(3 * N + 1)
            z[-1] = w0
            
            # Place circles in a grid
            # 21 circles. 5x5 grid = 25. Skip 4 corners.
            positions = []
            for row in range(5):
                for col in range(5):
                    if (row==0 and col==0) or (row==4 and col==4) or (row==0 and col==4) or (row==4 and col==0):
                        continue
                    positions.append((row, col))
            
            for i in range(N):
                row, col = positions[i]
                cx = (col + 0.5) / 5.0 * w0
                cy = (row + 0.5) / 5.0 * h0
                r0 = min(w0/5.0, h0/5.0) * 0.45
                z[i*3] = cx
                z[i*3+1] = cy
                z[i*3+2] = r0
            guesses.append(z)
            
        # 2. Random starts
        for _ in range(10):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            z[-1] = w0
            for i in range(N):
                r0 = rng.uniform(0.03, 0.08)
                z[i*3] = rng.uniform(r0, w0 - r0)
                z[i*3+1] = rng.uniform(r0, h0 - r0)
                z[i*3+2] = r0
            guesses.append(z)
            
        return guesses

    guesses = get_initial_guesses()
    
    # Sort by potential (square is usually best for dense packing, but rectangle can be better)
    # Let's just iterate through them.
    
    for gi, z0 in enumerate(guesses):
        if time.time() > deadline - 20:
            break
            
        # Phase 1: SLSQP Optimization
        time_budget_slsqp = min(20, deadline - (time.time() - start_time) - 15)
        if time_budget_slsqp < 5:
            continue
            
        z_opt, _ = run_slsqp(z0, time_budget_slsqp)
        
        # Phase 2: Polish
        time_budget_polish = min(10, deadline - (time.time() - start_time) - 5)
        if time_budget_polish < 2:
            continue
            
        z_polished, val = polish_slack(z_opt, time_budget_polish)
        
        if val > best_value:
            best_value = val
            best_result = z_polished.copy()

    # Final check and return
    if best_result is None:
        # Fallback
        z = np.zeros(3*N + 1)
        z[-1] = 1.0
        for i in range(N):
            # Simple grid
            row = i // 5
            col = i % 5
            if row == 0 and col == 0: continue # Skip logic is messy here, just place all in 5x5
            # Actually, just place them in a 5x5 grid, if overlap, SLSQP will fix? 
            # No, fallback must be valid.
            # Let's use the first guess from get_initial_guesses if it was valid
            pass
        
        # Use a safe fallback: 5x5 grid in 1x1 square, r=0.1
        # 21 circles in 1x1.
        # 5x5 grid cells are 0.2x0.2. r=0.1 fits perfectly if centered.
        # But we only have 21.
        for i in range(N):
            row = i // 5
            col = i % 5
            z[i*3] = (col + 0.5) * 0.2
            z[i*3+1] = (row + 0.5) * 0.2
            z[i*3+2] = 0.1
        best_result = z
        best_value = 0.1 * N

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    
    return np.column_stack((x, y, r))
