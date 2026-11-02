import numpy as np
from scipy.optimize import minimize

N = 21

def construct():
    # Better initial configuration: Hexagonal lattice
    # We want to fit 21 circles. 
    # A 5x5 grid has 25 spots. We can take 21 from a 5x5 hex grid.
    # Or 4x5 = 20, plus 1. 
    # Let's try a 5x5 grid of hexagonal points, remove 4 corners or similar.
    
    def get_initial_config():
        # Hexagonal lattice
        # Rows of circles. 
        # Let's try to fit into a 1x1 square initially, then scale.
        # 5 columns, 5 rows? 
        # Hex spacing: dx = 2r, dy = sqrt(3)r
        # Let's just use a standard grid with jitter, it's safer for 21.
        # 5x5 grid = 25 points. We need 21.
        # Let's use 5 columns. 21/5 = 4.2. 
        # Rows: 5, 5, 5, 4, 2? 
        # Simpler: Random points in a square, then push apart.
        
        # Let's stick to a structured grid but optimize the aspect ratio.
        # 5x5 grid, remove 4 random points? 
        # Or 4x5 grid (20 points) + 1?
        
        # Let's use a 5x5 grid, keep the central 21? No, just first 21.
        side = 5
        points = []
        for i in range(side):
            for j in range(side):
                # Hexagonal offset
                x = i + (j % 2) * 0.5
                y = j * np.sqrt(3) / 2
                points.append((x, y))
        
        # We have 25 points. We need 21.
        # Sort by distance from center? Or just take first 21?
        # Taking first 21 from a 5x5 hex grid leaves a rectangular-ish shape.
        # Let's take the 21 points closest to the center of the 5x5 hex grid.
        center_x = (side - 1) / 2.0
        center_y = (side - 1) * np.sqrt(3) / 4.0
        points.sort(key=lambda p: (p[0]-center_x)**2 + (p[1]-center_y)**2)
        points = points[:N]
        
        # Normalize to fit in 1x1
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Scale to fit in 1x1 with margin
        w = max_x - min_x
        h = max_y - min_y
        scale = 1.0 / max(w, h)
        
        x = np.array([(p[0] - min_x) * scale for p in points])
        y = np.array([(p[1] - min_y) * scale for p in points])
        
        # Initial radius: estimate from min distance
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                min_dist = min(min_dist, d)
        r = np.full(N, min_dist * 0.4) # 0.4 to be safe
        
        # Rectangle dimensions
        # Perimeter 4 => W+H=2.
        # If we scaled to 1x1, W=1, H=1.
        W = 1.0
        H = 1.0
        
        return x, y, r, W, H

    def is_valid(x, y, r, W, H):
        if W <= 0 or H <= 0:
            return False
        if abs(2*(W+H) - 4) > 1e-5:
            return False
        for i in range(N):
            if r[i] <= 1e-9:
                return False
            if x[i] - r[i] < -1e-6 or x[i] + r[i] > W + 1e-6:
                return False
            if y[i] - r[i] < -1e-6 or y[i] + r[i] > H + 1e-6:
                return False
            for j in range(i+1, N):
                dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                rad_sum = r[i] + r[j]
                if dist_sq < rad_sum**2 - 1e-8:
                    return False
        return True

    def penalty_obj(v, lam):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        W = v[3*N]
        H = v[3*N+1]
        
        r_pos = np.maximum(r, 0)
        
        obj = -np.sum(r_pos)
        penalty = 0
        
        # Overlaps
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                rad_sum = r_pos[i] + r_pos[j]
                if dist_sq < rad_sum**2:
                    penalty += (rad_sum**2 - dist_sq)
        
        # Bounds
        for i in range(N):
            if x[i] < r_pos[i]:
                penalty += (r_pos[i] - x[i])**2
            if x[i] > W - r_pos[i]:
                penalty += (x[i] - (W - r_pos[i]))**2
            if y[i] < r_pos[i]:
                penalty += (r_pos[i] - y[i])**2
            if y[i] > H - r_pos[i]:
                penalty += (y[i] - (H - r_pos[i]))**2
        
        # Perimeter
        perim = 2 * (W + H)
        penalty += (perim - 4)**2
        
        return obj + lam * penalty

    # Multi-start optimization
    best_val = -np.inf
    best_x = None
    
    x_init, y_init, r_init, W_init, H_init = get_initial_config()
    v_init = np.concatenate([x_init, y_init, r_init, [W_init, H_init]])
    
    # Run several optimizations with different random perturbations and penalty weights
    for restart in range(5):
        # Perturb initial state
        if restart > 0:
            noise = np.random.normal(0, 0.05, v_init.shape)
            v_start = v_init + noise
        else:
            v_start = v_init.copy()
            
        # Ensure W+H approx 2 initially
        W_s = v_start[3*N]
        H_s = v_start[3*N+1]
        scale = 2.0 / (W_s + H_s)
        v_start[3*N] = W_s * scale
        v_start[3*N+1] = H_s * scale
        # Scale x,y,r accordingly? No, just let optimizer fix it.
        
        # Use Nelder-Mead or Powell for robustness
        # Powell is often better for this kind of non-smooth penalty
        res = minimize(
            penalty_obj, 
            v_start, 
            args=(100.0,), 
            method='Powell', 
            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        
        v_res = res.x
        x_res = v_res[:N]
        y_res = v_res[N:2*N]
        r_res = v_res[2*N:3*N]
        W_res = v_res[3*N]
        H_res = v_res[3*N+1]
        
        # Check validity
        if is_valid(x_res, y_res, r_res, W_res, H_res):
            val = np.sum(r_res)
            if val > best_val:
                best_val = val
                best_x = v_res.copy()
        
        # Try SLSQP polish on the best found so far? 
        # Or just keep the penalty result.
        # Let's try a quick SLSQP on the current best if it's valid.
        # Actually, SLSQP is slow. Let's just do more penalty iterations.
        
    if best_x is None:
        # Fallback to initial
        best_x = v_init
        best_val = np.sum(v_init[2*N:3*N])
        
    # Final Polish with SLSQP on the best candidate
    # This might fail, but if it succeeds, it's great.
    # We need to define constraints properly for SLSQP.
    # Given the time limit, maybe skip SLSQP if it's risky.
    # The current code had SLSQP. Let's keep a lightweight SLSQP attempt.
    
    def obj_slsqp(v):
        return -np.sum(v[2*N:3*N])
        
    def jac_slsqp(v):
        g = np.zeros_like(v)
        g[2*N:3*N] = -1.0
        return g
        
    constraints = []
    constraints.append({'type': 'eq', 'fun': lambda v: 2*(v[3*N] + v[3*N+1]) - 4})
    
    # Only add non-overlap constraints for pairs that are close? 
    # No, SLSQP needs all. 210 constraints is a lot.
    # Maybe just boundary constraints and perimeter? 
    # If we assume the penalty method got us close to non-overlapping, 
    # SLSQP might refine radii without breaking overlaps if we don't enforce them?
    # No, that's dangerous.
    
    # Let's stick to the penalty result. It's robust.
    
    x_final = best_x[:N]
    y_final = best_x[N:2*N]
    r_final = best_x[2*N:3*N]
    W_final = best_x[3*N]
    H_final = best_x[3*N+1]
    
    # Final scaling to ensure perimeter <= 4
    perim = 2 * (W_final + H_final)
    if perim > 4:
        scale = 4.0 / perim
        x_final *= scale
        y_final *= scale
        r_final *= scale
        W_final *= scale
        H_final *= scale
        
    # Final validity check
    if not is_valid(x_final, y_final, r_final, W_final, H_final):
        # Try to shrink radii slightly
        r_final *= 0.99
        if not is_valid(x_final, y_final, r_final, W_final, H_final):
            # Fallback to a simple valid configuration
            # 21 circles in 1x1, r=0.1
            x_fb = np.array([(i % 5 + 0.5) / 5.0 for i in range(21)])
            y_fb = np.array([(i // 5 + 0.5) / 5.0 for i in range(21)])
            r_fb = np.full(21, 0.09)
            x_final = x_fb
            y_final = y_fb
            r_final = r_fb
            W_final = 1.0
            H_final = 1.0

    result = np.zeros((N, 3))
    result[:, 0] = x_final
    result[:, 1] = y_final
    result[:, 2] = r_final
    
    return result
