"""Improved circle packing in perimeter-4 rectangle."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    best_value = -np.inf
    best_result = None
    best_w = 1.4
    
    def check_and_update(x, w, h):
        nonlocal best_value, best_result, best_w
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return
        # x is [x1,y1,r1, x2,y2,r2, ...]
        valid = True
        # Check bounds
        for i in range(N):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            if ri < 1e-9 or xi - ri < -1e-6 or xi + ri > w + 1e-6 or yi - ri < -1e-6 or yi + ri > h + 1e-6:
                valid = False
                break
        if not valid:
            return
        # Check disjointness
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                dist_sq = (xi-xj)**2 + (yi-yj)**2
                min_dist = ri + rj
                if dist_sq < (min_dist - 1e-7)**2:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            val = np.sum(x[2::3])
            if val > best_value:
                best_value = val
                best_result = x.copy()
                best_w = w
    
    def make_initial_grid(a, b, w, h, r0=0.09):
        """Create initial grid configuration."""
        x0 = np.zeros(3*N)
        for i in range(N):
            col = i % a
            row = i // a
            x0[3*i] = (col + 0.5) * w / a
            x0[3*i+1] = (row + 0.5) * h / b
            x0[3*i+2] = r0
        return x0
    
    def objective(params, soft_beta=50.0):
        """
        Smooth penalty objective using soft constraints.
        params: [w, x1,y1,r1, ..., x21,y21,r21]
        """
        w_opt = params[0]
        h_opt = 2.0 - w_opt
        if w_opt < 0.2 or h_opt < 0.2:
            return 1e10
        x = params[1:]
        
        penalty = 0.0
        
        # Boundary constraints (soft)
        for i in range(N):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            if ri < 1e-6:
                penalty += 1000 * (1e-6 - ri)**2
            # Left
            viol = max(0, ri - xi)
            if viol > 0:
                penalty += 50 * viol**2
            # Right
            viol = max(0, xi + ri - w_opt)
            if viol > 0:
                penalty += 50 * viol**2
            # Bottom
            viol = max(0, ri - yi)
            if viol > 0:
                penalty += 50 * viol**2
            # Top
            viol = max(0, yi + ri - h_opt)
            if viol > 0:
                penalty += 50 * viol**2
        
        # Pairwise non-overlap (soft)
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                dist_sq = (xi-xj)**2 + (yi-yj)**2
                min_dist = ri + rj
                dist = np.sqrt(dist_sq + 1e-12)
                if dist < min_dist:
                    viol = min_dist - dist
                    penalty += 50 * viol**2
        
        return penalty - np.sum(x[2::3])
    
    # Try multiple initial configurations with different aspect ratios
    configurations = [
        # (a, b, w, h, r0)
        (7, 3, 1.4, 0.6, 0.09),
        (5, 5, 1.0, 1.0, 0.09),
        (6, 4, 1.2, 0.8, 0.09),
        (4, 6, 0.8, 1.2, 0.09),
        (3, 7, 0.6, 1.4, 0.09),
        (7, 3, 1.3, 0.7, 0.085),
        (6, 4, 1.1, 0.9, 0.085),
        (5, 5, 1.1, 0.9, 0.095),
    ]
    
    for trial, (a, b, w_init, h_init, r0) in enumerate(configurations):
        if time.time() - start_time > deadline:
            break
        
        x0 = make_initial_grid(a, b, w_init, h_init, r0)
        x0_full = np.concatenate(([w_init], x0))
        
        # Try with different optimization settings
        try:
            # First pass: Nelder-Mead
            result = minimize(objective, x0_full, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7,
                                    'adaptive': True, 'maxfev': 5000})
            if time.time() - start_time > deadline:
                break
            if result.fun < -1.0:
                w_opt = result.x[0]
                h_opt = 2.0 - w_opt
                x = result.x[1:]
                check_and_update(x, w_opt, h_opt)
                
                # Second pass: polish with L-BFGS-B using tighter tolerance
                if time.time() - start_time < deadline - 5:
                    try:
                        # Use SLSQP for polishing with tighter constraints
                        def obj_polish(params):
                            w_opt = params[0]
                            h_opt = 2.0 - w_opt
                            if w_opt < 0.3 or h_opt < 0.3:
                                return 1e10
                            x = params[1:]
                            
                            penalty = 0.0
                            for i in range(N):
                                xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                                if ri < 1e-7:
                                    penalty += 10000 * (1e-7 - ri)**2
                                viol = max(0, ri - xi)
                                if viol > 0:
                                    penalty += 100 * viol**2
                                viol = max(0, xi + ri - w_opt)
                                if viol > 0:
                                    penalty += 100 * viol**2
                                viol = max(0, ri - yi)
                                if viol > 0:
                                    penalty += 100 * viol**2
                                viol = max(0, yi + ri - h_opt)
                                if viol > 0:
                                    penalty += 100 * viol**2
                            
                            for i in range(N):
                                for j in range(i+1, N):
                                    xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                                    xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                                    dist_sq = (xi-xj)**2 + (yi-yj)**2
                                    min_dist = ri + rj
                                    dist = np.sqrt(dist_sq + 1e-12)
                                    if dist < min_dist:
                                        viol = min_dist - dist
                                        penalty += 100 * viol**2
                            
                            return penalty - np.sum(x[2::3])
                        
                        result2 = minimize(obj_polish, result.x, method='Nelder-Mead',
                                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8,
                                                  'adaptive': True, 'maxfev': 3000})
                        if time.time() - start_time > deadline:
                            break
                        if result2.fun < result.fun - 1e-6:
                            w_opt = result2.x[0]
                            h_opt = 2.0 - w_opt
                            x = result2.x[1:]
                            check_and_update(x, w_opt, h_opt)
                    except Exception:
                        pass
        except Exception:
            pass
    
    # If optimization didn't improve, use best grid solution
    if best_result is None:
        a, b = 7, 3
        w, h = 1.4, 0.6
        r = 0.1 - 1e-4
        best_result = np.zeros(3*N)
        for i in range(a):
            for j in range(b):
                idx = i*b + j
                best_result[3*idx] = (i + 0.5) * w / a
                best_result[3*idx+1] = (j + 0.5) * h / b
                best_result[3*idx+2] = r
    
    # Format output: (x, y, r) for each circle
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = best_result[3*i:3*i+3]
    
    return result
# EVOLVE-BLOCK-END
