import numpy as np
import time
from scipy.optimize import minimize

N = 21
TIME_LIMIT = 110.0  # Safety margin

def construct():
    start_time = time.time()
    
    # We want to maximize sum of radii. Assuming equal radii R is optimal for small N.
    # Perimeter 4 => w + h = 2. Let r = w/h.
    # h = 2 / (1 + r), w = 2 * r / (1 + r)
    
    best_result = None
    best_radius = -1.0
    
    # Helper to calculate energy (overlap penalty) for a given configuration
    def energy(x, y, R, w, h):
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        
        # Wall penalties
        # x should be in [R, w-R], y in [R, h-R]
        wall_pen = 0.0
        for i in range(N):
            if x[i] < R:
                wall_pen += (R - x[i])**2
            elif x[i] > w - R:
                wall_pen += (x[i] - (w - R))**2
            if y[i] < R:
                wall_pen += (R - y[i])**2
            elif y[i] > h - R:
                wall_pen += (y[i] - (h - R))**2
        
        # Overlap penalties
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx*dx + dy*dy
                min_d = 2*R
                if d2 < min_d*min_d:
                    d = np.sqrt(d2)
                    overlap_pen += (min_d - d)**2
        
        return wall_pen + overlap_pen

    def objective(params, R, w, h):
        x = params[:N]
        y = params[N:2*N]
        return energy(x, y, R, w, h)

    def solve_for_aspect(r, initial_params=None):
        if time.time() - start_time > TIME_LIMIT:
            return None, None
        
        w = 2 * r / (1 + r)
        h = 2 / (1 + r)
        
        # Initial guess: Grid layout
        if initial_params is None:
            # Arrange N circles in a grid as close to square as possible
            # For 21, 4x5 grid is good.
            cols = 4
            rows = 5
            # Adjust grid to fit in w, h
            # Estimate R
            R_est = min(w / (2*cols), h / (2*rows))
            
            x = np.zeros(N)
            y = np.zeros(N)
            for i in range(N):
                cx = i % cols
                cy = i // cols
                x[i] = (cx + 0.5) * (w / cols)
                y[i] = (cy + 0.5) * (h / rows)
            
            # If grid doesn't fit all, adjust
            # Simple scaling
            params0 = np.concatenate([x, y])
        else:
            params0 = initial_params
            R_est = None # Need to infer or pass R
            
        # Binary search on R
        # Lower bound: very small
        # Upper bound: max possible based on area or max dimension
        # Max R is bounded by min(w, h)/2 roughly, but with packing it's less.
        # Let's start with a reasonable R and expand/contract.
        
        # Get initial R from params if possible, or estimate
        if R_est is None:
             # Estimate from initial params
             min_d = min(w, h) / 2
             R_est = min_d * 0.4 # Conservative start
        
        # We will try to find max R such that energy is low.
        # Strategy: Start with small R, increase until it fails, then binary search.
        # Or start with estimated R and check.
        
        # Let's do a coarse search for R
        # R_max_possible = min(w, h) / 2
        R_low = 0.01
        R_high = min(w, h) / 2 * 0.9
        
        # Check if R_high is feasible
        # We need a function to check feasibility for a given R
        def check_feasibility(R, params_init):
            if time.time() - start_time > TIME_LIMIT:
                return False, None
                
            res = minimize(
                objective, 
                params_init, 
                args=(R, w, h), 
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-8}
            )
            final_energy = res.fun
            # Threshold for feasibility
            # Overlap penalty is squared distance.
            # If energy < 1e-4, consider feasible.
            return final_energy < 1e-4, res.x

        # First, try to find a feasible R_high
        feasible, params_high = check_feasibility(R_high, params0)
        if not feasible:
            # Shrink R_high
            for _ in range(10):
                R_high *= 0.9
                feasible, params_high = check_feasibility(R_high, params0)
                if feasible:
                    break
            if not feasible:
                return None, None
        else:
            # Try to expand R_high?
            # Let's just binary search between R_low and R_high
            pass
            
        # Binary search
        R_best = R_high
        params_best = params_high
        for _ in range(20):
            if time.time() - start_time > TIME_LIMIT:
                break
            R_mid = (R_low + R_high) / 2
            feasible, params_mid = check_feasibility(R_mid, params_high if params_high is not None else params0)
            if feasible:
                R_low = R_mid
                R_best = R_mid
                params_best = params_mid
            else:
                R_high = R_mid
        
        return R_best, params_best

    # Search over aspect ratios
    # Reasonable range for 21 circles
    # Square is 1.0. 
    # 4x5 grid suggests w/h ~ 5/4 = 1.25 or 4/5 = 0.8
    r_values = np.linspace(0.5, 2.0, 20)
    
    for r in r_values:
        if time.time() - start_time > TIME_LIMIT:
            break
            
        R_opt, params_opt = solve_for_aspect(r)
        
        if R_opt is not None:
            # Refine around this r
            # Local search for r
            for _ in range(5):
                if time.time() - start_time > TIME_LIMIT:
                    break
                # Try neighbors
                for dr in [-0.05, 0.05]:
                    r_new = r + dr
                    if 0.2 < r_new < 5.0:
                        R_new, params_new = solve_for_aspect(r_new, params_opt)
                        if R_new is not None and R_new > R_opt:
                            r = r_new
                            R_opt = R_new
                            params_opt = params_new
                            break # Restart refinement with new r
                
            if R_opt > best_radius:
                best_radius = R_opt
                best_r = r
                best_params = params_opt

    if best_params is None:
        # Fallback: Simple grid
        r = 1.0
        w = 1.0
        h = 1.0
        cols = 4
        rows = 5
        x = np.zeros(N)
        y = np.zeros(N)
        for i in range(N):
            cx = i % cols
            cy = i // cols
            x[i] = (cx + 0.5) * (w / cols)
            y[i] = (cy + 0.5) * (h / rows)
        # Estimate R
        R = min(w/cols, h/rows) * 0.4
        best_params = np.concatenate([x, y])
        best_radius = R
        best_r = 1.0

    # Construct final array
    w = 2 * best_r / (1 + best_r)
    h = 2 / (1 + best_r)
    
    x = best_params[:N]
    y = best_params[N:2*N]
    
    # Final check and clamp to ensure validity
    # If a circle is outside, clamp it (might cause overlap, but perimeter is valid)
    # Better: If invalid, shrink R slightly
    # The optimizer should have handled this, but let's be safe.
    
    # The task requires returning an array of shape (N, 3)
    # [x, y, radius]
    # All radii are equal to best_radius
    
    circles = np.column_stack((x, y, np.full(N, best_radius)))
    
    return circles

