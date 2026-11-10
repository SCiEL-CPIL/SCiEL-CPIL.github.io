import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 600s limit

    best_score = -np.inf
    best_centers = None
    best_sides = None

    def is_valid(centers, sides):
        tol = 1e-7
        # Check bounds and positive sides
        for i in range(n):
            x, y, s = centers[i, 0], centers[i, 1], sides[i]
            if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(s):
                return False
            if s <= 0:
                return False
            if x - s/2 < -tol or x + s/2 > 1 + tol:
                return False
            if y - s/2 < -tol or y + s/2 > 1 + tol:
                return False
        
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j])/2 - tol
                if dx < t and dy < t:
                    return False
        return True

    def update_best(params):
        nonlocal best_score, best_centers, best_sides
        # Unpack based on parameter order: [x0, y0, s0, x1, y1, s1, ...]
        # Current program uses: centers = params[0:2*n], sides = params[2*n:3*n]
        # But inspiration uses: cx = x[0::3], cy = x[1::3], s = x[2::3]
        # Let's stick to the current program's parameter layout for consistency with its logic, 
        # OR switch to the inspiration's layout if it's better.
        # The current program's layout: [x0, y0, x1, y1, ..., x9, y9, s0, s1, ..., s9]
        # The inspiration's layout: [x0, y0, s0, x1, y1, s1, ...]
        # The inspiration's layout is more natural for per-square operations.
        # Let's adopt the inspiration's layout: x = [cx0, cy0, s0, cx1, cy1, s1, ...]
        
        cx = params[0::3]
        cy = params[1::3]
        s = params[2::3]
        
        centers = np.stack([cx, cy], axis=1)
        sides = s
        
        if is_valid(centers, sides):
            score = np.sum(sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()

    def objective(params):
        # params: [cx0, cy0, s0, cx1, cy1, s1, ...]
        cx = params[0::3]
        cy = params[1::3]
        s = params[2::3]
        
        # Ensure positive sides for calculation stability, but penalize if too small
        s_eff = np.maximum(s, 1e-6)
        
        # Objective: -sum(s)
        obj = -np.sum(s_eff)
        
        # Boundary Penalty
        # We want cx - s/2 >= 0 and cx + s/2 <= 1
        # Penalty if violated
        pen = 0.0
        for i in range(n):
            # Lower bound
            if cx[i] - s_eff[i]/2 < 0:
                pen += (s_eff[i]/2 - cx[i])**2
            if cy[i] - s_eff[i]/2 < 0:
                pen += (s_eff[i]/2 - cy[i])**2
            
            # Upper bound
            if cx[i] + s_eff[i]/2 > 1:
                pen += (cx[i] + s_eff[i]/2 - 1)**2
            if cy[i] + s_eff[i]/2 > 1:
                pen += (cy[i] + s_eff[i]/2 - 1)**2
                
        # Overlap Penalty
        # Smooth penalty for overlaps
        # If |cx_i - cx_j| < (s_i+s_j)/2 AND |cy_i - cy_j| < (s_i+s_j)/2
        # Penalty = max(0, (s_i+s_j)/2 - |cx_i - cx_j|) * max(0, (s_i+s_j)/2 - |cy_i - cy_j|)
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                dist_needed = (s_eff[i] + s_eff[j]) / 2
                
                pen_x = max(0, dist_needed - dx)
                pen_y = max(0, dist_needed - dy)
                
                # Product penalty is 0 if separated in either dimension
                pen += pen_x * pen_y

        # Weights
        # High weight to ensure constraints are satisfied
        return obj + 1000.0 * pen

    # Generate seeds
    seeds = []
    
    # Seed 1: The "1.2" configuration from inspiration
    # 2 large (0.5), 8 medium (0.25)
    centers_init = np.array([
        [0.25, 0.25],
        [0.75, 0.25],
        [0.125, 0.625],
        [0.375, 0.625],
        [0.625, 0.625],
        [0.875, 0.625],
        [0.125, 0.875],
        [0.375, 0.875],
        [0.625, 0.875],
        [0.875, 0.875]
    ])
    sides_init = np.array([
        0.5, 0.5,
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25
    ])
    
    # Convert to flat params [cx0, cy0, s0, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = sides_init[i]
    seeds.append(x0)
    
    # Seed 2: Random valid-ish start
    np.random.seed(42)
    for k in range(5):
        # Generate random positions and sizes
        # Try to create a non-overlapping set by rejection or just random and let optimizer fix
        # Random small squares
        s_rand = np.random.rand(n) * 0.1 + 0.05
        cx_rand = np.random.rand(n) * (1 - s_rand) + s_rand/2
        cy_rand = np.random.rand(n) * (1 - s_rand) + s_rand/2
        
        x_rand = np.zeros(3 * n)
        for i in range(n):
            x_rand[3*i] = cx_rand[i]
            x_rand[3*i+1] = cy_rand[i]
            x_rand[3*i+2] = s_rand[i]
        seeds.append(x_rand)

    # Optimization Loop
    # Use SLSQP or Nelder-Mead?
    # SLSQP is good for constrained problems but we are using penalty method.
    # Nelder-Mead is derivative-free and robust to non-smoothness (though our penalty is smooth).
    # Let's try SLSQP first with the penalty objective, as it handles the gradient better.
    # Or use COBYLA?
    # Let's use SLSQP with bounds.
    
    bounds = []
    for i in range(n):
        # cx: [0, 1]
        # cy: [0, 1]
        # s: [1e-4, 1.0]
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-4, 1.0))

    for seed in seeds:
        if time.time() > deadline:
            break
            
        try:
            # Run SLSQP
            res = minimize(
                objective,
                seed,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if res.success or res.fun < objective(seed):
                update_best(res.x)
                
            # Also try Nelder-Mead if time permits, as it can escape local minima differently
            if time.time() < deadline - 60:
                res_nm = minimize(
                    objective,
                    seed,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res_nm.fun < objective(seed):
                    update_best(res_nm.x)

        except Exception:
            pass

    # Final Polish:
    # If we have a best solution, try a high-precision local search around it
    if best_centers is not None and time.time() < deadline - 10:
        # Reconstruct params
        cx = best_centers[:, 0]
        cy = best_centers[:, 1]
        s = best_sides
        
        x_polish = np.zeros(3 * n)
        for i in range(n):
            x_polish[3*i] = cx[i]
            x_polish[3*i+1] = cy[i]
            x_polish[3*i+2] = s[i]
            
        try:
            res_polish = minimize(
                objective,
                x_polish,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            if res_polish.fun < objective(x_polish):
                update_best(res_polish.x)
        except Exception:
            pass

    # Fallback if no valid solution found (should not happen with seed 1)
    if best_centers is None:
        best_centers = centers_init
        best_sides = sides_init

    return best_centers, best_sides
