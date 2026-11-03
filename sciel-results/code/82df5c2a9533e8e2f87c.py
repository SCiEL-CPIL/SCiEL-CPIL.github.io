import numpy as np
from scipy.optimize import minimize

N = 21

def construct():
    # We will use a penalty-based approach with L-BFGS-B.
    # Variables: x (N), y (N), r (N), aspect_ratio (1) -> W = 2*aspect/(1+aspect), H = 2/(1+aspect)
    # Actually, simpler: W and H with constraint W+H=2 handled by parameterizing W in [0.1, 1.9], H = 2-W.
    # Total variables: 3N + 1 = 64.
    
    def smooth_penalty(x, y, r, W, H):
        # x, y, r are arrays of size N. W, H are scalars.
        # Penalty for overlaps: sum max(0, r_i + r_j - dist_ij)^2
        # Penalty for bounds: sum max(0, r_i - x_i)^2 + ...
        
        pen = 0.0
        
        # Overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                rad_sum = r[i] + r[j]
                if dist < rad_sum:
                    pen += (rad_sum - dist)**2
        
        # Bounds
        for i in range(N):
            if x[i] < r[i]:
                pen += (r[i] - x[i])**2
            if x[i] > W - r[i]:
                pen += (x[i] - (W - r[i]))**2
            if y[i] < r[i]:
                pen += (r[i] - y[i])**2
            if y[i] > H - r[i]:
                pen += (y[i] - (H - r[i]))**2
        
        return pen

    def objective(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        W = v[3*N]
        H = 2.0 - W # Enforce W+H=2 strictly in variable space
        
        # Ensure W, H are positive
        if W <= 0.01 or H <= 0.01:
            return 1e10
            
        # Clip radii to be non-negative for calculation, but let optimizer handle it
        r_pos = np.maximum(r, 0.0)
        
        pen = smooth_penalty(x, y, r_pos, W, H)
        
        # Objective: Maximize sum of radii, penalize violations
        # Weight for penalty needs to be high enough to enforce constraints
        # but not so high it causes numerical issues.
        # Dynamic penalty weight? Start high, reduce? 
        # For L-BFGS-B, a fixed high weight is usually okay if initialized well.
        return -np.sum(r_pos) + 100.0 * pen

    def jac(v):
        # Numerical gradient is safer and often fast enough for N=21 (64 vars)
        # L-BFGS-B can handle numerical gradients.
        # We can provide analytic gradient if needed, but numerical is robust.
        # Let's use finite difference for simplicity and robustness against implementation errors.
        # scipy.optimize.minimize with method='L-BFGS-B' uses finite difference by default if jac is None.
        return None

    best_value = -np.inf
    best_x = None
    
    # Multiple starts
    # 1. Grid start
    # 2. Random start
    
    starts = []
    
    # Start 1: Grid 5x5 (approx)
    side = 5
    x_grid = np.array([(i + 0.5) / side for i in range(side) for _ in range(side)][:N])
    y_grid = np.array([(j + 0.5) / side for _ in range(side) for j in range(side)][:N])
    r_grid = np.full(N, 0.1) # Small initial radii
    W_grid = 1.0
    starts.append(np.concatenate([x_grid, y_grid, r_grid, [W_grid]]))
    
    # Start 2: Random
    np.random.seed(42)
    for k in range(5):
        W_rand = np.random.uniform(0.5, 1.5)
        H_rand = 2.0 - W_rand
        x_rand = np.random.uniform(0.1, W_rand-0.1, N)
        y_rand = np.random.uniform(0.1, H_rand-0.1, N)
        r_rand = np.full(N, 0.05)
        starts.append(np.concatenate([x_rand, y_rand, r_rand, [W_rand]]))
        
    # Start 3: Aspect ratio variations of grid
    for W_val in [0.8, 1.2, 1.5]:
        H_val = 2.0 - W_val
        # Scale grid to fit W, H
        x_s = x_grid * W_val
        y_s = y_grid * H_val
        r_s = np.full(N, 0.05)
        starts.append(np.concatenate([x_s, y_s, r_s, [W_val]]))

    for start_v in starts:
        # Ensure start is valid-ish (positive radii, inside box)
        # L-BFGS-B handles bounds, but we can provide them.
        # Bounds:
        # x_i, y_i, r_i >= 0
        # W in [0.1, 1.9]
        
        bounds = []
        for i in range(3*N):
            bounds.append((0.0, None))
        bounds.append((0.1, 1.9)) # W
        
        try:
            res = minimize(objective, start_v, method='L-BFGS-B', jac=None, bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-9})
            
            if res.success:
                x_final = res.x[:N]
                y_final = res.x[N:2*N]
                r_final = res.x[2*N:3*N]
                W_final = res.x[3*N]
                H_final = 2.0 - W_final
                
                # Check validity
                valid = True
                for i in range(N):
                    if r_final[i] <= 0:
                        valid = False
                        break
                    if x_final[i] - r_final[i] < -1e-5 or x_final[i] + r_final[i] > W_final + 1e-5:
                        valid = False
                        break
                    if y_final[i] - r_final[i] < -1e-5 or y_final[i] + r_final[i] > H_final + 1e-5:
                        valid = False
                        break
                    for j in range(i+1, N):
                        dx = x_final[i] - x_final[j]
                        dy = y_final[i] - y_final[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < r_final[i] + r_final[j] - 1e-5:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    val = np.sum(r_final)
                    if val > best_value:
                        best_value = val
                        best_x = res.x.copy()
                        
        except Exception as e:
            pass

    if best_x is None:
        # Fallback to a very simple valid configuration if all optimizations fail
        # 21 circles in 1x1 square, radius 0.04
        x_f = np.array([(i % 5 + 0.5) / 5.0 for i in range(21)])
        y_f = np.array([(i // 5 + 0.5) / 5.0 for i in range(21)])
        r_f = np.full(21, 0.04)
        W_f = 1.0
        H_f = 1.0
        result = np.zeros((N, 3))
        result[:, 0] = x_f
        result[:, 1] = y_f
        result[:, 2] = r_f
        return result

    x_final = best_x[:N]
    y_final = best_x[N:2*N]
    r_final = best_x[2*N:3*N]
    W_final = best_x[3*N]
    H_final = 2.0 - W_final

    # Final cleanup: ensure perimeter is exactly 4 (it is by construction W+H=2)
    # and radii are positive.
    r_final = np.maximum(r_final, 1e-9)
    
    result = np.zeros((N, 3))
    result[:, 0] = x_final
    result[:, 1] = y_final
    result[:, 2] = r_final
    
    return result
