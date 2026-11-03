import numpy as np
from scipy.optimize import minimize
import random

N = 21

def construct():
    # We will try multiple random permutations and initial aspect ratios
    best_sum_r = -np.inf
    best_config = None
    
    # Try a few aspect ratios for W, H with 2(W+H)=4 => W+H=2
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.7, 2.0, 0.5]
    
    for ar in aspect_ratios:
        W = ar
        H = 2.0 - W
        if H <= 0: continue
        
        # Try a few random permutations
        for perm_trial in range(5):
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            
            # Shuffle indices to determine insertion order
            # Actually, we just insert circles 0..N-1 in a random order of "priority"
            # But the problem is fixed circles. The "permutation" refers to which circle 
            # we place first. Since all circles are identical in capacity, we can just 
            # place them one by one. The "randomness" comes from the starting guess 
            # for the local optimizer.
            
            for k in range(N):
                # Define objective for circle k: maximize r_k
                # Variables: x_k, y_k, r_k
                # Constraints:
                # 1. x_k >= r_k, x_k <= W - r_k
                # 2. y_k >= r_k, y_k <= H - r_k
                # 3. dist(k, j) >= r_k + r_j for j < k
                
                def obj_k(v):
                    xk, yk, rk = v
                    # Penalize violations
                    pen = 0
                    if xk < rk: pen += (rk - xk)**2
                    if xk > W - rk: pen += (xk - (W - rk))**2
                    if yk < rk: pen += (rk - yk)**2
                    if yk > H - rk: pen += (yk - (H - rk))**2
                    
                    for j in range(k):
                        dx = xk - x[j]
                        dy = yk - y[j]
                        d = np.sqrt(dx*dx + dy*dy)
                        min_d = rk + r[j]
                        if d < min_d:
                            pen += (min_d - d)**2
                            
                    return -rk + 100 * pen
                
                # Initial guess for circle k
                # Try to find a spot in the middle of the remaining space
                # Simple heuristic: start from center, or random
                # Random start is better for diversity
                rx = random.uniform(0.1, W - 0.1)
                ry = random.uniform(0.1, H - 0.1)
                # Estimate max possible r
                # Distance to boundaries
                d_bound = min(rx, W-rx, ry, H-ry)
                # Distance to existing circles
                d_circ = np.inf
                for j in range(k):
                    dx = rx - x[j]
                    dy = ry - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    d_circ = min(d_circ, d - r[j])
                r_est = max(0.01, min(d_bound, d_circ) * 0.9)
                
                v0 = np.array([rx, ry, r_est])
                
                # Bounds for L-BFGS-B
                # r_k must be > 0
                # x, y can be anywhere, penalty handles bounds
                # But L-BFGS-B needs bounds.
                # Let's use SLSQP for this small sub-problem as it handles inequalities well
                # Or just Nelder-Mead which is robust for small dims
                
                # Using Nelder-Mead is safer for this non-smooth penalty landscape
                res = minimize(obj_k, v0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
                
                xk, yk, rk = res.x
                # Clip to ensure validity if optimizer failed
                if rk <= 0:
                    rk = 1e-4
                if xk < rk: xk = rk
                if xk > W - rk: xk = W - rk
                if yk < rk: yk = rk
                if yk > H - rk: yk = H - rk
                
                # Final check for overlaps, if failed, shrink r
                valid = True
                for j in range(k):
                    dx = xk - x[j]
                    dy = yk - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < rk + r[j] - 1e-5:
                        # Shrink rk to fit
                        rk = max(1e-4, d - r[j] - 1e-4)
                        valid = False
                        break
                
                x[k] = xk
                y[k] = yk
                r[k] = rk
                
            sum_r = np.sum(r)
            if sum_r > best_sum_r:
                best_sum_r = sum_r
                best_config = (x.copy(), y.copy(), r.copy(), W, H)
                
    if best_config is None:
        # Fallback to a simple grid
        side = int(np.ceil(np.sqrt(N)))
        x = np.array([(i + 0.5) / side for i in range(side) for _ in range(side)][:N])
        y = np.array([(j + 0.5) / side for _ in range(side) for j in range(side)][:N])
        r = np.full(N, 0.5 / side - 1e-4)
        W = 1.0
        H = 1.0
    else:
        x, y, r, W, H = best_config
        
    # Final Polish: Global SLSQP to refine the best configuration
    # This step is crucial to squeeze out more radius
    
    x_start = np.concatenate([x, y, r, [W, H]])
    
    def obj_global(v):
        x_g = v[:N]
        y_g = v[N:2*N]
        r_g = v[2*N:3*N]
        W_g = v[3*N]
        H_g = v[3*N+1]
        
        pen = 0
        # Perimeter
        pen += (2*(W_g + H_g) - 4)**2
        
        # Overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = x_g[i] - x_g[j]
                dy = y_g[i] - y_g[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = r_g[i] + r_g[j]
                if d < min_d:
                    pen += (min_d - d)**2
                    
        # Bounds
        for i in range(N):
            if x_g[i] < r_g[i]: pen += (r_g[i] - x_g[i])**2
            if x_g[i] > W_g - r_g[i]: pen += (x_g[i] - (W_g - r_g[i]))**2
            if y_g[i] < r_g[i]: pen += (r_g[i] - y_g[i])**2
            if y_g[i] > H_g - r_g[i]: pen += (y_g[i] - (H_g - r_g[i]))**2
            
        return -np.sum(r_g) + 100 * pen
        
    res = minimize(obj_global, x_start, method='SLSQP', options={'maxiter': 50, 'ftol': 1e-6})
    
    if res.success:
        x_final = res.x[:N]
        y_final = res.x[N:2*N]
        r_final = res.x[2*N:3*N]
        W_final = res.x[3*N]
        H_final = res.x[3*N+1]
        
        # Verify validity
        valid = True
        if abs(2*(W_final + H_final) - 4) > 1e-4:
            valid = False
        for i in range(N):
            if r_final[i] <= 0:
                valid = False
                break
            if x_final[i] < r_final[i] - 1e-4 or x_final[i] > W_final - r_final[i] + 1e-4:
                valid = False
                break
            if y_final[i] < r_final[i] - 1e-4 or y_final[i] > H_final - r_final[i] + 1e-4:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    dx = x_final[i] - x_final[j]
                    dy = y_final[i] - y_final[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < r_final[i] + r_final[j] - 1e-4:
                        valid = False
                        break
                if not valid: break
        
        if not valid:
            # Fall back to best_config
            x_final = x
            y_final = y
            r_final = r
            W_final = W
            H_final = H
    else:
        x_final = x
        y_final = y
        r_final = r
        W_final = W
        H_final = H
        
    # Scale to ensure perimeter is exactly 4 if needed
    perim = 2 * (W_final + H_final)
    if perim > 4:
        scale = 4.0 / perim
        x_final *= scale
        y_final *= scale
        r_final *= scale
        W_final *= scale
        H_final *= scale
        
    result = np.zeros((N, 3))
    result[:, 0] = x_final
    result[:, 1] = y_final
    result[:, 2] = r_final
    
    return result
