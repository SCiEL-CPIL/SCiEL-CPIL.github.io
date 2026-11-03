import numpy as np
from scipy.optimize import minimize

N = 21

def construct():
    best_score = -np.inf
    best_result = None
    
    # Try different initial aspect ratios
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2]
    
    for W_init in aspect_ratios:
        H_init = 2.0 - W_init
        if H_init <= 0.1:
            continue
            
        # 1. Initialize grid
        side = int(np.ceil(np.sqrt(N)))
        # Create a grid of side x side
        # We need N points.
        # Standard grid:
        x = np.array([(i + 0.5) / side * W_init for i in range(side) for _ in range(side)][:N])
        y = np.array([(j + 0.5) / side * H_init for _ in range(side) for j in range(side)][:N])
        
        # Initial radii: small
        r = np.full(N, 0.01)
        
        # 2. Greedy Expansion
        x_curr = x.copy()
        y_curr = y.copy()
        r_curr = r.copy()
        W_curr = W_init
        H_curr = H_init
        
        # Greedy loop
        step = 0.005
        for iteration in range(50): # Reduced iterations for speed
            improved = False
            for i in range(N):
                new_r = r_curr[i] + step
                if new_r <= 0:
                    continue
                # Check bounds
                if x_curr[i] - new_r < 0 or x_curr[i] + new_r > W_curr:
                    continue
                if y_curr[i] - new_r < 0 or y_curr[i] + new_r > H_curr:
                    continue
                # Check overlap
                valid = True
                for j in range(N):
                    if i == j:
                        continue
                    dist_sq = (x_curr[i]-x_curr[j])**2 + (y_curr[i]-y_curr[j])**2
                    rad_sum = new_r + r_curr[j]
                    if dist_sq < rad_sum**2:
                        valid = False
                        break
                if valid:
                    r_curr[i] = new_r
                    improved = True
            if not improved:
                step *= 0.9
                if step < 1e-5:
                    break

        # 3. SLSQP Optimization
        def obj(v):
            r = v[2*N:3*N]
            return -np.sum(r)
            
        def jac(v):
            g = np.zeros_like(v)
            g[2*N:3*N] = -1.0
            return g
            
        constraints = []
        
        # Perimeter
        constraints.append({'type': 'eq', 'fun': lambda v: 2*(v[3*N] + v[3*N+1]) - 4})
        
        # Non-overlap
        for i in range(N):
            for j in range(i+1, N):
                constraints.append({
                    'type': 'ineq', 
                    'fun': lambda v, i=i, j=j: (v[i]-v[N+j])**2 + (v[N+i]-v[2*N+j])**2 - (v[2*N+i]+v[2*N+j])**2
                })
                
        # Boundary
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i]})
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*N] - v[i] - v[2*N+i]})
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i]})
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*N+1] - v[N+i] - v[2*N+i]})
            
        # Radii > 0
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*N+i]})
            
        # W, H > 0
        constraints.append({'type': 'ineq', 'fun': lambda v: v[3*N]})
        constraints.append({'type': 'ineq', 'fun': lambda v: v[3*N+1]})

        x_start = np.concatenate([x_curr, y_curr, r_curr, [W_curr, H_curr]])
        
        try:
            res = minimize(obj, x_start, jac=jac, method='SLSQP', constraints=constraints, options={'maxiter': 50, 'ftol': 1e-6})
            
            if res.success or res.x is not None:
                x_final = res.x[:N]
                y_final = res.x[N:2*N]
                r_final = res.x[2*N:3*N]
                W_final = res.x[3*N]
                H_final = res.x[3*N+1]
                
                # Validate
                valid = True
                if W_final <= 0 or H_final <= 0:
                    valid = False
                if abs(2*(W_final+H_final) - 4) > 1e-4:
                    valid = False
                
                for i in range(N):
                    if r_final[i] <= 0:
                        valid = False
                        break
                    if x_final[i] - r_final[i] < -1e-4 or x_final[i] + r_final[i] > W_final + 1e-4:
                        valid = False
                        break
                    if y_final[i] - r_final[i] < -1e-4 or y_final[i] + r_final[i] > H_final + 1e-4:
                        valid = False
                        break
                    for j in range(i+1, N):
                        dist_sq = (x_final[i]-x_final[j])**2 + (y_final[i]-y_final[j])**2
                        rad_sum = r_final[i] + r_final[j]
                        if dist_sq < rad_sum**2 - 1e-4:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    score = np.sum(r_final)
                    if score > best_score:
                        best_score = score
                        # Store result
                        res_arr = np.zeros((N, 3))
                        res_arr[:, 0] = x_final
                        res_arr[:, 1] = y_final
                        res_arr[:, 2] = r_final
                        best_result = res_arr
        except Exception:
            pass

    if best_result is None:
        # Fallback to a simple valid configuration
        # 1x1 square, grid
        x = np.array([(i + 0.5) / 5 for i in range(5) for _ in range(5)][:N])
        y = np.array([(j + 0.5) / 5 for _ in range(5) for j in range(5)][:N])
        r = np.full(N, 0.1)
        # Scale to fit perimeter 4 (1x1 is already perimeter 4)
        # Check overlaps for fallback? Just return a known valid one.
        # For 21 circles in 1x1, max r is roughly 1/5 = 0.2? No, grid is 5x5.
        # r=0.1 works if centers are 0.2 apart.
        # Centers: 0.1, 0.3, 0.5, 0.7, 0.9. Dist 0.2. r=0.1. Touching.
        # Sum r = 21 * 0.1 = 2.1.
        res_arr = np.zeros((N, 3))
        res_arr[:, 0] = x
        res_arr[:, 1] = y
        res_arr[:, 2] = r
        return res_arr
        
    return best_result
