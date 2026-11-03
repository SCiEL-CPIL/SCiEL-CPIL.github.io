import numpy as np
from scipy.optimize import minimize

N = 21

def construct():
    # Initial rectangle
    W = 1.0
    H = 1.0
    
    # Lists to store placed circles
    xs = []
    ys = []
    rs = []
    
    # Phase 1: Sequential Placement
    for k in range(N):
        # Initial guess for the k-th circle
        # Try a grid of positions to find a good starting point
        best_x, best_y, best_r = 0.5, 0.5, 1e-6
        
        # Grid search for initial position
        # We want to maximize the radius of the new circle
        # Radius is limited by distance to boundaries and previous circles
        n_grid = 10
        for i in range(n_grid):
            for j in range(n_grid):
                cx = (i + 0.5) / n_grid * W
                cy = (j + 0.5) / n_grid * H
                # Calculate max radius at this center
                r_max = min(cx, W - cx, cy, H - cy)
                for m in range(k):
                    dist = np.sqrt((cx - xs[m])**2 + (cy - ys[m])**2)
                    r_max = min(r_max, dist - rs[m])
                
                if r_max > best_r:
                    best_r = r_max
                    best_x = cx
                    best_y = cy
        
        # Local optimization for the k-th circle
        # Variables: x, y, r
        # Objective: maximize r (minimize -r)
        # Constraints:
        #   x - r >= 0
        #   x + r <= W
        #   y - r >= 0
        #   y + r <= H
        #   (x - xs[m])^2 + (y - ys[m])^2 >= (r + rs[m])^2 for all m < k
        
        def obj_seq(v):
            return -v[2] # Maximize r
        
        constraints_seq = []
        # Boundary
        constraints_seq.append({'type': 'ineq', 'fun': lambda v: v[0] - v[2]})
        constraints_seq.append({'type': 'ineq', 'fun': lambda v: W - v[0] - v[2]})
        constraints_seq.append({'type': 'ineq', 'fun': lambda v: v[1] - v[2]})
        constraints_seq.append({'type': 'ineq', 'fun': lambda v: H - v[1] - v[2]})
        
        # Non-overlap with previous circles
        for m in range(k):
            xm, ym, rm = xs[m], ys[m], rs[m]
            constraints_seq.append({
                'type': 'ineq', 
                'fun': lambda v, xm=xm, ym=ym, rm=rm: (v[0]-xm)**2 + (v[1]-ym)**2 - (v[2]+rm)**2
            })
            
        # Radius > 0
        constraints_seq.append({'type': 'ineq', 'fun': lambda v: v[2]})
        
        x0_seq = np.array([best_x, best_y, best_r])
        
        res_seq = minimize(obj_seq, x0_seq, method='SLSQP', constraints=constraints_seq, options={'maxiter': 100, 'ftol': 1e-8})
        
        if res_seq.success and res_seq.x[2] > 0:
            xs.append(res_seq.x[0])
            ys.append(res_seq.x[1])
            rs.append(res_seq.x[2])
        else:
            # Fallback to grid best
            xs.append(best_x)
            ys.append(best_y)
            rs.append(max(best_r, 1e-6))

    # Phase 2: Global Refinement
    # Variables: x (N), y (N), r (N), W, H
    x0 = np.array(xs)
    y0 = np.array(ys)
    r0 = np.array(rs)
    
    v0 = np.concatenate([x0, y0, r0, [W, H]])
    
    def obj_global(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        W_val = v[3*N]
        H_val = v[3*N+1]
        return -np.sum(r)
        
    constraints_global = []
    
    # Perimeter constraint
    constraints_global.append({'type': 'eq', 'fun': lambda v: 2*(v[3*N] + v[3*N+1]) - 4})
    
    # Non-overlap
    for i in range(N):
        for j in range(i+1, N):
            constraints_global.append({
                'type': 'ineq', 
                'fun': lambda v, i=i, j=j: (v[i]-v[N+j])**2 + (v[N+i]-v[2*N+j])**2 - (v[2*N+i]+v[2*N+j])**2
            })
            
    # Boundary
    for i in range(N):
        constraints_global.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i]})
        constraints_global.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*N] - v[i] - v[2*N+i]})
        constraints_global.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i]})
        constraints_global.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*N+1] - v[N+i] - v[2*N+i]})
        
    # Radii > 0
    for i in range(N):
        constraints_global.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*N+i]})
        
    # W, H > 0
    constraints_global.append({'type': 'ineq', 'fun': lambda v: v[3*N]})
    constraints_global.append({'type': 'ineq', 'fun': lambda v: v[3*N+1]})
    
    res_global = minimize(obj_global, v0, method='SLSQP', constraints=constraints_global, options={'maxiter': 200, 'ftol': 1e-8})
    
    if res_global.success:
        x_final = res_global.x[:N]
        y_final = res_global.x[N:2*N]
        r_final = res_global.x[2*N:3*N]
        W_final = res_global.x[3*N]
        H_final = res_global.x[3*N+1]
    else:
        x_final = x0
        y_final = y0
        r_final = r0
        W_final = W
        H_final = H

    # Validation and scaling
    def is_valid(x, y, r, W, H):
        if W <= 0 or H <= 0:
            return False
        if abs(2*(W+H) - 4) > 1e-5:
            return False
        for i in range(N):
            if r[i] <= 0:
                return False
            if x[i] - r[i] < -1e-5 or x[i] + r[i] > W + 1e-5:
                return False
            if y[i] - r[i] < -1e-5 or y[i] + r[i] > H + 1e-5:
                return False
            for j in range(i+1, N):
                dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                rad_sum = r[i] + r[j]
                if dist_sq < rad_sum**2 - 1e-5:
                    return False
        return True

    if not is_valid(x_final, y_final, r_final, W_final, H_final):
        # Fallback to sequential result
        x_final = x0
        y_final = y0
        r_final = r0
        W_final = W
        H_final = H

    # Scale to ensure perimeter is exactly 4
    perim = 2 * (W_final + H_final)
    if perim > 0:
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
