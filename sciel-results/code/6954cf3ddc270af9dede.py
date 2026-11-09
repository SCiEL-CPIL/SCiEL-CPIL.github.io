import numpy as np
from scipy.optimize import minimize

N = 21

def construct():
    def greedy_pack(W, H, grid_res=80):
        xs = np.linspace(0, W, grid_res)
        ys = np.linspace(0, H, grid_res)
        circles = []
        
        for _ in range(N):
            dist_box_x = np.minimum(xs, W - xs)
            dist_box_y = np.minimum(ys, H - ys)
            dist_box = np.minimum(dist_box_x[:, None], dist_box_y[None, :])
            
            dist_circ = np.full((grid_res, grid_res), np.inf)
            
            for cx, cy, cr in circles:
                dx = xs[:, None] - cx
                dy = ys[None, :] - cy
                dist = np.sqrt(dx**2 + dy**2) - cr
                dist_circ = np.minimum(dist_circ, dist)
                
            max_r = np.minimum(dist_box, dist_circ)
            
            idx = np.unravel_index(np.argmax(max_r), max_r.shape)
            r = max_r[idx]
            x = xs[idx[0]]
            y = ys[idx[1]]
            
            # Ensure r is positive
            if r < 1e-6:
                r = 1e-6
            
            circles.append((x, y, r))
            
        return circles

    def refine(circles, W, H):
        n = len(circles)
        x0 = np.array([c[0] for c in circles])
        y0 = np.array([c[1] for c in circles])
        r0 = np.array([c[2] for c in circles])
        
        vars0 = np.concatenate((x0, y0, r0))
        
        def objective(v):
            return -np.sum(v[2*N:])
            
        def constraints(v):
            x = v[:N]
            y = v[N:2*N]
            r = v[2*N:]
            cons = []
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: W - v[i] - v[2*N+i]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: H - v[N+i] - v[2*N+i]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*N+i]})
            for i in range(N):
                for j in range(i+1, N):
                    cons.append({'type': 'ineq', 'fun': lambda v, i=i, j=j: 
                                 (v[i]-v[j])**2 + (v[N+i]-v[N+j])**2 - (v[2*N+i]+v[2*N+j])**2})
            return cons
            
        try:
            res = minimize(objective, vars0, method='SLSQP', constraints=constraints,
                           bounds=[(0.0, W)]*N + [(0.0, H)]*N + [(1e-6, 1.0)]*N,
                           options={'maxiter': 300, 'ftol': 1e-9})
            if res.success:
                x = res.x[:N]
                y = res.x[N:2*N]
                r = res.x[2*N:]
                return np.column_stack((x, y, r))
        except Exception:
            pass
            
        return np.column_stack((x0, y0, r0))

    best_val = -1
    best_result = None
    
    # Try different aspect ratios
    for W in [0.8, 0.9, 1.0, 1.1, 1.2]:
        H = 2.0 - W
        if H <= 0: continue
        
        circles = greedy_pack(W, H, grid_res=80)
        
        result = refine(circles, W, H)
        
        x, y, r = result[:, 0], result[:, 1], result[:, 2]
        valid = True
        for i in range(N):
            if x[i] < r[i] - 1e-9 or x[i] > W - r[i] + 1e-9 or y[i] < r[i] - 1e-9 or y[i] > H - r[i] + 1e-9 or r[i] <= 0:
                valid = False
                break
        for i in range(N):
            for j in range(i+1, N):
                dist2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
                sum_r = r[i] + r[j]
                if dist2 < sum_r**2 - 1e-9:
                    valid = False
                    break
            if not valid: break
            
        if valid:
            val = np.sum(r)
            if val > best_val:
                best_val = val
                best_result = result
                
    if best_result is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side * 0.9) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
        
    return best_result
