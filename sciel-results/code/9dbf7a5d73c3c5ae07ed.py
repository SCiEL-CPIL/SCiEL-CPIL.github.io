import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = 140.0
    
    # Variables: w, x1, y1, r1, ..., xN, yN, rN
    # h = 2 - w
    
    def objective_reduced(v):
        # v[0] = w
        # r_i is at index 3 + 3*i
        r = v[3::3]
        return -np.sum(r)
        
    def constraints_reduced(v):
        w = v[0]
        h = 2.0 - w
        cons = []
        
        for i in range(N):
            xi = v[1 + 3*i]
            yi = v[2 + 3*i]
            ri = v[3 + 3*i]
            
            # Inside rect
            cons.append(xi - ri)
            cons.append(w - xi - ri)
            cons.append(yi - ri)
            cons.append(h - yi - ri)
            
        for i in range(N):
            xi = v[1 + 3*i]
            yi = v[2 + 3*i]
            ri = v[3 + 3*i]
            for j in range(i+1, N):
                xj = v[1 + 3*j]
                yj = v[2 + 3*j]
                rj = v[3 + 3*j]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                sum_r = ri + rj
                cons.append(dist_sq - sum_r**2)
                
        return cons

    def bounds_reduced():
        b = [(0.1, 1.9)]
        for i in range(N):
            b.append((0.0, 2.0))
            b.append((0.0, 2.0))
            b.append((0.01, 0.5))
        return b

    def run_solver(w_init, jitter=0.0, seed=42):
        np.random.seed(seed)
        h_init = 2.0 - w_init
        
        v0 = np.zeros(1 + 3*N)
        v0[0] = w_init
        
        side = int(np.ceil(np.sqrt(N)))
        pts = []
        for i in range(side):
            for j in range(side):
                pts.append((i, j))
        
        np.random.shuffle(pts)
        pts = pts[:N]
        
        dx = w_init / side
        dy = h_init / side
        r_init = min(dx, dy) * 0.49
        
        for k, (i, j) in enumerate(pts):
            x = (i + 0.5) * dx + np.random.uniform(-jitter, jitter)
            y = (j + 0.5) * dy + np.random.uniform(-jitter, jitter)
            r = r_init + np.random.uniform(-0.01, 0.01)
            
            v0[1 + 3*k] = x
            v0[2 + 3*k] = y
            v0[3 + 3*k] = r
            
        # Clip to bounds
        for k in range(N):
            v0[3 + 3*k] = max(0.01, min(v0[3 + 3*k], 0.5))
            v0[1 + 3*k] = max(v0[3 + 3*k], min(v0[1 + 3*k], w_init - v0[3 + 3*k]))
            v0[2 + 3*k] = max(v0[3 + 3*k], min(v0[2 + 3*k], h_init - v0[3 + 3*k]))
            
        res = minimize(objective_reduced, v0, method='SLSQP', bounds=bounds_reduced(), 
                       constraints=[{'type': 'ineq', 'fun': constraints_reduced}],
                       options={'maxiter': 300, 'ftol': 1e-9, 'disp': False})
        
        if res.success:
            v_opt = res.x
            w_opt = v_opt[0]
            h_opt = 2.0 - w_opt
            
            valid = True
            for i in range(N):
                xi = v_opt[1 + 3*i]
                yi = v_opt[2 + 3*i]
                ri = v_opt[3 + 3*i]
                if xi < ri - 1e-5 or w_opt - xi < ri - 1e-5 or yi < ri - 1e-5 or h_opt - yi < ri - 1e-5:
                    valid = False
                    break
                for j in range(i+1, N):
                    xj = v_opt[1 + 3*j]
                    yj = v_opt[2 + 3*j]
                    rj = v_opt[3 + 3*j]
                    dist_sq = (xi - xj)**2 + (yi - yj)**2
                    sum_r = ri + rj
                    if dist_sq < sum_r**2 - 1e-5:
                        valid = False
                        break
                if not valid: break
                
            if valid:
                out = np.zeros((N, 3))
                for i in range(N):
                    out[i, 0] = v_opt[1 + 3*i]
                    out[i, 1] = v_opt[2 + 3*i]
                    out[i, 2] = v_opt[3 + 3*i]
                return np.sum(out[:, 2]), w_opt, out
            else:
                return 0.0, w_init, None
        else:
            return 0.0, w_init, None

    best_score = 0.0
    best_w = 1.0
    best_sol = None
    
    aspect_ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    
    for idx, w_init in enumerate(aspect_ratios):
        if time.time() > deadline:
            break
            
        # Vary seed for each aspect ratio
        seed = 100 + idx
        score, w_res, sol = run_solver(w_init, jitter=0.02, seed=seed)
        
        if score > best_score:
            best_score = score
            best_w = w_res
            best_sol = sol

    # Polish
    if best_sol is not None and time.time() < deadline - 15:
        # Reconstruct v0 from best_sol
        v0 = np.zeros(1 + 3*N)
        v0[0] = best_w
        for i in range(N):
            v0[1 + 3*i] = best_sol[i, 0]
            v0[2 + 3*i] = best_sol[i, 1]
            v0[3 + 3*i] = best_sol[i, 2]
            
        res = minimize(objective_reduced, v0, method='SLSQP', bounds=bounds_reduced(), 
                       constraints=[{'type': 'ineq', 'fun': constraints_reduced}],
                       options={'maxiter': 500, 'ftol': 1e-12, 'disp': False})
        
        if res.success:
            v_opt = res.x
            w_opt = v_opt[0]
            h_opt = 2.0 - w_opt
            
            valid = True
            for i in range(N):
                xi = v_opt[1 + 3*i]
                yi = v_opt[2 + 3*i]
                ri = v_opt[3 + 3*i]
                if xi < ri - 1e-5 or w_opt - xi < ri - 1e-5 or yi < ri - 1e-5 or h_opt - yi < ri - 1e-5:
                    valid = False
                    break
                for j in range(i+1, N):
                    xj = v_opt[1 + 3*j]
                    yj = v_opt[2 + 3*j]
                    rj = v_opt[3 + 3*j]
                    dist_sq = (xi - xj)**2 + (yi - yj)**2
                    sum_r = ri + rj
                    if dist_sq < sum_r**2 - 1e-5:
                        valid = False
                        break
                if not valid: break
            
            if valid:
                out = np.zeros((N, 3))
                for i in range(N):
                    out[i, 0] = v_opt[1 + 3*i]
                    out[i, 1] = v_opt[2 + 3*i]
                    out[i, 2] = v_opt[3 + 3*i]
                new_score = np.sum(out[:, 2])
                if new_score > best_score:
                    best_score = new_score
                    best_sol = out

    if best_sol is None:
        # Fallback to seed-like grid
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3))
        return np.array(cells[:N])
        
    return best_sol
