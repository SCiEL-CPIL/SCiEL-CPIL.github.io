import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    def objective(z):
        return -np.sum(z[2::3])
        
    # Pre-calculate pairs for overlap constraints
    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            pairs.append((i, j))
            
    def boundary_constraints(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        c1 = x - r
        c2 = 1 - x - r
        c3 = y - r
        c4 = 1 - y - r
        return np.concatenate([c1, c2, c3, c4])
        
    def overlap_constraints(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        c = np.zeros(len(pairs))
        for k, (i, j) in enumerate(pairs):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            c[k] = dist - (r[i] + r[j])
        return c
        
    def all_constraints(z):
        return np.concatenate([boundary_constraints(z), overlap_constraints(z)])
        
    bounds = []
    for i in range(N):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r
        
    best_val = -np.inf
    best_z = None
    
    num_restarts = 100
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if restart < 50:
            # Grid-based start with jitter
            side = int(np.ceil(np.sqrt(N)))
            grid_x = []
            grid_y = []
            idx = 0
            for i in range(side):
                for j in range(side):
                    if idx < N:
                        jitter = 0.05
                        gx = (i + 0.5) / side + np.random.uniform(-jitter, jitter)
                        gy = (j + 0.5) / side + np.random.uniform(-jitter, jitter)
                        gx = np.clip(gx, 0.05, 0.95)
                        gy = np.clip(gy, 0.05, 0.95)
                        grid_x.append(gx)
                        grid_y.append(gy)
                        idx += 1
                    else:
                        break
                if idx >= N:
                    break
            
            r_init = 0.5 / side * 0.8
            r_init = np.clip(r_init, 0.01, 0.4)
            
            z0 = np.zeros(3*N)
            for i in range(N):
                z0[3*i] = grid_x[i]
                z0[3*i+1] = grid_y[i]
                z0[3*i+2] = r_init + np.random.uniform(-0.01, 0.01)
        else:
            # Random start
            x0 = np.random.uniform(0.1, 0.9, N)
            y0 = np.random.uniform(0.1, 0.9, N)
            r0 = np.random.uniform(0.05, 0.2, N)
            
            z0 = np.zeros(3*N)
            for i in range(N):
                z0[3*i] = x0[i]
                z0[3*i+1] = y0[i]
                z0[3*i+2] = r0[i]
                
        try:
            res = minimize(
                objective, 
                z0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=[{'type': 'ineq', 'fun': all_constraints}],
                options={'maxiter': 200, 'ftol': 1e-8}
            )
        except Exception:
            continue
            
        if res.success:
            z_opt = res.x
            val = -res.fun
            
            # Verify validity
            x = z_opt[0::3]
            y = z_opt[1::3]
            r = z_opt[2::3]
            
            valid = True
            if np.any(x < r - 1e-5) or np.any(1 - x < r - 1e-5) or np.any(y < r - 1e-5) or np.any(1 - y < r - 1e-5):
                valid = False
            if valid:
                for k, (i, j) in enumerate(pairs):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < r[i] + r[j] - 1e-5:
                        valid = False
                        break
                        
            if valid and val > best_val:
                best_val = val
                best_z = z_opt.copy()
                
    if best_z is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    x = best_z[0::3]
    y = best_z[1::3]
    r = best_z[2::3]
    result = np.column_stack((x, y, r))
    return result
