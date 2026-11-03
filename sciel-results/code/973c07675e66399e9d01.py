import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    def unpack(z):
        W = z[0]
        H = 2.0 - W
        if W <= 0.01 or W >= 1.99:
            return None, None
        pts = z[1:].reshape(N, 3)
        return W, H, pts

    def objective(z):
        W, H, pts = unpack(z)
        if W is None:
            return 1e10
        return -np.sum(pts[:, 2])

    def jac_objective(z):
        jac = np.zeros(1 + 3*N)
        for i in range(N):
            idx_r = 1 + i*3 + 2
            jac[idx_r] = -1.0
        return jac

    def constraints(z):
        W, H, pts = unpack(z)
        if W is None:
            return [1e10] * (4*N + N*(N-1)//2)
        
        cons = []
        for i in range(N):
            x, y, r = pts[i]
            cons.append(r - x)
            cons.append(x + r - W)
            cons.append(r - y)
            cons.append(y + r + W - 2.0)
            
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ri = pts[i]
                xj, yj, rj = pts[j]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                sum_r_sq = (ri + rj)**2
                cons.append(sum_r_sq - dist_sq)
                
        return cons

    def jac_constraints(z):
        W, H, pts = unpack(z)
        if W is None:
            return np.ones((4*N + N*(N-1)//2, 1 + 3*N))
            
        num_cons = 4*N + N*(N-1)//2
        jac = np.zeros((num_cons, 1 + 3*N))
        k = 0
        
        for i in range(N):
            x, y, r = pts[i]
            idx_x = 1 + i*3
            idx_y = 1 + i*3 + 1
            idx_r = 1 + i*3 + 2
            
            jac[k, idx_r] = 1.0
            jac[k, idx_x] = -1.0
            k += 1
            
            jac[k, 0] = -1.0
            jac[k, idx_x] = 1.0
            jac[k, idx_r] = 1.0
            k += 1
            
            jac[k, idx_r] = 1.0
            jac[k, idx_y] = -1.0
            k += 1
            
            jac[k, 0] = 1.0
            jac[k, idx_y] = 1.0
            jac[k, idx_r] = 1.0
            k += 1
            
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ri = pts[i]
                xj, yj, rj = pts[j]
                
                idx_xi = 1 + i*3
                idx_yi = 1 + i*3 + 1
                idx_ri = 1 + i*3 + 2
                idx_xj = 1 + j*3
                idx_yj = 1 + j*3 + 1
                idx_rj = 1 + j*3 + 2
                
                dx = xi - xj
                dy = yi - yj
                sr = ri + rj
                
                jac[k, idx_xi] = -2.0 * dx
                jac[k, idx_xj] = 2.0 * dx
                jac[k, idx_yi] = -2.0 * dy
                jac[k, idx_yj] = 2.0 * dy
                jac[k, idx_ri] = 2.0 * sr
                jac[k, idx_rj] = 2.0 * sr
                k += 1
                
        return jac

    cons_dict = {
        'type': 'ineq',
        'fun': constraints,
        'jac': jac_constraints
    }

    best_val = -1e10
    best_z = None
    
    starts = []
    
    # Grid start
    side = 5
    W_init = 1.0
    H_init = 1.0
    z_grid = np.zeros(1 + 3*N)
    z_grid[0] = W_init
    for i in range(N):
        col = i % side
        row = i // side
        x = (col + 0.5) / side * W_init
        y = (row + 0.5) / side * H_init
        r = 0.45 / side
        z_grid[1 + i*3] = x
        z_grid[1 + i*3 + 1] = y
        z_grid[1 + i*3 + 2] = r
    starts.append(z_grid)
    
    np.random.seed(42)
    for _ in range(20):
        W = np.random.uniform(0.6, 1.4)
        H = 2.0 - W
        z = np.zeros(1 + 3*N)
        z[0] = W
        for i in range(N):
            x = np.random.uniform(0.1, W - 0.1)
            y = np.random.uniform(0.1, H - 0.1)
            r = np.random.uniform(0.08, 0.12)
            z[1 + i*3] = x
            z[1 + i*3 + 1] = y
            z[1 + i*3 + 2] = r
        starts.append(z)

    bounds = [(0.01, 1.99)]
    for i in range(N):
        bounds.append((0.0, 2.0))
        bounds.append((0.0, 2.0))
        bounds.append((0.001, 1.0))

    for z0 in starts:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                objective, 
                z0, 
                method='SLSQP', 
                jac=jac_objective, 
                bounds=bounds, 
                constraints=cons_dict, 
                options={'maxiter': 300, 'ftol': 1e-10}
            )
            
            val = -objective(res.x)
            cons_vals = constraints(res.x)
            if np.all(np.array(cons_vals) <= 1e-7):
                if val > best_val:
                    best_val = val
                    best_z = res.x
        except Exception:
            continue

    if best_z is None:
        # Fallback
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.45 / side) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    W = best_z[0]
    H = 2.0 - W
    pts = best_z[1:].reshape(N, 3)
    
    for i in range(N):
        if pts[i, 2] < 1e-6:
            pts[i, 2] = 1e-6
        pts[i, 0] = np.clip(pts[i, 0], pts[i, 2], W - pts[i, 2])
        pts[i, 1] = np.clip(pts[i, 1], pts[i, 2], H - pts[i, 2])
        
    return pts
