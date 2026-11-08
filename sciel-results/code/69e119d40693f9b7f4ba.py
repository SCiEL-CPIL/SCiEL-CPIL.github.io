"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    best_sum_r = -np.inf
    best_x = None
    
    # Vectorized constraint functions
    def boundary_constraints(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        c1 = x - r
        c2 = 1.0 - x - r
        c3 = y - r
        c4 = 1.0 - y - r
        c5 = r
        
        return np.concatenate([c1, c2, c3, c4, c5])
        
    def overlap_constraints(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist_sq = dx**2 + dy**2
        
        sum_r = r[:, None] + r[None, :]
        sum_r_sq = sum_r**2
        
        i, j = np.triu_indices(N, k=1)
        cons = dist_sq[i, j] - sum_r_sq[i, j]
        
        return cons
        
    constraints = [
        {'type': 'ineq', 'fun': boundary_constraints},
        {'type': 'ineq', 'fun': overlap_constraints}
    ]
    
    def objective(v):
        r = v[2::3]
        return -np.sum(r)
        
    n_restarts = 40
    
    for k in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
            
        x0 = np.zeros(3 * N)
        
        if k % 2 == 0:
            # Grid-based start with perturbation
            side = int(np.ceil(np.sqrt(N)))
            grid_points = []
            for i in range(side):
                for j in range(side):
                    if len(grid_points) < N:
                        grid_points.append(((i + 0.5) / side, (j + 0.5) / side))
            
            for i in range(N):
                noise = 0.1 * np.random.rand(2)
                x0[3*i] = np.clip(grid_points[i][0] + noise[0], 0.05, 0.95)
                x0[3*i+1] = np.clip(grid_points[i][1] + noise[1], 0.05, 0.95)
                x0[3*i+2] = 0.05
        else:
            # Random start
            for i in range(N):
                x0[3*i] = 0.1 + 0.8 * np.random.rand()
                x0[3*i+1] = 0.1 + 0.8 * np.random.rand()
                x0[3*i+2] = 0.05
                
        try:
            res = minimize(objective, x0, method='SLSQP', constraints=constraints, 
                           options={'maxiter': 150, 'ftol': 1e-8})
            
            if res.success:
                v_res = res.x
                x = v_res[0::3]
                y = v_res[1::3]
                r = v_res[2::3]
                
                valid = True
                if np.any(x < r - 1e-6) or np.any(x > 1.0 - r + 1e-6):
                    valid = False
                if np.any(y < r - 1e-6) or np.any(y > 1.0 - r + 1e-6):
                    valid = False
                if np.any(r < 0):
                    valid = False
                    
                if valid:
                    dx = x[:, None] - x[None, :]
                    dy = y[:, None] - y[None, :]
                    dist_sq = dx**2 + dy**2
                    sum_r = r[:, None] + r[None, :]
                    sum_r_sq = sum_r**2
                    i, j = np.triu_indices(N, k=1)
                    if np.any(dist_sq[i, j] < sum_r_sq[i, j] - 1e-6):
                        valid = False
                        
                if valid:
                    sum_r = np.sum(r)
                    if sum_r > best_sum_r:
                        best_sum_r = sum_r
                        best_x = v_res.copy()
                        
        except Exception:
            pass
            
    if best_x is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    result = best_x.reshape(N, 3)
    return result
# EVOLVE-BLOCK-END
