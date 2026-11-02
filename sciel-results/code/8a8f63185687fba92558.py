import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11
H = np.sqrt(3) / 2

# Precompute indices for combinations
TRIPLETS = list(itertools.combinations(range(N), 3))
IDX_I = np.array([t[0] for t in TRIPLETS])
IDX_J = np.array([t[1] for t in TRIPLETS])
IDX_K = np.array([t[2] for t in TRIPLETS])

def construct():
    start_time = time.time()
    timeout = 115.0 # Safety margin
    
    # Barycentric to Cartesian
    # P = u*V0 + v*V1 + w*V2
    # V0=(0,0), V1=(1,0), V2=(0.5, H)
    # x = v + 0.5w = v + 0.5(1-u-v) = 0.5 + 0.5v - 0.5u
    # y = Hw = H(1-u-v)
    def bary_to_cart(u, v):
        # u, v shape (N,)
        x = 0.5 + 0.5 * v - 0.5 * u
        y = H * (1.0 - u - v)
        return x, y

    def compute_areas(u, v):
        x, y = bary_to_cart(u, v)
        # P_i, P_j, P_k
        xi = x[IDX_I]; xj = x[IDX_J]; xk = x[IDX_K]
        yi = y[IDX_I]; yj = y[IDX_J]; yk = y[IDX_K]
        
        # Area = 0.5 * | xi(yj - yk) + xj(yk - yi) + xk(yi - yj) |
        areas = 0.5 * np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        return areas

    def objective(params, beta):
        # params shape (2N,)
        u = params[:N]
        v = params[N:]
        
        # Check validity and add penalty
        # u >= 0, v >= 0, u+v <= 1
        # L-BFGS-B handles bounds, but not u+v <= 1
        # Penalty for u+v > 1
        sum_uv = u + v
        viol = np.maximum(0, sum_uv - 1.0)
        penalty = 100.0 * np.sum(viol ** 2)
        
        areas = compute_areas(u, v)
        
        # Soft min: -1/beta * log(sum(exp(-beta * areas)))
        # We want to maximize min area, so minimize -soft_min
        # -soft_min = 1/beta * log(sum(exp(-beta * areas)))
        
        # Numerical stability
        exp_term = np.exp(-beta * areas)
        log_sum = np.log(np.sum(exp_term))
        soft_min_val = -1.0 / beta * log_sum
        
        obj = -soft_min_val + penalty
        return obj

    best_points = None
    best_val = float('inf')
    
    # Initial points: random valid
    while True:
        u_init = np.random.rand(N)
        v_init = np.random.rand(N)
        if np.all(u_init + v_init <= 1.0):
            break
            
    params_init = np.concatenate([u_init, v_init])
    
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Multi-stage optimization
    betas = [10, 50, 100]
    
    for beta in betas:
        if time.time() - start_time > timeout - 5:
            break
            
        # Try a few restarts for each beta
        n_restarts = 3
        for r in range(n_restarts):
            if time.time() - start_time > timeout - 5:
                break
                
            if r == 0 and beta == betas[0]:
                p0 = params_init
            else:
                # Perturb previous best or random
                if best_points is not None:
                    p0 = best_points + np.random.randn(2*N) * 0.05
                    # Clip to bounds
                    p0 = np.clip(p0, 0.0, 1.0)
                else:
                    while True:
                        u_init = np.random.rand(N)
                        v_init = np.random.rand(N)
                        if np.all(u_init + v_init <= 1.0):
                            break
                    p0 = np.concatenate([u_init, v_init])
            
            res = minimize(
                objective, 
                p0, 
                args=(beta,), 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if res.fun < best_val:
                # Check validity
                u_opt = res.x[:N]
                v_opt = res.x[N:]
                # Allow small numerical errors
                if np.all(u_opt >= -1e-9) and np.all(v_opt >= -1e-9) and np.all(u_opt + v_opt <= 1.0 + 1e-9):
                    # Clip to valid range
                    u_opt = np.clip(u_opt, 0.0, 1.0)
                    v_opt = np.clip(v_opt, 0.0, 1.0)
                    
                    best_val = res.fun
                    best_points = res.x.copy()

    # Finalize best points
    if best_points is None:
        # Fallback
        u = np.random.rand(N)
        v = np.random.rand(N)
        while np.any(u + v > 1.0):
            u = np.random.rand(N)
            v = np.random.rand(N)
        best_points = np.concatenate([u, v])
        
    u_final = best_points[:N]
    v_final = best_points[N:]
    
    x, y = bary_to_cart(u_final, v_final)
    points = np.column_stack([x, y])
    
    return points
