import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Build the matrix multiplication tensor T
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
                
    def to_list(arr):
        return [[float(x) for x in row] for row in arr]
        
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    def reconstruct(U, V, W):
        return np.einsum('ir,jr,kr->ijk', U, V, W)
        
    def optimized_try_rank(target_rank, max_restarts=5):
        dim_u = n * m
        dim_v = m * p
        dim_w = p * n
        
        def loss_func(params):
            u_size = dim_u * target_rank
            v_size = dim_v * target_rank
            w_size = dim_w * target_rank
            
            U = params[:u_size].reshape(dim_u, target_rank)
            V = params[u_size:u_size+v_size].reshape(dim_v, target_rank)
            W = params[u_size+v_size:].reshape(dim_w, target_rank)
            
            recon = reconstruct(U, V, W)
            diff = T - recon
            return np.sum(diff**2)
            
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(2000 + restart * 7)
            scale = 1.0
            U_init = np.random.randn(dim_u, target_rank) * scale
            V_init = np.random.randn(dim_v, target_rank) * scale
            W_init = np.random.randn(dim_w, target_rank) * scale
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(
                    loss_func, 
                    params_init, 
                    method='L-BFGS-B', 
                    options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10}
                )
            except Exception:
                continue
                
            params_best = res.x
            u_size = dim_u * target_rank
            v_size = dim_v * target_rank
            
            U_best = params_best[:u_size].reshape(dim_u, target_rank)
            V_best = params_best[u_size:u_size+v_size].reshape(dim_v, target_rank)
            W_best = params_best[u_size+v_size:].reshape(dim_w, target_rank)
            
            recon = reconstruct(U_best, V_best, W_best)
            err = np.max(np.abs(T - recon))
            
            if err < 1e-5:
                # Attempt snapping to half-integers
                U_snap = np.round(U_best * 2) / 2
                V_snap = np.round(V_best * 2) / 2
                W_snap = np.round(W_best * 2) / 2
                
                recon_snap = reconstruct(U_snap, V_snap, W_snap)
                err_snap = np.max(np.abs(T - recon_snap))
                
                if err_snap < 1e-9:
                    return U_snap, V_snap, W_snap
                    
        return None

    # Try ranks from 96 upwards
    # Prioritize 96, 97, 98 as they are theoretically likely
    ranks_to_try = [96, 97, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        # Use more restarts for smaller ranks
        restarts = 10 if r < 100 else 5
        
        result = optimized_try_rank(r, max_restarts=restarts)
        if result is not None:
            U_f, V_f, W_f = result
            best_U, best_V, best_W = U_f, V_f, W_f
            best_rank = r
            break
            
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
