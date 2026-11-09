import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Target tensor T[u, v, w] where u=im+j, v=jp+k, w=kn+i
    T = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    # Schoolbook solution (Rank 36)
    rank_school = n * m * p
    U_init = np.zeros((dim_U, rank_school))
    V_init = np.zeros((dim_V, rank_school))
    W_init = np.zeros((dim_W, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_init[u_idx, col] = 1
                V_init[v_idx, col] = 1
                W_init[w_idx, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_init, V_init, W_init
    best_rank = rank_school

    def error_func(params, k):
        U = params[:dim_U * k].reshape((dim_U, k))
        V = params[dim_U * k : dim_U * k + dim_V * k].reshape((dim_V, k))
        W = params[dim_U * k + dim_V * k :].reshape((dim_W, k))
        Rec = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.sum((Rec - T) ** 2)
        return err

    start_time = time.time()
    max_time = 145  # Safety margin

    # Search for lower ranks
    # Based on literature, M_3,3,4 might be 25 or 26.
    ranks_to_try = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    
    for k in ranks_to_try:
        if time.time() - start_time > max_time:
            break
            
        # More restarts for lower ranks as they are more critical
        restarts = 20 if k <= 26 else 5
        
        found = False
        for restart in range(restarts):
            if time.time() - start_time > max_time:
                break
                
            seed_val = 1000 + k * 100 + restart
            np.random.seed(seed_val)
            
            # Sparse-ish initialization with small integer/half-integer values
            # This helps the optimizer stay in the vicinity of valid solutions
            init_U = np.zeros((dim_U, k))
            init_V = np.zeros((dim_V, k))
            init_W = np.zeros((dim_W, k))
            
            # Populate with random +/- 0.5 or +/- 1
            for c in range(k):
                # U
                nz_u = np.random.choice(dim_U, size=3, replace=False)
                for idx in nz_u:
                    init_U[idx, c] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                # V
                nz_v = np.random.choice(dim_V, size=4, replace=False)
                for idx in nz_v:
                    init_V[idx, c] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                # W
                nz_w = np.random.choice(dim_W, size=3, replace=False)
                for idx in nz_w:
                    init_W[idx, c] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            
            params0 = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
            
            try:
                # Use L-BFGS-B with limited iterations
                res = minimize(error_func, params0, args=(k,), method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6})
                
                p_best = res.x
                U_opt = p_best[:dim_U * k].reshape((dim_U, k))
                V_opt = p_best[dim_U * k : dim_U * k + dim_V * k].reshape((dim_V, k))
                W_opt = p_best[dim_U * k + dim_V * k :].reshape((dim_W, k))
                
                # Round to nearest half-integer
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                # Verify exactness
                Rec_round = np.einsum('ic,jc,kc->ijk', U_round, V_round, W_round)
                if np.allclose(Rec_round, T, atol=1e-9):
                    return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                    
            except Exception:
                pass

    # If no better rank found, return schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
