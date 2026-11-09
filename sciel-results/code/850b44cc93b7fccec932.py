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
    
    # Target tensor
    T = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    rank_school = n * m * p
    
    # Schoolbook solution
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
                
    best_rank = rank_school
    best_U, best_V, best_W = U_init, V_init, W_init

    def error_func(params, k):
        U = params[:dim_U * k].reshape((dim_U, k))
        V = params[dim_U * k : dim_U * k + dim_V * k].reshape((dim_V, k))
        W = params[dim_U * k + dim_V * k :].reshape((dim_W, k))
        Rec = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.sum((Rec - T) ** 2)
        return err

    start_time = time.time()
    max_time = 145  # Safety margin

    ranks_to_try = [25, 26, 27, 28, 29, 30]
    
    for k in ranks_to_try:
        if time.time() - start_time > max_time:
            break
            
        restarts = 30 if k == 25 else 10
        for restart in range(restarts):
            if time.time() - start_time > max_time:
                break
                
            seed_val = 1000 + k * 100 + restart
            np.random.seed(seed_val)
            
            if restart % 3 == 0:
                init_U = np.random.randn(dim_U, k) * 0.5
                init_V = np.random.randn(dim_V, k) * 0.5
                init_W = np.random.randn(dim_W, k) * 0.5
            elif restart % 3 == 1:
                init_U = np.random.randn(dim_U, k) * 0.5
                init_U[np.abs(init_U) < 0.3] = 0
                init_V = np.random.randn(dim_V, k) * 0.5
                init_V[np.abs(init_V) < 0.3] = 0
                init_W = np.random.randn(dim_W, k) * 0.5
                init_W[np.abs(init_W) < 0.3] = 0
            else:
                init_U = np.random.choice([-1, 0, 1], size=(dim_U, k)).astype(float)
                init_V = np.random.choice([-1, 0, 1], size=(dim_V, k)).astype(float)
                init_W = np.random.choice([-1, 0, 1], size=(dim_W, k)).astype(float)
                
            params0 = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
            
            try:
                res = minimize(error_func, params0, args=(k,), method='L-BFGS-B', 
                               options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
                
                p_best = res.x
                U_opt = p_best[:dim_U * k].reshape((dim_U, k))
                V_opt = p_best[dim_U * k : dim_U * k + dim_V * k].reshape((dim_V, k))
                W_opt = p_best[dim_U * k + dim_V * k :].reshape((dim_W, k))
                
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                Rec_round = np.einsum('ic,jc,kc->ijk', U_round, V_round, W_round)
                if np.allclose(Rec_round, T, atol=1e-9):
                    if k < best_rank:
                        best_rank = k
                        best_U, best_V, best_W = U_round, V_round, W_round
                    if k == 25:
                        return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                        
            except Exception:
                pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
