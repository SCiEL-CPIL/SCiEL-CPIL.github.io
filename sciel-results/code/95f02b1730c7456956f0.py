import numpy as np
from scipy.optimize import minimize
import time
import itertools

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Target tensor T
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

    # Theoretical minimal rank for 3x3x4 is 25.
    # We will prioritize searching for rank 25, then 26, etc.
    # We use multiple restarts for each rank.
    
    ranks_to_try = [25, 26, 27, 28]
    
    for k in ranks_to_try:
        if time.time() - start_time > max_time:
            break
            
        # Number of restarts depends on how likely this rank is.
        # Rank 25 is the target, so give it more restarts.
        restarts = 300 if k == 25 else (150 if k == 26 else 50)
        
        found = False
        for restart in range(restarts):
            if time.time() - start_time > max_time:
                break
                
            # Deterministic but varied seeds
            seed_val = 1000 + k * 100 + restart
            np.random.seed(seed_val)
            
            # Initialization: Small integers, as the solution is likely to have small integer entries.
            # Using randint is often more effective for discrete-like structures.
            init_U = np.random.randint(-3, 4, size=(dim_U, k)).astype(float)
            init_V = np.random.randint(-3, 4, size=(dim_V, k)).astype(float)
            init_W = np.random.randint(-3, 4, size=(dim_W, k)).astype(float)
            
            # Mix in some Gaussian noise to escape symmetries
            init_U += np.random.randn(dim_U, k) * 0.1
            init_V += np.random.randn(dim_V, k) * 0.1
            init_W += np.random.randn(dim_W, k) * 0.1
            
            params0 = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
            
            try:
                # L-BFGS-B is good for large N, but 900 vars is moderate.
                # We use a tight tolerance to ensure convergence to a local min.
                res = minimize(error_func, params0, args=(k,), method='L-BFGS-B', 
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                
                p_best = res.x
                U_opt = p_best[:dim_U * k].reshape((dim_U, k))
                V_opt = p_best[dim_U * k : dim_U * k + dim_V * k].reshape((dim_V, k))
                W_opt = p_best[dim_U * k + dim_V * k :].reshape((dim_W, k))
                
                # Check error before rounding to avoid unnecessary rounding of bad solutions
                Rec_opt = np.einsum('ic,jc,kc->ijk', U_opt, V_opt, W_opt)
                err_opt = np.sum((Rec_opt - T) ** 2)
                
                if err_opt < 1e-6:
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
