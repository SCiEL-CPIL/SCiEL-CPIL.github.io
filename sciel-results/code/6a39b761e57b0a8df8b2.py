import numpy as np
from scipy.optimize import minimize
import time
import itertools

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m  # 9
    dim_V = m * p  # 12
    dim_W = p * n  # 12
    
    # Construct the target tensor T[i,j,k] = 1 if (i,j) in A, (j,k) in B, (k,i) in C
    # Indices: u_idx = i*m+j, v_idx = j*p+k, w_idx = k*n+i
    T = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    # Schoolbook rank
    rank_school = n * m * p  # 36
    
    # Time guard
    start_time = time.time()
    max_time = 145  # Leave 5s for safety against 150s limit
    
    # Initial solution: Schoolbook
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

    def try_rank(k, seed_base, max_restarts=10):
        nonlocal best_U, best_V, best_W, best_rank
        for restart in range(max_restarts):
            if time.time() - start_time > max_time:
                return False
                
            np.random.seed(seed_base + restart)
            
            # Random initialization
            scale = 1.0
            init_U = np.random.randn(dim_U, k) * scale
            init_V = np.random.randn(dim_V, k) * scale
            init_W = np.random.randn(dim_W, k) * scale
            params0 = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
            
            try:
                # Use L-BFGS-B with tighter tolerances
                res = minimize(error_func, params0, args=(k,), method='L-BFGS-B', 
                               options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
                
                p_best = res.x
                U_opt = p_best[:dim_U * k].reshape((dim_U, k))
                V_opt = p_best[dim_U * k : dim_U * k + dim_V * k].reshape((dim_V, k))
                W_opt = p_best[dim_U * k + dim_V * k :].reshape((dim_W, k))
                
                # Round to nearest integer or half-integer
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                # Verify exactness
                Rec_round = np.einsum('ic,jc,kc->ijk', U_round, V_round, W_round)
                if np.allclose(Rec_round, T, atol=1e-9):
                    best_U, best_V, best_W = U_round, V_round, W_round
                    best_rank = k
                    return True
                    
            except Exception:
                pass
        return False

    # Try ranks from a reasonable lower bound up to rank_school - 1
    # Known lower bounds for 3x3x4 are around 20-25
    # We start at 23 and go up
    for k in range(23, rank_school):
        if time.time() - start_time > max_time:
            break
            
        # Try this rank
        found = try_rank(k, seed_base=1000 + k * 100)
        if found:
            # Found a solution at rank k, try to go lower? No, we go up from lower bound.
            # Since we start at 23 and go up, the first success is the minimal rank found in this range.
            # But wait, if 23 fails, we try 24, etc.
            # We should break immediately upon finding a valid solution at the current k
            # because we are iterating k in increasing order.
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # If no better rank found, return schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
