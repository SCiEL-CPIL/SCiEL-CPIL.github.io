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
    
    # Construct the target tensor
    T = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    # Schoolbook rank
    rank_school = n * m * p
    
    # Time guard
    start_time = time.time()
    max_time = 135  # Leave 15s for safety against 150s limit
    
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

    def error_func(params, k):
        U = params[:dim_U * k].reshape((dim_U, k))
        V = params[dim_U * k : dim_U * k + dim_V * k].reshape((dim_V, k))
        W = params[dim_U * k + dim_V * k :].reshape((dim_W, k))
        Rec = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.sum((Rec - T) ** 2)
        return err

    def try_rank(k, U0, V0, W0):
        """Try to optimize from given initial values for rank k."""
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            # First pass: L-BFGS-B with moderate tolerance
            res = minimize(error_func, params0, args=(k,), method='L-BFGS-B', 
                           options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            
            p_best = res.x
            U_opt = p_best[:dim_U * k].reshape((dim_U, k))
            V_opt = p_best[dim_U * k : dim_U * k + dim_V * k].reshape((dim_V, k))
            W_opt = p_best[dim_U * k + dim_V * k :].reshape((dim_W, k))
            
            # Check if we're close enough to try rounding
            Rec_check = np.einsum('ic,jc,kc->ijk', U_opt, V_opt, W_opt)
            err = np.sum((Rec_check - T) ** 2)
            
            if err < 1e-6:
                # Polish with tighter tolerance
                res2 = minimize(error_func, p_best, args=(k,), method='L-BFGS-B', 
                               options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
                p_best2 = res2.x
                U_opt = p_best2[:dim_U * k].reshape((dim_U, k))
                V_opt = p_best2[dim_U * k : dim_U * k + dim_V * k].reshape((dim_V, k))
                W_opt = p_best2[dim_U * k + dim_V * k :].reshape((dim_W, k))
                
                # Round to nearest integer or half-integer
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                # Verify exactness
                Rec_round = np.einsum('ic,jc,kc->ijk', U_round, V_round, W_round)
                if np.allclose(Rec_round, T, atol=1e-9):
                    return True, U_round, V_round, W_round
                    
        except Exception:
            pass
        
        return False, None, None, None

    # Try ranks from a reasonable lower bound up to rank_school - 1
    # For 3x3x4, known results suggest rank around 23-25
    for k in range(23, rank_school):
        if time.time() - start_time > max_time:
            break
            
        # Multiple restarts with different strategies
        found = False
        
        # Strategy 1: Random initialization
        for restart in range(3):
            if time.time() - start_time > max_time:
                break
            np.random.seed(2000 + k * 10 + restart)
            init_U = np.random.randn(dim_U, k)
            init_V = np.random.randn(dim_V, k)
            init_W = np.random.randn(dim_W, k)
            success, U_r, V_r, W_r = try_rank(k, init_U, init_V, init_W)
            if success:
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        # Strategy 2: Perturbed schoolbook (take first k columns and perturb)
        if k < rank_school:
            U_pert = U_init[:, :k].copy() + 0.1 * np.random.randn(dim_U, k)
            V_pert = V_init[:, :k].copy() + 0.1 * np.random.randn(dim_V, k)
            W_pert = W_init[:, :k].copy() + 0.1 * np.random.randn(dim_W, k)
            success, U_r, V_r, W_r = try_rank(k, U_pert, V_pert, W_pert)
            if success:
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        # Strategy 3: Sparse random initialization (many zeros)
        for restart in range(2):
            if time.time() - start_time > max_time:
                break
            np.random.seed(3000 + k * 10 + restart)
            density = 0.3
            init_U = np.random.randn(dim_U, k)
            init_U[np.random.rand(dim_U, k) > density] = 0
            init_V = np.random.randn(dim_V, k)
            init_V[np.random.rand(dim_V, k) > density] = 0
            init_W = np.random.randn(dim_W, k)
            init_W[np.random.rand(dim_W, k) > density] = 0
            success, U_r, V_r, W_r = try_rank(k, init_U, init_V, init_W)
            if success:
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    # If no better rank found, return schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
