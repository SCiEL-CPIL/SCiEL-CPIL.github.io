import numpy as np
from scipy.optimize import minimize, least_squares
import time
import itertools

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m  # 9
    dim_V = m * p  # 12
    dim_W = p * n  # 12
    
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
    rank_school = n * m * p  # 36
    
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
    best_rank = rank_school

    def error_func(params, k):
        """Compute sum of squared errors for rank-k decomposition."""
        U = params[:dim_U * k].reshape((dim_U, k))
        V = params[dim_U * k : dim_U * k + dim_V * k].reshape((dim_V, k))
        W = params[dim_U * k + dim_V * k :].reshape((dim_W, k))
        Rec = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.sum((Rec - T) ** 2)
        return err

    def polish_and_round(params, k):
        """Try to polish the solution and round to half-integers."""
        U = params[:dim_U * k].reshape((dim_U, k))
        V = params[dim_U * k : dim_U * k + dim_V * k].reshape((dim_V, k))
        W = params[dim_U * k + dim_V * k :].reshape((dim_W, k))
        
        # Check current error
        Rec_check = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.sum((Rec_check - T) ** 2)
        
        if err < 1e-6:
            # Polish with tighter tolerance using multiple methods
            try:
                # First polish with L-BFGS-B
                res = minimize(error_func, params, args=(k,), method='L-BFGS-B', 
                               options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12})
                params2 = res.x
                U2 = params2[:dim_U * k].reshape((dim_U, k))
                V2 = params2[dim_U * k : dim_U * k + dim_V * k].reshape((dim_V, k))
                W2 = params2[dim_U * k + dim_V * k :].reshape((dim_W, k))
                
                Rec2 = np.einsum('ic,jc,kc->ijk', U2, V2, W2)
                err2 = np.sum((Rec2 - T) ** 2)
                
                # If still good, try Nelder-Mead for final polish
                if err2 < 1e-10:
                    res3 = minimize(error_func, params2, args=(k,), method='Nelder-Mead', 
                                   options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14})
                    params3 = res3.x
                    U3 = params3[:dim_U * k].reshape((dim_U, k))
                    V3 = params3[dim_U * k : dim_U * k + dim_V * k].reshape((dim_V, k))
                    W3 = params3[dim_U * k + dim_V * k :].reshape((dim_W, k))
                    
                    Rec3 = np.einsum('ic,jc,kc->ijk', U3, V3, W3)
                    err3 = np.sum((Rec3 - T) ** 2)
                    if err3 < err2:
                        U2, V2, W2 = U3, V3, W3
                        err2 = err3
                
                # Round to nearest integer or half-integer
                U_round = np.round(U2 * 2) / 2
                V_round = np.round(V2 * 2) / 2
                W_round = np.round(W2 * 2) / 2
                
                # Verify exactness
                Rec_round = np.einsum('ic,jc,kc->ijk', U_round, V_round, W_round)
                if np.allclose(Rec_round, T, atol=1e-9):
                    return True, U_round, V_round, W_round
                    
            except Exception:
                pass
        
        return False, None, None, None

    def try_rank(k, U0, V0, W0):
        """Try to optimize from given initial values for rank k."""
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            # Check initial error
            err_init = error_func(params0, k)
            if err_init < 1e-6:
                return polish_and_round(params0, k)
            
            # Try L-BFGS-B
            res = minimize(error_func, params0, args=(k,), method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.success or error_func(res.x, k) < err_init * 0.1:
                p_best = res.x
                return polish_and_round(p_best, k)
            
        except Exception:
            pass
        
        return False, None, None, None

    def try_rank_least_squares(k, U0, V0, W0):
        """Try using least_squares for potentially better convergence."""
        def lsq_func(params):
            U = params[:dim_U * k].reshape((dim_U, k))
            V = params[dim_U * k : dim_U * k + dim_V * k].reshape((dim_V, k))
            W = params[dim_U * k + dim_V * k :].reshape((dim_W, k))
            Rec = np.einsum('ic,jc,kc->ijk', U, V, W)
            return (Rec - T).flatten()
        
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            res = least_squares(lsq_func, params0, method='trf', max_nfev=200)
            
            err = np.sum(res.fun ** 2)
            if err < 1e-6:
                return polish_and_round(res.x, k)
                
        except Exception:
            pass
        
        return False, None, None, None

    # Known lower bounds and likely ranks for 3x3x4 matrix multiplication
    # Literature suggests rank between 23 and 26
    rank_candidates = list(range(22, rank_school))
    
    for k in rank_candidates:
        if time.time() - start_time > max_time:
            break
        
        if k >= best_rank:
            continue
            
        # Multiple restarts with different strategies
        found = False
        
        # Strategy 1: Random initialization with multiple seeds
        for restart in range(5):
            if time.time() - start_time > max_time:
                break
            seed = 1000 + k * 100 + restart
            np.random.seed(seed)
            init_U = np.random.randn(dim_U, k)
            init_V = np.random.randn(dim_V, k)
            init_W = np.random.randn(dim_W, k)
            
            # Scale to reasonable magnitude
            scale = np.sqrt(max(dim_U, dim_V, dim_W) / k)
            init_U *= scale
            init_V *= scale
            init_W *= scale
            
            success, U_r, V_r, W_r = try_rank(k, init_U, init_V, init_W)
            if success:
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        # Strategy 2: Perturbed schoolbook
        if k < rank_school:
            np.random.seed(2000 + k * 10)
            U_pert = U_init[:, :k].copy()
            V_pert = V_init[:, :k].copy()
            W_pert = W_init[:, :k].copy()
            
            # Add structured perturbation
            for col in range(k):
                if col < n * m:
                    # Keep some structure
                    pass
                U_pert[np.random.randint(dim_U), col] += 0.5 * np.random.randn()
                V_pert[np.random.randint(dim_V), col] += 0.5 * np.random.randn()
                W_pert[np.random.randint(dim_W), col] += 0.5 * np.random.randn()
            
            success, U_r, V_r, W_r = try_rank(k, U_pert, V_pert, W_pert)
            if success:
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        # Strategy 3: Sparse random initialization
        for restart in range(3):
            if time.time() - start_time > max_time:
                break
            np.random.seed(3000 + k * 10 + restart)
            density = 0.4
            init_U = np.random.randn(dim_U, k)
            init_U[np.random.rand(dim_U, k) > density] = 0
            init_V = np.random.randn(dim_V, k)
            init_V[np.random.rand(dim_V, k) > density] = 0
            init_W = np.random.randn(dim_W, k)
            init_W[np.random.rand(dim_W, k) > density] = 0
            
            success, U_r, V_r, W_r = try_rank(k, init_U, init_V, init_W)
            if success:
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        # Strategy 4: Least squares approach
        for restart in range(2):
            if time.time() - start_time > max_time:
                break
            np.random.seed(4000 + k * 10 + restart)
            init_U = np.random.randn(dim_U, k)
            init_V = np.random.randn(dim_V, k)
            init_W = np.random.randn(dim_W, k)
            
            success, U_r, V_r, W_r = try_rank_least_squares(k, init_U, init_V, init_W)
            if success:
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        # Strategy 5: Structured initialization based on tensor symmetry
        if time.time() - start_time > max_time:
            break
        np.random.seed(5000 + k * 10)
        init_U = np.zeros((dim_U, k))
        init_V = np.zeros((dim_V, k))
        init_W = np.zeros((dim_W, k))
        
        # Try to exploit the structure of matrix multiplication
        for col in range(k):
            # Randomly assign to some positions
            u_idx = np.random.randint(dim_U)
            v_idx = np.random.randint(dim_V)
            w_idx = np.random.randint(dim_W)
            
            init_U[u_idx, col] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
            init_V[v_idx, col] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
            init_W[w_idx, col] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
        
        success, U_r, V_r, W_r = try_rank(k, init_U, init_V, init_W)
        if success:
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    # If no better rank found, return schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END

