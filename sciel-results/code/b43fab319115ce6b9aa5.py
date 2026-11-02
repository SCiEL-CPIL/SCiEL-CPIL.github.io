import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Precompute index arrays
    i_idx = np.array([i for i in range(n) for j in range(m) for k in range(p)])
    j_idx = np.array([j for i in range(n) for j in range(m) for k in range(p)])
    k_idx = np.array([k for i in range(n) for j in range(m) for k in range(p)])
    
    U_idx = i_idx * m + j_idx
    V_idx = j_idx * p + k_idx
    W_idx = k_idx * n + i_idx
    
    def objective(params, target_rank):
        U = params[0:target_rank * n * m].reshape(n * m, target_rank)
        V = params[target_rank * n * m:2 * target_rank * n * m].reshape(m * p, target_rank)
        W = params[2 * target_rank * n * m:3 * target_rank * n * m].reshape(p * n, target_rank)
        
        U_sel = U[U_idx, :]
        V_sel = V[V_idx, :]
        W_sel = W[W_idx, :]
        
        prod = U_sel * V_sel * W_sel
        sums = np.sum(prod, axis=1)
        
        return np.sum((sums - 1.0) ** 2)
    
    for target_rank in [46, 45, 44, 42, 40, 36, 30, 24, 20, 18]:
        if time.time() > deadline - 20:
            break
        
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + target_rank * 100 + trial)
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            for r in range(target_rank):
                nz_U = np.random.choice(n * m, size=min(3, n * m), replace=False)
                U[nz_U, r] = np.random.choice([-1, 1], size=len(nz_U))
                
                nz_V = np.random.choice(m * p, size=min(3, m * p), replace=False)
                V[nz_V, r] = np.random.choice([-1, 1], size=len(nz_V))
                
                nz_W = np.random.choice(p * n, size=min(3, p * n), replace=False)
                W[nz_W, r] = np.random.choice([-1, 1], size=len(nz_W))
            
            params_init = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = minimize(lambda x: objective(x, target_rank), params_init,
                                 method='L-BFGS-B',
                                 options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6,
                                         'maxfun': 2000, 'maxls': 20})
                
                if result.fun < 1e-6:
                    U_opt = result.x[0:target_rank * n * m].reshape(n * m, target_rank)
                    V_opt = result.x[target_rank * n * m:2 * target_rank * n * m].reshape(m * p, target_rank)
                    W_opt = result.x[2 * target_rank * n * m:3 * target_rank * n * m].reshape(p * n, target_rank)
                    
                    if objective(result.x, target_rank) < 1e-6:
                        best_U, best_V, best_W, best_rank = U_opt, V_opt, W_opt, target_rank
                        break
            except Exception:
                pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
