import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 4, 4, 9
    nm = n * m       # 16
    mp = m * p       # 36
    pn = p * n       # 36
    
    # Helper: build full tensor matrix from U, V, W
    # T[r, i, j, k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    def tensor_from_UVW(U, V, W):
        r = U.shape[1]
        T = np.zeros((r, n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[:, i, j, k] = U[i*m+j] * V[j*p+k] * W[k*n+i]
        return T
    
    def tensor_residual(U, V, W):
        T = tensor_from_UVW(U, V, W)
        # Target: 1 for all (i,j,k)
        target = np.ones((T.shape[0], n*m*p))
        T_flat = T.reshape(T.shape[0], -1)
        return np.sum((T_flat - target)**2)
    
    # Start with schoolbook decomposition
    rank_sb = n * m * p  # 144
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    # Try to find a lower-rank decomposition using numerical optimization
    # We'll try ranks from 50 to 100 and see if we can find a valid one
    
    best_U, best_V, best_W = None, None, None
    best_rank = rank_sb
    
    # Try some candidate ranks
    candidate_ranks = [60, 70, 80, 90, 100]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
            
        # Initialize with random values
        np.random.seed(42 + target_rank)
        U_init = np.random.randn(nm, target_rank)
        V_init = np.random.randn(mp, target_rank)
        W_init = np.random.randn(pn, target_rank)
        
        # Scale to reasonable magnitude
        U_init /= np.sqrt(nm)
        V_init /= np.sqrt(mp)
        W_init /= np.sqrt(pn)
        
        # Flatten parameters
        def flatten_params(x):
            U = x[:nm*target_rank].reshape(nm, target_rank)
            V = x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
            W = x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
            return U, V, W
        
        def objective(x):
            if time.time() > deadline - 10:
                return 1e10
            U, V, W = flatten_params(x)
            T = tensor_from_UVW(U, V, W)
            T_flat = T.reshape(target_rank, -1)
            # We need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all (i,j,k)
            # This is equivalent to T being such that for each (i,j,k), 
            # sum over r of T[r,i,j,k] = 1
            # But T[r,i,j,k] is already the product for rank r
            # So we need sum_r T[r,i,j,k] = 1
            # Actually, the tensor equation is:
            # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all (i,j,k)
            # This means the tensor T (where T[r,i,j,k] = U[i,j,r]*V[j,k,r]*W[k,i,r])
            # should satisfy sum_r T[r,i,j,k] = 1
            # So residual = sum_{i,j,k} (sum_r T[r,i,j,k] - 1)^2
            S = np.sum(T, axis=0)  # shape (n, m, p)
            return np.sum((S - 1)**2)
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Use L-BFGS-B for optimization
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 50, 'maxfun': 2000, 'ftol': 1e-6})
        
        if result.success or result.fun < 1e-3:
            U_opt, V_opt, W_opt = flatten_params(result.x)
            # Verify
            S = np.sum(tensor_from_UVW(U_opt, V_opt, W_opt), axis=0)
            max_err = np.max(np.abs(S - 1))
            if max_err < 1e-3:
                best_U, best_V, best_W = U_opt, V_opt, W_opt
                best_rank = target_rank
                break
    
    # If optimization didn't find something good, use schoolbook
    if best_U is None:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
