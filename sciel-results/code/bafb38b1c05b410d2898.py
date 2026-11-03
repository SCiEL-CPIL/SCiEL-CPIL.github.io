"""Matrix multiplication 3x3x7 rank-minimal decomposition search."""
import numpy as np
import time
import random
import sys

DIMENSIONS = [3, 3, 7]

def _verify(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is correct.
    U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    T = np.einsum('ad,bd,cd->abc', U_arr, V_arr, W_arr)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > 1e-6:
                    return False
    
    valid = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid[a, b, c] = True
    
    invalid_mask = ~valid
    if np.any(np.abs(T[invalid_mask]) > 1e-6):
        return False
        
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Precompute target tensor
    T_target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                valid_indices.append((a, b, c))
    
    valid_mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    for (a, b, c) in valid_indices:
        valid_mask[a, b, c] = True
    
    # Schoolbook baseline (rank 63)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    from scipy.optimize import minimize
    
    def snap(val):
        nearest = round(val * 2) / 2.0
        if abs(val - nearest) < 1e-6:
            return nearest
        return val
    
    # Try ranks from 62 down to 45
    for target_rank in range(62, 44, -1):
        if time.time() > deadline - 15.0:
            break
        
        found = False
        max_restarts = 8
        if target_rank <= 50:
            max_restarts = 5
        
        for restart in range(max_restarts):
            if time.time() > deadline - 8.0:
                break
            
            np.random.seed(hash((target_rank, restart)) % (2**31))
            
            # Random initialization
            scale = 1.0 / np.sqrt(target_rank)
            U = np.random.randn(n*m, target_rank) * scale
            V = np.random.randn(m*p, target_rank) * scale
            W = np.random.randn(p*n, target_rank) * scale
            
            def objective_maxnorm(x):
                """Minimize max absolute error (better for sparse target)."""
                U_flat = x[:n*m*target_rank]
                V_flat = x[n*m*target_rank:n*m*target_rank + m*p*target_rank]
                W_flat = x[n*m*target_rank + m*p*target_rank:]
                
                U_mat = U_flat.reshape(n*m, target_rank)
                V_mat = V_flat.reshape(m*p, target_rank)
                W_mat = W_flat.reshape(p*n, target_rank)
                
                T = np.einsum('ad,bd,cd->abc', U_mat, V_mat, W_mat)
                err = np.abs(T - T_target)
                return np.max(err)
            
            def objective_sumnorm(x):
                """Sum of squared errors."""
                U_flat = x[:n*m*target_rank]
                V_flat = x[n*m*target_rank:n*m*target_rank + m*p*target_rank]
                W_flat = x[n*m*target_rank + m*p*target_rank:]
                
                U_mat = U_flat.reshape(n*m, target_rank)
                V_mat = V_flat.reshape(m*p, target_rank)
                W_mat = W_flat.reshape(p*n, target_rank)
                
                T = np.einsum('ad,bd,cd->abc', U_mat, V_mat, W_mat)
                return np.sum((T - T_target)**2)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Phase 1: L-BFGS-B with sum-of-squares (smooth)
            try:
                res1 = minimize(objective_sumnorm, x0, method='L-BFGS-B', 
                               options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 3000})
            except Exception:
                continue
            
            # Phase 2: Polish with max-norm (nonsmooth but captures exactness)
            if res1.fun < 1e-4:
                try:
                    res2 = minimize(objective_maxnorm, res1.x, method='L-BFGS-B', 
                                   options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 2000})
                except Exception:
                    res2 = res1
            else:
                res2 = res1
            
            if res2.fun < 1e-6:
                U_opt = res2.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_opt = res2.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_opt = res2.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                U_list = [[float(x) for x in row] for row in U_opt]
                V_list = [[float(x) for x in row] for row in V_opt]
                W_list = [[float(x) for x in row] for row in W_opt]
                
                # Try snapping to half-integers
                U_snap = [[snap(v) for v in row] for row in U_list]
                V_snap = [[snap(v) for v in row] for row in V_list]
                W_snap = [[snap(v) for v in row] for row in W_list]
                
                if _verify(U_snap, V_snap, W_snap, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_snap, V_snap, W_snap
                    found = True
                    break
                elif _verify(U_list, V_list, W_list, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_list, V_list, W_list
                    found = True
                    break
        
        if found:
            pass  # Continue to lower ranks
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
