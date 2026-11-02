import numpy as np
import time
import itertools
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Indices:
    # U: (i, j) -> i*m + j
    # V: (j, k) -> j*p + k
    # W: (k, i) -> k*n + i
    
    # Target Tensor Indices
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Term: A[i, j] B[j, k] contributes to C[i, k]
    # Triplet: (i, j, k)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i (Task says: W[k*n+i,r])
    
    triplets = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Precompute indices for speed
    u_indices = np.array([u_idx(i, j) for i, j, k in triplets])
    v_indices = np.array([v_idx(j, k) for i, j, k in triplets])
    w_indices = np.array([w_idx(k, i) for i, j, k in triplets])
    
    def get_residuals(U, V, W):
        # U, V, W are 2D arrays of shape (dim, rank)
        # We need sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
        # This can be done efficiently
        # U_flat = U[u_indices] -> shape (num_triplets, rank)
        # V_flat = V[v_indices] -> shape (num_triplets, rank)
        # W_flat = W[w_indices] -> shape (num_triplets, rank)
        # Residual = sum_r (U_flat * V_flat * W_flat) - 1
        U_flat = U[u_indices]
        V_flat = V[v_indices]
        W_flat = W[w_indices]
        return np.sum(U_flat * V_flat * W_flat, axis=1) - 1.0

    def objective(params):
        # params is a flat vector
        # U: n*m * rank
        # V: m*p * rank
        # W: p*n * rank
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        
        U = params[0:size_U].reshape((n * m, rank))
        V = params[size_U:size_U + size_V].reshape((m * p, rank))
        W = params[size_U + size_V:].reshape((p * n, rank))
        
        res = get_residuals(U, V, W)
        return np.sum(res * res)

    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')
    
    start_time = time.time()
    deadline = 140 # 10s margin
    
    # Try ranks from 47 down to 40 (or lower if time permits)
    # Schoolbook is 48. We want < 48.
    for r in range(47, 39, -1):
        if time.time() > deadline:
            break
            
        rank = r
        # Multiple restarts
        num_restarts = 5 if r >= 45 else 10
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Initialize with random values
            # Scale matters. Let's try small random values.
            U_init = np.random.randn(n * m, rank) * 0.1
            V_init = np.random.randn(m * p, rank) * 0.1
            W_init = np.random.randn(p * n, rank) * 0.1
            
            # Alternative init: Schoolbook-like but perturbed
            # If r < 48, we can't use full schoolbook.
            # Let's stick to random for now, or use a structured init if possible.
            # For r close to 48, maybe we can start with 48 terms and merge?
            # Random is safer for generic search.
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(objective, params_init, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.success or res.fun < 1e-6:
                    # Extract U, V, W
                    size_U = n * m * rank
                    size_V = m * p * rank
                    size_W = p * n * rank
                    
                    U_opt = res.x[0:size_U].reshape((n * m, rank))
                    V_opt = res.x[size_U:size_U + size_V].reshape((m * p, rank))
                    W_opt = res.x[size_U + size_V:].reshape((p * n, rank))
                    
                    # Verify exactness
                    res_val = get_residuals(U_opt, V_opt, W_opt)
                    err = np.sum(res_val * res_val)
                    
                    if err < 1e-6:
                        # Found a valid decomposition of rank r
                        return {
                            "U": U_opt.tolist(),
                            "V": V_opt.tolist(),
                            "W": W_opt.tolist()
                        }
                    else:
                        if err < best_err:
                            best_err = err
                            best_U = U_opt
                            best_V = V_opt
                            best_W = W_opt
                            best_rank = r
            except Exception:
                pass
    
    # If no success, return the best found or fallback to schoolbook
    if best_U is not None and best_rank < n * m * p:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Fallback: Schoolbook Rank 48
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
