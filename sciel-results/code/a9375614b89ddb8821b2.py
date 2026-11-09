import numpy as np
import time
import random

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Target rank to attempt
    target_rank = 17
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # 1. Attempt to find a rank-17 decomposition via numerical optimization
    # The tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We want U (nm x R), V (mp x R), W (pn x R) such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    def compute_tensor(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Result: (n, m, p)
        R = U.shape[1]
        T = np.zeros((n, m, p))
        # Efficiently compute using einsum or loops
        # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # Dot product of row u_idx of U, row v_idx of V, row w_idx of W
                    # This is element-wise product then sum
                    T[i, j, k] = np.dot(np.dot(U[u_idx, :], V[v_idx, :]), W[w_idx, :])
        return T

    def error_function(params):
        R = target_rank
        U = params[0:n*m*R].reshape(n*m, R)
        V = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W = params[n*m*R + m*p*R:].reshape(p*n, R)
        T = compute_tensor(U, V, W)
        # Target is all ones
        err = T - 1.0
        return np.sum(err**2)

    best_result = None
    
    # Try a few random initializations
    # To save time, limit iterations
    max_attempts = 5
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
            
        # Initialize with random values
        np.random.seed(42 + attempt)
        R = target_rank
        init_U = np.random.randn(n*m, R) * 0.1
        init_V = np.random.randn(m*p, R) * 0.1
        init_W = np.random.randn(p*n, R) * 0.1
        
        params = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
        
        # Use scipy.optimize.minimize
        try:
            from scipy.optimize import minimize
            # L-BFGS-B is good for large unconstrained problems
            res = minimize(error_function, params, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success or res.fun < 1e-8:
                # Extract matrices
                U_opt = res.x[0:n*m*R].reshape(n*m, R)
                V_opt = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_opt = res.x[n*m*R + m*p*R:].reshape(p*n, R)
                
                # Round to nearest integer or half-integer
                # The problem states "integer/half-integer entries"
                # Let's try rounding to nearest integer first, then half-integer
                U_int = np.round(U_opt).astype(int)
                V_int = np.round(V_opt).astype(int)
                W_int = np.round(W_opt).astype(int)
                
                T_int = compute_tensor(U_int, V_int, W_int)
                if np.allclose(T_int, 1.0, atol=1e-9):
                    best_result = (U_int, V_int, W_int, R)
                    break
                    
                U_half = (np.round(U_opt * 2) / 2)
                V_half = (np.round(V_opt * 2) / 2)
                W_half = (np.round(W_opt * 2) / 2)
                
                T_half = compute_tensor(U_half, V_half, W_half)
                if np.allclose(T_half, 1.0, atol=1e-9):
                    # Convert to list of lists for output
                    best_result = (U_half, V_half, W_half, R)
                    break
                    
        except Exception:
            pass

    if best_result is not None:
        U, V, W, R = best_result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
        
    # Fallback to Schoolbook Rank 18
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
# EVOLVE-BLOCK-END
