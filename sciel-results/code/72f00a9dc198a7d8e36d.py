import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

def check_decomposition(U, V, W, n, m, p):
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Check if sum_r U[i,j] V[j,k] W[k,i] == 1 for all i,j,k
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1) > 1e-9:
                    return False
    return True

def is_half_integer(x, tol=1e-6):
    return np.allclose(x, np.round(x * 2) / 2, atol=tol)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Naive solution
    rank_naive = n * m * p
    U_naive = np.zeros((n * m, rank_naive))
    V_naive = np.zeros((m * p, rank_naive))
    W_naive = np.zeros((p * n, rank_naive))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i * m + j, col] = 1
                V_naive[j * p + k, col] = 1
                W_naive[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_naive
    best_U = U_naive
    best_V = V_naive
    best_W = W_naive
    
    # Try to find lower rank
    # We search from rank_naive - 1 down to some limit
    for rank in range(rank_naive - 1, 20, -1):
        if time.time() > deadline - 10:
            break
            
        # Try a few random restarts for this rank
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Initialize parameters
            # Use small random values around 0
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Flatten parameters
            def loss(params):
                U_flat = params[0 : n*m*rank]
                V_flat = params[n*m*rank : n*m*rank + m*p*rank]
                W_flat = params[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank]
                
                U_mat = U_flat.reshape(n * m, rank)
                V_mat = V_flat.reshape(m * p, rank)
                W_mat = W_flat.reshape(p * n, rank)
                
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for r in range(rank):
                                val += U_mat[i * m + j, r] * V_mat[j * p + k, r] * W_mat[k * n + i, r]
                            error += (val - 1) ** 2
                return error
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Optimize
            # Use Nelder-Mead for small problems, or L-BFGS-B
            # Given the non-convexity, Nelder-Mead with random restarts is robust
            res = minimize(loss, params, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.success or res.fun < 1e-6:
                # Extract optimized parameters
                U_opt = res.x[0 : n*m*rank].reshape(n * m, rank)
                V_opt = res.x[n*m*rank : n*m*rank + m*p*rank].reshape(m * p, rank)
                W_opt = res.x[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank].reshape(p * n, rank)
                
                # Snap to half-integers
                U_snap = np.round(U_opt * 2) / 2
                V_snap = np.round(V_opt * 2) / 2
                W_snap = np.round(W_opt * 2) / 2
                
                # Check if snapped version is valid
                if check_decomposition(U_snap, V_snap, W_snap, n, m, p):
                    best_rank = rank
                    best_U = U_snap
                    best_V = V_snap
                    best_W = W_snap
                    break # Found a valid decomposition for this rank, move to lower rank
                    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
