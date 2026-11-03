import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    n, m, p = 3, 3, 5
    deadline = time.time() + 140
    
    # Try ranks from low to high
    # Known upper bound is 45.
    # We try to find a rank < 45.
    # Ranks to try: 25, 28, 30, 32, 34, 36, 38, 40, 42, 44
    target_ranks = [25, 28, 30, 32, 34, 36, 38, 40, 42, 44]
    
    for rank in target_ranks:
        if time.time() > deadline - 5:
            break
        result = _try_rank_optimized(n, m, p, rank, deadline)
        if result is not None:
            return result
            
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_rank_optimized(n, m, p, rank, deadline):
    np.random.seed(42)
    
    # Dimensions
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Total variables
    total_vars = rank * (dim_u + dim_v + dim_w)
    
    def residuals(x):
        U = x[:rank*dim_u].reshape(rank, dim_u).T # rank x dim_u -> dim_u x rank
        V = x[rank*dim_u:rank*(dim_u+dim_v)].reshape(rank, dim_v).T
        W = x[rank*(dim_u+dim_v):].reshape(rank, dim_w).T
        
        # Compute tensor approximation
        # T_approx[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We need this to be 1 for all i, j, k.
        
        # Efficient computation:
        # For each c, U_c is a vector of length dim_u.
        # We can compute the sum for all i,j,k using broadcasting or loops.
        # Given small dimensions (3,3,5), a loop is fine.
        
        res = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                for k in range(p):
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # Sum over c
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    res[idx] = 1.0 - val
                    idx += 1
        return res
    
    best_solution = None
    best_cost = np.inf
    
    # Try multiple random initializations
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Random initialization
        x0 = np.random.randn(total_vars)
        # Scale to reasonable magnitude
        x0 /= np.linalg.norm(x0) * np.sqrt(total_vars)
        
        try:
            sol = least_squares(residuals, x0, method='lm', max_nfev=1000)
            if sol.success or sol.cost < 1e-8:
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_solution = sol.x
        except Exception:
            pass
            
    if best_solution is not None and best_cost < 1e-8:
        # Extract U, V, W
        U = best_solution[:rank*dim_u].reshape(rank, dim_u).T
        V = best_solution[rank*dim_u:rank*(dim_u+dim_v)].reshape(rank, dim_v).T
        W = best_solution[rank*(dim_u+dim_v):].reshape(rank, dim_w).T
        
        # Verify
        if _verify(U, V, W, n, m, p, rank):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    return None

def _verify(U, V, W, n, m, p, rank):
    tol = 1e-6
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True
