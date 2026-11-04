"""
Rewritten program for 3x4x5 matrix multiplication tensor decomposition.
Attempts to find a rank-59 decomposition using local optimization.
Falls back to the schoolbook rank-60 decomposition if optimization fails.
"""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 5]

def _schoolbook(n, m, p):
    """
    Standard schoolbook matrix multiplication tensor decomposition.
    Rank = n * m * p.
    """
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
    return {"U": U, "V": V, "W": W}

def _decomposition_error(U, V, W, n, m, p):
    """
    Computes the error of the decomposition.
    The condition is:
    For all i, k, j', j'':
    sum_c U[i*m+j', c] * V[j'*p+k, c] * W[k*n+i, c] == delta_{j', j''}
    """
    errors = []
    for i in range(n):
        for k in range(p):
            # W_idx is k*n + i
            w_vec = W[k * n + i, :]  # shape (rank,)
            for jp in range(m):
                for jq in range(m):
                    # We need sum_c U[i*m+jp, c] * V[jq*p+k, c] * W[k*n+i, c]
                    # Note: V index is j*p + k. Here j is the middle index.
                    # In the sum for C[i,k], we sum over j.
                    # The term is A[i,j] B[j,k].
                    # So U depends on i,j. V depends on j,k. W depends on i,k.
                    # The identity is sum_j A[i,j] B[j,k] = sum_c (sum_j U[i,j,c] A[i,j]) (sum_j V[j,k,c] B[j,k]) W[i,k,c]
                    # Expanding RHS: sum_c W[i,k,c] sum_j sum_l U[i,j,c] V[l,k,c] A[i,j] B[l,k]
                    # For this to match LHS sum_j A[i,j] B[j,k], we need:
                    # sum_c W[i,k,c] U[i,j,c] V[l,k,c] = delta_{j,l}
                    
                    u_vec = U[i * m + jp, :]
                    v_vec = V[jq * p + k, :]
                    
                    # sum_c u_vec[c] * v_vec[c] * w_vec[c]
                    val = np.sum(u_vec * v_vec * w_vec)
                    target = 1.0 if jp == jq else 0.0
                    errors.append(val - target)
    
    return np.array(errors)

def _try_find_rank59(n, m, p, time_limit=10.0):
    """
    Attempts to find a rank-59 decomposition using least squares.
    """
    rank = 59
    start_time = time.time()
    
    # Initial guess: Random
    np.random.seed(42)
    # Scale is important. Start with small random values.
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    def residual(params):
        # params is a flattened vector of U, V, W
        u_flat = params[:n*m*rank]
        v_flat = params[n*m*rank : n*m*rank + m*p*rank]
        w_flat = params[n*m*rank + m*p*rank : ]
        
        U = u_flat.reshape(n*m, rank)
        V = v_flat.reshape(m*p, rank)
        W = w_flat.reshape(p*n, rank)
        
        return _decomposition_error(U, V, W, n, m, p)
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    # Use least squares
    try:
        result = least_squares(residual, x0, method='lm', max_nfev=1000)
        
        if result.success or np.linalg.norm(result.fun) < 1e-8:
            # Reconstruct matrices
            U_sol = result.x[:n*m*rank].reshape(n*m, rank)
            V_sol = result.x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
            W_sol = result.x[n*m*rank + m*p*rank : ].reshape(p*n, rank)
            
            # Verify
            err = np.linalg.norm(_decomposition_error(U_sol, V_sol, W_sol, n, m, p))
            if err < 1e-8:
                return {"U": U_sol, "V": V_sol, "W": W_sol}
    except Exception:
        pass
        
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # 1. Try to find rank 59
    # We allocate a small amount of time for this search.
    # If it fails, we return schoolbook.
    best = _schoolbook(n, m, p)
    
    # Try rank 59 with a short timeout
    # Note: The optimization might not converge quickly from random init.
    # However, it is a "stronger" attempt.
    if time.time() - start_time < 5.0:
        res_59 = _try_find_rank59(n, m, p, time_limit=5.0)
        if res_59 is not None:
            best = res_59
            # We can try to polish or just return
            return best
            
    return best
