"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def _schoolbook_decomp(n, m, p):
    """Return schoolbook decomposition for n x m * m x p -> n x p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _random_search(n, m, p, target_rank, num_trials=50, timeout=80):
    """Try to find a rank-target_rank decomposition via random init + ALS."""
    start = time.time()
    
    for trial in range(num_trials):
        if time.time() - start > timeout:
            break
        
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        for it in range(30):
            # Update W
            W_new = W.copy()
            for i in range(n):
                for k in range(p):
                    A_mat = np.zeros((m, target_rank))
                    for j in range(m):
                        A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W_new[k*n+i, :] = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)[0]
                    except:
                        pass
            W = W_new
            
            # Update U
            U_new = U.copy()
            for i in range(n):
                for j in range(m):
                    A_mat = np.zeros((p, target_rank))
                    for k in range(p):
                        A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U_new[i*m+j, :] = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)[0]
                    except:
                        pass
            U = U_new
            
            # Update V
            V_new = V.copy()
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, target_rank))
                    for i in range(n):
                        A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V_new[j*p+k, :] = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)[0]
                    except:
                        pass
            V = V_new
        
        # Check exactness
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    error += (val - 1.0)**2
        
        if error < 1e-8:
            # Round to half-integers
            U_r = np.round(U * 2) / 2.0
            V_r = np.round(V * 2) / 2.0
            W_r = np.round(W * 2) / 2.0
            
            err2 = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                        err2 += (val - 1.0)**2
            
            if err2 < 1e-8:
                return {
                    'U': U_r.tolist(),
                    'V': V_r.tolist(),
                    'W': W_r.tolist()
                }
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Try to find a lower-rank decomposition
    schoolbook_rank = n * m * p  # 96
    
    # Try ranks from 50 to 95
    for target_rank in range(50, schoolbook_rank):
        if time.time() > deadline:
            break
        result = _random_search(n, m, p, target_rank, num_trials=10, timeout=5)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook_decomp(n, m, p)
# EVOLVE-BLOCK-END
