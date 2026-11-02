"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    def trivial_decomposition():
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
    
    best = trivial_decomposition()
    
    # Try to find lower rank via ALS
    for target_rank in [40, 38, 36, 34, 32, 30, 28, 26, 24]:
        if time.time() > deadline - 20:
            break
        
        result = _als_search(n, m, p, target_rank, deadline)
        if result is not None:
            best = result
            break
    
    return best

def _als_search(n, m, p, target_rank, deadline):
    """Use ALS to find a low-rank decomposition of the matrix multiplication tensor."""
    np.random.seed(42)
    
    # Initialize
    U = np.random.randn(n * m, target_rank) * 0.5
    V = np.random.randn(m * p, target_rank) * 0.5
    W = np.random.randn(p * n, target_rank) * 0.5
    
    max_iter = 50
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Update U: for each (i,j), solve least squares
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                M = np.zeros((p, target_rank))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    U[idx, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V: for each (j,k), solve least squares
        for j in range(m):
            for k in range(p):
                idx = j * p + k
                M = np.zeros((n, target_rank))
                for i in range(n):
                    M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    V[idx, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W: for each (k,i), solve least squares
        for k in range(p):
            for i in range(n):
                idx = k * n + i
                M = np.zeros((m, target_rank))
                for j in range(m):
                    M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    W[idx, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass
    
    # Check error
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                max_err = max(max_err, abs(val - 1.0))
    
    if max_err < 1e-5:
        # Try rounding to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        max_err_r = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_r[i * m + j, :], V_r[j * p + k, :] * W_r[k * n + i, :])
                    max_err_r = max(max_err_r, abs(val - 1.0))
        
        if max_err_r < 1e-9:
            return {
                "U": U_r.tolist(),
                "V": V_r.tolist(),
                "W": W_r.tolist()
            }
    
    return None
# EVOLVE-BLOCK-END
