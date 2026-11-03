"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, tol=1e-6):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T_decomp = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                T_decomp[i,j,k] = s
    
    T_target = np.ones((n, m, p), dtype=float)
    return np.allclose(T_decomp, T_target, atol=tol)

def _try_rank_reduction():
    """
    Attempt to find a lower rank decomposition by searching for linear 
    dependencies or using structured Strassen-like decompositions.
    """
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook()
    best_rank = n * m * p
    best_result = schoolbook
    
    # Try to find a rank 59, 58, etc. decomposition
    # This is computationally expensive, so we limit the search.
    
    # For now, we return the schoolbook decomposition as it is guaranteed valid.
    # A future evolution might implement a specific Strassen-based 
    # decomposition for 3x4x4x5.
    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    # Try the rank reduction approach
    result = _try_rank_reduction()
    
    if _verify(result['U'], result['V'], result['W']):
        return result
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
