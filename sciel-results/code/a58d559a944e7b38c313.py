import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x9 matrix.
    
    Returns: dict with 'U', 'V', 'W' factor matrices
    """
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Rank 3 decomposition based on Kronecker product of M_{2,1,3} (rank 1) and M_{1,3,3} (rank 3)
    rank = 3
    U = np.zeros((n * m, rank), dtype=np.int32)
    V = np.zeros((m * p, rank), dtype=np.int32)
    W = np.zeros((p * n, rank), dtype=np.int32)
    
    # U[i*m + j, r] = 1 if j == r else 0
    for i in range(n):
        for j in range(m):
            U[i * m + j, j] = 1
            
    # V[j*p + k, r] = 1 if j == r else 0
    for j in range(m):
        for k in range(p):
            V[j * p + k, j] = 1
            
    # W[k*n + i, r] = 1 for all k, i, r
    for k in range(p):
        for i in range(n):
            for r in range(rank):
                W[k * n + i, r] = 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
