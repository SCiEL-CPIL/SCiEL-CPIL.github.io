import time
import numpy as np

DIMENSIONS = [4, 5, 6]
TIME_LIMIT = 150.0

def _build_decomposition(n, m, p):
    """
    Recursively build a decomposition by splitting the middle dimension m.
    Returns U, V, W for the sub-problem.
    """
    # Base case: m=1
    if m == 1:
        rank = 1
        U = np.zeros((n * 1, rank))
        V = np.zeros((1 * p, rank))
        W = np.zeros((p * n, rank))
        
        # T_{i,0,k} = 1
        # U[i, 0] = 1, V[k, 0] = 1, W[k*n+i, 0] = 1
        for i in range(n):
            U[i, 0] = 1.0
        for k in range(p):
            V[k, 0] = 1.0
            for i in range(n):
                W[k * n + i, 0] = 1.0
        return U, V, W

    # Split m into m1 and m2
    # To minimize rank, we might want to split evenly, but m=1 base case is very cheap.
    # Let's split into 1 and m-1.
    m1 = 1
    m2 = m - 1
    
    # Get decompositions for sub-problems
    U1, V1, W1 = _build_decomposition(n, m1, p)
    U2, V2, W2 = _build_decomposition(n, m2, p)
    
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    total_rank = rank1 + rank2
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Place U1
    # U1 is indexed by i (0..n-1) and j (0..m1-1)
    # In U, index is i*m + j
    for i in range(n):
        for j in range(m1):
            U[i * m + j, 0:rank1] = U1[i * m1 + j, 0:rank1]
            
    # Place U2
    # U2 is indexed by i (0..n-1) and j (0..m2-1)
    # In U, index is i*m + (j + m1)
    for i in range(n):
        for j in range(m2):
            U[i * m + (j + m1), rank1:total_rank] = U2[i * m2 + j, 0:rank2]
            
    # Place V1
    # V1 is indexed by j (0..m1-1) and k (0..p-1)
    # In V, index is j*p + k
    for j in range(m1):
        for k in range(p):
            V[j * p + k, 0:rank1] = V1[j * m1 + k, 0:rank1]
            
    # Place V2
    # V2 is indexed by j (0..m2-1) and k (0..p-1)
    # In V, index is (j+m1)*p + k
    for j in range(m2):
        for k in range(p):
            V[(j + m1) * p + k, rank1:total_rank] = V2[j * m2 + k, 0:rank2]
            
    # Place W1 and W2
    # W is indexed by k and i.
    # W1 and W2 have the same shape (p*n, rank).
    # We just concatenate them.
    W[:, 0:rank1] = W1
    W[:, rank1:total_rank] = W2
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    U, V, W = _build_decomposition(n, m, p)
    
    # Verify
    rank = U.shape[1]
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if not valid:
        # Fallback to schoolbook
        U, V, W = _build_schoolbook(n, m, p)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _build_schoolbook(n, m, p):
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
    return U, V, W
