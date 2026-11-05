import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # For 3x3x3, use a known rank-23 decomposition
    # Actually, let's first try the naive approach and verify, then optimize
    
    # Naive rank-63 decomposition
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Verify
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > 1e-10:
                        return False
        return True
    
    if not verify(U, V, W):
        raise ValueError("Naive decomposition failed verification")
    
    # Try to reduce rank using the slice method
    # Partition columns into blocks
    # Block 1: k=0,1,2
    # Block 2: k=3,4,5  
    # Block 3: k=6
    
    # For each block of 3 columns, we need a 3x3x3 decomposition
    # For the single column, we need a 3x3x1 decomposition
    
    # Let's use a known 3x3x3 rank-23 decomposition
    # Actually, let me just return the naive for now and iterate
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
