import numpy as np
import time

def construct():
    """
    Rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    Uses Kronecker product of Strassen (2x2x2 rank-7).
    """
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin

    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 rank-7 decomposition
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0,-1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1,  0, 0, 1, 0],
        [0, 0, 0,  1, 0, 0, 1],
        [1, 0,-1,  0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1,  0, 0, 0],
        [0, 0, 1, 0,  1, 0, 0],
        [1,-1, 1, 0,  0, 1, 0]
    ], dtype=float)
    
    # Build 4x4x4 rank-49 decomposition via Kronecker product
    # U4 = U2 kron U2
    # V4 = V2 kron V2
    # W4 = W2 kron W2
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # The Kronecker product order matches the index ordering i*4+j, j*4+k, k*4+i
    # U2[i1*2+j1, t] * U2[i2*2+j2, s] -> index (i1*2+i2)*4 + (j1*2+j2)
    # np.kron(A, B) has element (i1*2+i2, j1*2+j2) = A[i1,j1]*B[i2,j2]
    # This matches the required indexing.
    
    # Build 4x4x8 rank-98 decomposition
    rank = 98
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # First block (output cols 0-3)
    U[:, 0:49] = U4
    # V indices: j*8+k. For k in 0..3, V[j*8+k] = V4[j*4+k]
    # We can reshape V4 to (4,4,49) and expand?
    # V4 is (16, 49). Index j*4+k.
    # We need to place this in V[:, 0:49] at rows j*8+k for k=0..3.
    # Rows 0,1,2,3 (j=0, k=0..3) -> V4[0..3]
    # Rows 8,9,10,11 (j=1, k=0..3) -> V4[4..7]
    # Rows 16,17,18,19 (j=2, k=0..3) -> V4[8..11]
    # Rows 24,25,26,27 (j=3, k=0..3) -> V4[12..15]
    
    for j in range(4):
        V[j*8:(j*8+4), 0:49] = V4[j*4:(j*4+4), :]
        
    # W indices: k*4+i. For k in 0..3, W[k*4+i] = W4[k*4+i]
    W[0:16, 0:49] = W4
    
    # Second block (output cols 4-7)
    U[:, 49:98] = U4
    for j in range(4):
        V[j*8+4:(j*8+8), 49:98] = V4[j*4:(j*4+4), :]
    W[16:32, 49:98] = W4
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct_naive(n, m, p):
    U = np.zeros((n * m, n * m * p), dtype=float)
    V = np.zeros((m * p, n * m * p), dtype=float)
    W = np.zeros((p * n, n * m * p), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1.0
                V[j*p+k, idx] = 1.0
                W[k*n+i, idx] = 1.0
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
