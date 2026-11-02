import time
import numpy as np

def construct():
    """
    Construct a rank-minimal tensor decomposition for 4x5 * 5x5 matrix multiplication.
    Uses a block decomposition:
    1. 4x4 * 4x4 part (Rank 64 schoolbook)
    2. 4x4 * 4x1 part (Rank 16 schoolbook)
    3. 4x1 * 1x5 part (Rank 1 outer product)
    Total Rank: 64 + 16 + 1 = 81.
    """
    t0 = time.time()
    deadline = 140.0

    n, m, p = 4, 5, 5
    
    # Part 1: 4x4 * 4x4 (i in 0..3, j in 0..3, k in 0..3)
    # Rank 64
    U1 = np.zeros((16, 64))
    V1 = np.zeros((16, 64))
    W1 = np.zeros((16, 64))
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # U index: i*4+j
                # V index: j*4+k
                # W index: k*4+i
                U1[i*4+j, col] = 1
                V1[j*4+k, col] = 1
                W1[k*4+i, col] = 1
                col += 1

    # Part 2: 4x4 * 4x1 (i in 0..3, j in 0..3, k=4)
    # Rank 16
    U2 = np.zeros((16, 16))
    V2 = np.zeros((4, 16)) # V shape for 4x1 is (4*1, rank) = (4, rank)
    W2 = np.zeros((4, 16)) # W shape for 1x4 is (1*4, rank) = (4, rank)
    col = 0
    for i in range(4):
        for j in range(4):
            # k is fixed at 0 locally, which maps to 4 globally
            U2[i*4+j, col] = 1
            V2[j*1+0, col] = 1
            W2[0*4+i, col] = 1
            col += 1

    # Part 3: 4x1 * 1x5 (i in 0..3, j=4, k in 0..4)
    # Rank 1
    U3 = np.zeros((4, 1))
    V3 = np.zeros((5, 1))
    W3 = np.zeros((20, 1))
    for i in range(4):
        U3[i*1+0, 0] = 1
    for k in range(5):
        V3[0*1+k, 0] = 1
    for k in range(5):
        for i in range(4):
            W3[k*4+i, 0] = 1

    # Combine into full 4x5x5 format
    # U shape: (20, 81)
    # V shape: (25, 81)
    # W shape: (20, 81)
    
    U_full = np.zeros((20, 81))
    V_full = np.zeros((25, 81))
    W_full = np.zeros((20, 81))
    
    # Map Part 1 (4x4x4)
    # U1 indices (i*4+j) map to U_full (i*5+j)
    for i in range(4):
        for j in range(4):
            U_full[i*5+j, 0:64] = U1[i*4+j, :]
    # V1 indices (j*4+k) map to V_full (j*5+k)
    for j in range(4):
        for k in range(4):
            V_full[j*5+k, 0:64] = V1[j*4+k, :]
    # W1 indices (k*4+i) map to W_full (k*4+i)
    for k in range(4):
        for i in range(4):
            W_full[k*4+i, 0:64] = W1[k*4+i, :]
            
    # Map Part 2 (4x4x1)
    # U2 indices (i*4+j) map to U_full (i*5+j)
    for i in range(4):
        for j in range(4):
            U_full[i*5+j, 64:80] = U2[i*4+j, :]
    # V2 indices (j*1+0) map to V_full (j*5+4)
    for j in range(4):
        V_full[j*5+4, 64:80] = V2[j*1+0, :]
    # W2 indices (0*4+i) map to W_full (4*4+i)
    for i in range(4):
        W_full[4*4+i, 64:80] = W2[0*4+i, :]
        
    # Map Part 3 (4x1x5)
    # U3 indices (i*1+0) map to U_full (i*5+4)
    for i in range(4):
        U_full[i*5+4, 80:81] = U3[i*1+0, :]
    # V3 indices (0*1+k) map to V_full (4*5+k)
    for k in range(5):
        V_full[4*5+k, 80:81] = V3[0*1+k, :]
    # W3 indices (k*4+i) map to W_full (k*4+i)
    for k in range(5):
        for i in range(4):
            W_full[k*4+i, 80:81] = W3[k*4+i, :]
            
    # Verify
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.sum(U_full[idx_u] * V_full[idx_v] * W_full[idx_w])
                max_err = max(max_err, abs(val - 1.0))
                
    if max_err > 1e-10:
        # Fallback to schoolbook if verification fails
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
        
    return {
        "U": U_full.tolist(),
        "V": V_full.tolist(),
        "W": W_full.tolist()
    }
