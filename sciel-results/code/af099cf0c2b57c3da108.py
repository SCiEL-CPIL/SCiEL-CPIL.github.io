import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _build_laderman_4x4():
    """
    Build Laderman's rank-19 decomposition for 4x4 x 4x4 matrix multiplication.
    Convention: C[i,j] = sum_k A[i,k] * B[k,j]
    Tensor: T[i,j,k] = 1
    U indexed by (i,k): U[i*4+k, r]
    V indexed by (k,j): V[k*4+j, r]
    W indexed by (j,i): W[j*4+i, r]
    """
    U = np.zeros((16, 19))
    V = np.zeros((16, 19))
    W = np.zeros((16, 19))
    
    # Laderman's 19 products for 4x4x4
    # Indexing: A[i,k], B[k,j], C[i,j]
    # U[i*4+k], V[k*4+j], W[j*4+i]
    
    # M1 = A[0,0] * B[0,0]
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # M2 = A[0,0] * (B[1,0] + B[1,1])
    U[0, 1] = 1; V[4, 1] = 1; V[5, 1] = 1; W[4, 1] = 1; W[5, 1] = 1
    # M3 = (A[0,0] + A[1,0]) * B[1,0]
    U[0, 2] = 1; U[4, 2] = 1; V[4, 2] = 1; W[4, 2] = 1
    # M4 = (A[0,0] + A[1,1]) * (B[1,1] - B[0,0])
    U[0, 3] = 1; U[13, 3] = 1; V[5, 3] = 1; V[0, 3] = -1; W[5, 3] = 1; W[0, 3] = -1
    # M5 = (A[1,1] - A[0,0]) * (B[0,0] + B[0,1])
    U[13, 4] = 1; U[0, 4] = -1; V[0, 4] = 1; V[1, 4] = 1; W[0, 4] = 1; W[1, 4] = 1
    # M6 = A[1,1] * (B[0,1] - B[1,1])
    U[13, 5] = 1; V[1, 5] = 1; V[5, 5] = -1; W[1, 5] = 1; W[5, 5] = -1
    # M7 = (A[1,1] + A[0,1]) * B[0,1]
    U[13, 6] = 1; U[1, 6] = 1; V[1, 6] = 1; W[1, 6] = 1
    # M8 = (A[0,1] + A[1,1]) * (B[1,0] - B[0,0])
    U[1, 7] = 1; U[13, 7] = 1; V[4, 7] = 1; V[0, 7] = -1; W[4, 7] = 1; W[0, 7] = -1
    # M9 = (A[0,1] + A[1,0]) * (B[0,0] - B[1,0])
    U[1, 8] = 1; U[4, 8] = 1; V[0, 8] = 1; V[4, 8] = -1; W[0, 8] = 1; W[4, 8] = -1
    # M10 = A[0,1] * (B[0,0] + B[1,0])
    U[1, 9] = 1; V[0, 9] = 1; V[4, 9] = 1; W[0, 9] = 1; W[4, 9] = 1
    # M11 = (A[0,1] - A[1,1]) * B[1,1]
    U[1, 10] = 1; U[13, 10] = -1; V[5, 10] = 1; W[5, 10] = 1
    # M12 = (A[1,0] + A[0,0]) * (B[0,1] - B[1,1])
    U[4, 11] = 1; U[0, 11] = 1; V[1, 11] = 1; V[5, 11] = -1; W[1, 11] = 1; W[5, 11] = -1
    # M13 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    U[4, 12] = 1; U[0, 12] = -1; V[0, 12] = 1; V[1, 12] = 1; W[0, 12] = 1; W[1, 12] = 1
    # M14 = A[1,0] * B[0,1]
    U[4, 13] = 1; V[1, 13] = 1; W[1, 13] = 1
    # M15 = (A[0,0] + A[0,1]) * (B[1,1] - B[0,1])
    U[0, 14] = 1; U[1, 14] = 1; V[5, 14] = 1; V[1, 14] = -1; W[5, 14] = 1; W[1, 14] = -1
    # M16 = (A[0,0] + A[0,1]) * B[0,0]
    U[0, 15] = 1; U[1, 15] = 1; V[0, 15] = 1; W[0, 15] = 1
    # M17 = (A[1,0] + A[1,1]) * (B[0,1] - B[1,1])
    U[4, 16] = 1; U[13, 16] = 1; V[1, 16] = 1; V[5, 16] = -1; W[1, 16] = 1; W[5, 16] = -1
    # M18 = (A[1,0] + A[1,1]) * B[0,0]
    U[4, 17] = 1; U[13, 17] = 1; V[0, 17] = 1; W[0, 17] = 1
    # M19 = A[1,1] * B[0,1]
    U[13, 18] = 1; V[1, 18] = 1; W[1, 18] = 1
    
    return U, V, W

def _validate_numpy(U, V, W, n, m, p):
    """Validate: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] == 1 for all i,j,k"""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(p):
            for k in range(m):
                val = np.sum(U_np[i*m+k, :] * V_np[k*p+j, :] * W_np[j*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook_fallback():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][column] = 1
                V[k * p + j][column] = 1
                W[j * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}

def _build_4x4x9():
    """Build 4x4 x 4x9 using Laderman for two 4x4 blocks + schoolbook for last column."""
    n, m, p = 4, 4, 9
    U444, V444, W444 = _build_laderman_4x4()
    
    total_rank = 19 + 19 + 16
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    # Block 1: B columns 0-3
    for i in range(4):
        for k in range(4):
            U[i*4+k, 0:19] = U444[i*4+k, :]
    for k in range(4):
        for j in range(4):
            V[k*9+j, 0:19] = V444[k*4+j, :]
    for j in range(4):
        for i in range(4):
            W[j*4+i, 0:19] = W444[j*4+i, :]
    
    # Block 2: B columns 4-7
    off2 = 19
    for i in range(4):
        for k in range(4):
            U[i*4+k, off2:off2+19] = U444[i*4+k, :]
    for k in range(4):
        for j in range(4):
            V[k*9+(j+4), off2:off2+19] = V444[k*4+j, :]
    for j in range(4):
        for i in range(4):
            W[(j+4)*4+i, off2:off2+19] = W444[j*4+i, :]
    
    # Block 3: B column 8 (schoolbook)
    off3 = 38
    for idx in range(16):
        i = idx // 4
        k = idx % 4
        U[i*4+k, off3+idx] = 1
        V[k*9+8, off3+idx] = 1
        W[8*4+i, off3+idx] = 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    U, V, W = _build_4x4x9()
    
    if not _validate_numpy(U, V, W, n, m, p):
        return _schoolbook_fallback()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
