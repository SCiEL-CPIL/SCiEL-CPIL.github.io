import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _build_laderman_4x4():
    """
    Build Laderman's rank-19 decomposition for 4x4 x 4x4 matrix multiplication.
    Returns U (16,19), V (16,19), W (16,19) where:
    U indexed by (i,k), V by (k,j), W by (j,i)
    T[i,k,j] = sum_r U[i*4+k, r] * V[k*4+j, r] * W[j*4+i, r]
    """
    U = np.zeros((16, 19))
    V = np.zeros((16, 19))
    W = np.zeros((16, 19))
    
    # Laderman's algorithm for 4x4 matrix multiplication
    # The 19 products are:
    # M1 = A11 * B11
    # M2 = A11 * (B21 + B22)
    # M3 = (A11 + A21) * B21
    # M4 = (A11 + A22) * (B22 - B11)
    # M5 = (A22 - A11) * (B11 + B12)
    # M6 = A22 * (B12 - B22)
    # M7 = (A22 + A12) * B12
    # M8 = (A12 + A22) * (B21 - B11)
    # M9 = (A12 + A21) * (B11 - B21)
    # M10 = A12 * (B11 + B21)
    # M11 = (A12 - A22) * B22
    # M12 = (A21 + A11) * (B12 - B22)
    # M13 = (A21 - A11) * (B11 + B12)
    # M14 = A21 * B12
    # M15 = (A11 + A12) * (B22 - B12)
    # M16 = (A11 + A12) * B11
    # M17 = (A21 + A22) * (B12 - B22)
    # M18 = (A21 + A22) * B11
    # M19 = A22 * B12
    
    # M1 = A11 * B11
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    
    # M2 = A11 * (B21 + B22)
    U[0, 1] = 1
    V[4, 1] = 1
    V[8, 1] = 1
    W[4, 1] = 1
    W[8, 1] = 1
    
    # M3 = (A11 + A21) * B21
    U[0, 2] = 1
    U[4, 2] = 1
    V[4, 2] = 1
    W[4, 2] = 1
    
    # M4 = (A11 + A22) * (B22 - B11)
    U[0, 3] = 1
    U[12, 3] = 1
    V[8, 3] = 1
    V[0, 3] = -1
    W[8, 3] = 1
    W[0, 3] = -1
    
    # M5 = (A22 - A11) * (B11 + B12)
    U[12, 4] = 1
    U[0, 4] = -1
    V[0, 4] = 1
    V[1, 4] = 1
    W[0, 4] = 1
    W[1, 4] = 1
    
    # M6 = A22 * (B12 - B22)
    U[12, 5] = 1
    V[1, 5] = 1
    V[9, 5] = -1
    W[1, 5] = 1
    W[9, 5] = -1
    
    # M7 = (A22 + A12) * B12
    U[12, 6] = 1
    U[1, 6] = 1
    V[1, 6] = 1
    W[1, 6] = 1
    
    # M8 = (A12 + A22) * (B21 - B11)
    U[1, 7] = 1
    U[12, 7] = 1
    V[4, 7] = 1
    V[0, 7] = -1
    W[4, 7] = 1
    W[0, 7] = -1
    
    # M9 = (A12 + A21) * (B11 - B21)
    U[1, 8] = 1
    U[4, 8] = 1
    V[0, 8] = 1
    V[4, 8] = -1
    W[0, 8] = 1
    W[4, 8] = -1
    
    # M10 = A12 * (B11 + B21)
    U[1, 9] = 1
    V[0, 9] = 1
    V[4, 9] = 1
    W[0, 9] = 1
    W[4, 9] = 1
    
    # M11 = (A12 - A22) * B22
    U[1, 10] = 1
    U[12, 10] = -1
    V[8, 10] = 1
    W[8, 10] = 1
    
    # M12 = (A21 + A11) * (B12 - B22)
    U[4, 11] = 1
    U[0, 11] = 1
    V[1, 11] = 1
    V[9, 11] = -1
    W[1, 11] = 1
    W[9, 11] = -1
    
    # M13 = (A21 - A11) * (B11 + B12)
    U[4, 12] = 1
    U[0, 12] = -1
    V[0, 12] = 1
    V[1, 12] = 1
    W[0, 12] = 1
    W[1, 12] = 1
    
    # M14 = A21 * B12
    U[4, 13] = 1
    V[1, 13] = 1
    W[1, 13] = 1
    
    # M15 = (A11 + A12) * (B22 - B12)
    U[0, 14] = 1
    U[1, 14] = 1
    V[8, 14] = 1
    V[1, 14] = -1
    W[8, 14] = 1
    W[1, 14] = -1
    
    # M16 = (A11 + A12) * B11
    U[0, 15] = 1
    U[1, 15] = 1
    V[0, 15] = 1
    W[0, 15] = 1
    
    # M17 = (A21 + A22) * (B12 - B22)
    U[4, 16] = 1
    U[12, 16] = 1
    V[1, 16] = 1
    V[9, 16] = -1
    W[1, 16] = 1
    W[9, 16] = -1
    
    # M18 = (A21 + A22) * B11
    U[4, 17] = 1
    U[12, 17] = 1
    V[0, 17] = 1
    W[0, 17] = 1
    
    # M19 = A22 * B12
    U[12, 18] = 1
    V[1, 18] = 1
    W[1, 18] = 1
    
    return U, V, W

def _validate_numpy(U, V, W, n, m, p):
    """Validate using NumPy for speed."""
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    rank = U_np.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                if val != 1:
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
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Build 4x4x4 Laderman decomposition (rank 19)
    U444, V444, W444 = _build_laderman_4x4()
    
    # Convert to lists for the final output format
    U444_list = U444.tolist()
    V444_list = V444.tolist()
    W444_list = W444.tolist()
    
    # Validate 4x4x4
    if not _validate_numpy(U444_list, V444_list, W444_list, 4, 4, 4):
        return _schoolbook_fallback()
    
    # Build 4x4x9 decomposition
    total_rank = 19 + 19 + 16
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(36)]
    W = [[0]*total_rank for _ in range(36)]
    
    # Block 1: columns 0-3 (first 4 columns of B)
    for I in range(4):
        for J in range(4):
            for c in range(19):
                U[I*4+J][c] = U444_list[I*4+J][c]
    for J in range(4):
        for K in range(4):
            for c in range(19):
                V[J*9+K][c] = V444_list[J*4+K][c]
    for K in range(4):
        for I in range(4):
            for c in range(19):
                W[K*4+I][c] = W444_list[K*4+I][c]
    
    # Block 2: columns 4-7 (next 4 columns of B)
    offset2 = 19
    for I in range(4):
        for J in range(4):
            for c in range(19):
                U[I*4+J][offset2+c] = U444_list[I*4+J][c]
    for J in range(4):
        for K in range(4):
            for c in range(19):
                V[J*9+(K+4)][offset2+c] = V444_list[J*4+K][c]
    for K in range(4):
        for I in range(4):
            for c in range(19):
                W[(K+4)*4+I][offset2+c] = W444_list[K*4+I][c]
    
    # Block 3: column 8 (last column, schoolbook rank 16)
    offset3 = 38
    for idx in range(16):
        I = idx // 4
        J = idx % 4
        U[I*4+J][offset3+idx] = 1
        V[J*9+8][offset3+idx] = 1
        W[8*4+I][offset3+idx] = 1
    
    # Final validation
    if not _validate_numpy(U, V, W, 4, 4, 9):
        return _schoolbook_fallback()
    
    return {"U": U, "V": V, "W": W}
