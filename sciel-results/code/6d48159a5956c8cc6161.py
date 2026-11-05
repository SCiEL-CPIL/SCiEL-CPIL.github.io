"""Program for 3x3 * 3x8 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]


def construct():
    """
    Find a low-rank exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    
    Strategy: Use the block-circulant embedding technique.
    - Embed the 3x3 matrix A into a 4x4 matrix by padding with a zero row/column.
    - For the first 4 columns of B (which are 3x4), use a rank-16 Strassen-like decomposition
      for the embedded 4x4 * 4x4 problem, projected back.
    - For the remaining 4 columns, use schoolbook or another low-rank scheme.
    
    Actually, let me use a cleaner approach:
    - Split B into B1 (3x4) and B2 (3x4).
    - For each 3x4 block, use a rank-12 decomposition (3x4 schoolbook is 12, but we can do better).
    
    Let me use the known result: 3x3 * 3x3 has rank 8 (Winograd).
    For 3x3 * 3x4, we can use rank 10 or 11.
    
    Let me just use a verified rank-8 for the 3x3 part and schoolbook for the rest.
    
    Actually, the simplest correct approach with good rank:
    - C = A * B where A is 3x3, B is 3x8
    - Split B into B[:,0:3] and B[:,3:8]
    - C[:,0:3] = A * B[:,0:3] (3x3 * 3x3) -> use rank-8 Winograd
    - C[:,3:8] = A * B[:,3:8] (3x3 * 3x5) -> use schoolbook rank 15
    - Total rank: 8 + 15 = 23
    
    Or split into B[:,0:4] and B[:,4:8]:
    - C[:,0:4] = A * B[:,0:4] (3x3 * 3x4) -> rank 12 schoolbook or better
    - C[:,4:8] = A * B[:,4:8] (3x3 * 3x4) -> rank 12 schoolbook or better
    - Total rank: 24
    
    Let me use the 8 + 15 = 23 approach.
    
    For the rank-8 Winograd 3x3*3x3, I need to carefully construct the U, V, W factors.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=8
    deadline = time.time() + 100  # Safety margin
    
    # Build the rank-8 Winograd decomposition for 3x3 * 3x3
    # Using the standard Winograd scheme.
    
    # The 8 multipliers:
    # M1 = (a00-a11-a22)(b11+b22)
    # M2 = (a11-a22)(b22-b11)
    # M3 = (a22-a11)(b11-b22)
    # M4 = a01(b10-b20)
    # M5 = a02(b20-b10)
    # M6 = (a10-a20)b01
    # M7 = (a20-a10)b02
    # M8 = a21(b02-b12)
    
    # Let me verify this is correct by constructing the full decomposition.
    
    # U factors (for A, 9x8 matrix)
    # V factors (for B, 9x8 matrix)
    # W factors (for C, 9x8 matrix)
    
    # Index convention:
    # U[i*m+j, r] for A[i][j]
    # V[j*p+k, r] for B[j][k]
    # W[k*n+i, r] for C[i][k]
    
    # For the 3x3*3x3 part (columns 0,1,2 of B):
    # U part: 9x8
    # V part: 9x8 (but only columns 0,1,2 of B are used, so V is 9x8 with non-zero in rows 0-8)
    # W part: 9x8 (columns 0,1,2 of C)
    
    U8 = np.zeros((9, 8))
    V8 = np.zeros((9, 8))
    W8 = np.zeros((9, 8))
    
    # Term 1: (a00-a11-a22)(b11+b22)
    U8[0, 0] = 1   # a00
    U8[4, 0] = -1  # a11
    U8[8, 0] = -1  # a22
    V8[4, 0] = 1   # b11
    V8[8, 0] = 1   # b22
    # W: C[i][k] contributions
    # C[0][1] += 1*1*M1 = M1 -> W[1*3+0, 0] = 1
    # C[0][2] += 1*1*M1 = M1 -> W[2*3+0, 0] = 1
    # C[1][1] += (-1)*1*M1 = -M1 -> W[1*3+1, 0] = -1
    # C[1][2] += (-1)*1*M1 = -M1 -> W[2*3+1, 0] = -1
    # C[2][1] += (-1)*1*M1 = -M1 -> W[1*3+2, 0] = -1
    # C[2][2] += (-1)*1*M1 = -M1 -> W[2*3+2, 0] = -1
    W8[3, 0] = 1   # C[0][1]
    W8[6, 0] = 1   # C[0][2]
    W8[4, 0] = -1  # C[1][1]
    W8[7, 0] = -1  # C[1][2]
    W8[5, 0] = -1  # C[2][1]
    W8[8, 0] = -1  # C[2][2]
    
    # Term 2: (a11-a22)(b22-b11)
    U8[4, 1] = 1   # a11
    U8[8, 1] = -1  # a22
    V8[8, 1] = 1   # b22
    V8[4, 1] = -1  # b11
    # W:
    # C[1][1] += 1*(-1)*M2 = -M2 -> W[1*3+1, 1] = -1
    # C[1][2] += 1*1*M2 = M2 -> W[2*3+1, 1] = 1
    # C[2][1] += (-1)*(-1)*M2 = M2 -> W[1*3+2, 1] = 1
    # C[2][2] += (-1)*1*M2 = -M2 -> W[2*3+2, 1] = -1
    W8[4, 1] = -1  # C[1][1]
    W8[7, 1] = 1   # C[1][2]
    W8[5, 1] = 1   # C[2][1]
    W8[8, 1] = -1  # C[2][2]
    
    # Term 3: (a22-a11)(b11-b22)
    U8[8, 2] = 1   # a22
    U8[4, 2] = -1  # a11
    V8[4, 2] = 1   # b11
    V8[8, 2] = -1  # b22
    # W:
    # C[1][1] += (-1)*1*M3 = -M3 -> W[1*3+1, 2] = -1
    # C[1][2] += (-1)*(-1)*M3 = M3 -> W[2*3+1, 2] = 1
    # C[2][1] += 1*1*M3 = M3 -> W[1*3+2, 2] = 1
    # C[2][2] += 1*(-1)*M3 = -M3 -> W[2*3+2, 2] = -1
    W8[4, 2] = -1  # C[1][1]
    W8[7, 2] = 1   # C[1][2]
    W8[5, 2] = 1   # C[2][1]
    W8[8, 2] = -1  # C[2][2]
    
    # Term 4: a01(b10-b20)
    U8[1, 3] = 1   # a01
    V8[3, 3] = 1   # b10
    V8[6, 3] = -1  # b20
    # W:
    # C[0][0] += 1*1*M4 = M4 -> W[0*3+0, 3] = 1
    # C[0][0] += 1*(-1)*M4 = -M4 -> already accounted
    W8[0, 3] = 1   # C[0][0]
    
    # Term 5: a02(b20-b10)
    U8[2, 4] = 1   # a02
    V8[6, 4] = 1   # b20
    V8[3, 4] = -1  # b10
    # W:
    # C[0][0] += 1*1*M5 = M5 -> W[0*3+0, 4] = 1
    W8[0, 4] = 1   # C[0][0]
    
    # Term 6: (a10-a20)b01
    U8[3, 5] = 1   # a10
    U8[6, 5] = -1  # a20
    V8[1, 5] = 1   # b01
    # W:
    # C[1][1] += 1*1*M6 = M6 -> W[1*3+1, 5] = 1
    # C[2][1] += (-1)*1*M6 = -M6 -> W[1*3+2, 5] = -1
    W8[4, 5] = 1   # C[1][1]
    W8[5, 5] = -1  # C[2][1]
    
    # Term 7: (a20-a10)b02
    U8[6, 6] = 1   # a20
    U8[3, 6] = -1  # a10
    V8[2, 6] = 1   # b02
    # W:
    # C[2][2] += 1*1*M7 = M7 -> W[2*3+2, 6] = 1
    # C[1][2] += (-1)*1*M7 = -M7 -> W[2*3+1, 6] = -1
    W8[8, 6] = 1   # C[2][2]
    W8[7, 6] = -1  # C[1][2]
    
    # Term 8: a21(b02-b12)
    U8[7, 7] = 1   # a21
    V8[2, 7] = 1   # b02
    V8[5, 7] = -1  # b12
    # W:
    # C[2][2] += 1*1*M8 = M8 -> W[2*3+2, 7] = 1
    W8[8, 7] = 1   # C[2][2]
    
    # Now verify the 3x3*3x3 decomposition
    # Build the tensor and check
    T_expected = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                T_expected[i*3+j, j*3+k, k*3+i] = 1.0
    
    T_decomp = np.zeros((9, 9, 9))
    for r in range(8):
        for idx_u in range(9):
            for idx_v in range(9):
                for idx_w in range(9):
                    T_decomp[idx_u, idx_v, idx_w] += U8[idx_u, r] * V8[idx_v, r] * W8[idx_w, r]
    
    error = np.max(np.abs(T_decomp - T_expected))
    
    if error > 1e-10:
        # If the Winograd decomposition is wrong, fall back to schoolbook
        # For now, just use schoolbook for everything
        rank = n * m * p  # 72
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # The 3x3*3x3 part works. Now add the schoolbook part for columns 3-7.
    # Schoolbook for 3x3 * 3x5: rank 15
    
    rank_total = 8 + 15  # 23
    U_total = np.zeros((9, rank_total))
    V_total = np.zeros((9, rank_total))
    W_total = np.zeros((24, rank_total))  # p*n = 8*3 = 24
    
    # Copy the Winograd part (columns 0-2 of B, columns 0-2 of C)
    U_total[:, 0:8] = U8
    V_total[:, 0:8] = V8
    W_total[0:9, 0:8] = W8
    
    # Schoolbook part for columns 3-7
    col_offset = 8
    for i in range(3):
        for j in range(3):
            for k in range(3, 8):
                U_total[i*3+j, col_offset] = 1
                V_total[j*3+k, col_offset] = 1
                W_total[k*3+i, col_offset] = 1
                col_offset += 1
    
    # Verify the full decomposition
    # T_full has shape (9, 24, 24)
    # T_full[(i*3+j), (j*8+k), (k*3+i)] = 1 for i,j in 0..2, k in 0..7
    
    T_full_expected = np.zeros((9, 24, 24))
    for i in range(3):
        for j in range(3):
            for k in range(8):
                T_full_expected[i*3+j, j*8+k, k*3+i] = 1.0
    
    T_full_decomp = np.zeros((9, 24, 24))
    for r in range(rank_total):
        for idx_u in range(9):
            for idx_v in range(24):
                for idx_w in range(24):
                    T_full_decomp[idx_u, idx_v, idx_w] += U_total[idx_u, r] * V_total[idx_v, r] * W_total[idx_w, r]
    
    error_full = np.max(np.abs(T_full_decomp - T_full_expected))
    
    if error_full > 1e-10:
        # Fallback to schoolbook
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
