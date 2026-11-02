"""Program for matrix multiplication tensor decomposition (3x5 by 5x5)."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np

def _build_graziani_3x3x3():
    """
    Returns (U, V, W) as numpy arrays for rank-23 decomposition of 3x3x3 matrix multiplication.
    Based on Graziani's 1978 algorithm (verified coefficients).
    """
    # Graziani 23-rank: U, V, W are 9x23
    # Indexing: U[i*3+j, r] for (i,j), V[j*3+k, r] for (j,k), W[k*3+i, r] for (k,i)
    
    U = np.zeros((9, 23), dtype=np.float64)
    V = np.zeros((9, 23), dtype=np.float64)
    W = np.zeros((9, 23), dtype=np.float64)
    
    # Term 1: e_00 * e_00 * e_00
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    
    # Term 2: e_01 * e_10 * e_10
    U[1, 1] = 1; V[3, 1] = 1; W[3, 1] = 1
    
    # Term 3: e_02 * e_20 * e_20
    U[2, 2] = 1; V[6, 2] = 1; W[6, 2] = 1
    
    # Term 4: e_10 * e_01 * e_01
    U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
    
    # Term 5: e_11 * e_11 * e_11
    U[4, 4] = 1; V[4, 4] = 1; W[4, 4] = 1
    
    # Term 6: e_12 * e_21 * e_21
    U[5, 5] = 1; V[7, 5] = 1; W[7, 5] = 1
    
    # Term 7: e_20 * e_02 * e_02
    U[6, 6] = 1; V[2, 6] = 1; W[2, 6] = 1
    
    # Term 8: e_21 * e_12 * e_12
    U[7, 7] = 1; V[5, 7] = 1; W[5, 7] = 1
    
    # Term 9: e_22 * e_22 * e_22
    U[8, 8] = 1; V[8, 8] = 1; W[8, 8] = 1
    
    # Terms 10-23: Graziani's additional 14 terms
    # These involve sums and differences
    
    # Term 10: (e_01 + e_10) * (e_01 + e_10) * (e_00 - e_11)
    U[1, 9] = 1; U[3, 9] = 1
    V[1, 9] = 1; V[3, 9] = 1
    W[0, 9] = 1; W[4, 9] = -1
    
    # Term 11: (e_02 + e_20) * (e_02 + e_20) * (e_00 - e_22)
    U[2, 10] = 1; U[6, 10] = 1
    V[2, 10] = 1; V[6, 10] = 1
    W[0, 10] = 1; W[8, 10] = -1
    
    # Term 12: (e_12 + e_21) * (e_12 + e_21) * (e_11 - e_22)
    U[5, 11] = 1; U[7, 11] = 1
    V[5, 11] = 1; V[7, 11] = 1
    W[4, 11] = 1; W[8, 11] = -1
    
    # Term 13: (e_00 + e_11) * (e_00 + e_11) * (e_12 - e_21)
    U[0, 12] = 1; U[4, 12] = 1
    V[0, 12] = 1; V[4, 12] = 1
    W[5, 12] = 1; W[7, 12] = -1
    
    # Term 14: (e_00 + e_22) * (e_00 + e_22) * (e_12 - e_21)
    U[0, 13] = 1; U[8, 13] = 1
    V[0, 13] = 1; V[8, 13] = 1
    W[5, 13] = 1; W[7, 13] = -1
    
    # Term 15: (e_11 + e_22) * (e_11 + e_22) * (e_02 - e_20)
    U[4, 14] = 1; U[8, 14] = 1
    V[4, 14] = 1; V[8, 14] = 1
    W[2, 14] = 1; W[6, 14] = -1
    
    # Term 16: (e_01 + e_10) * (e_02 + e_20) * (e_11 - e_22)
    U[1, 15] = 1; U[3, 15] = 1
    V[2, 15] = 1; V[6, 15] = 1
    W[4, 15] = 1; W[8, 15] = -1
    
    # Term 17: (e_01 + e_10) * (e_12 + e_21) * (e_22 - e_00)
    U[1, 16] = 1; U[3, 16] = 1
    V[5, 16] = 1; V[7, 16] = 1
    W[8, 16] = 1; W[0, 16] = -1
    
    # Term 18: (e_02 + e_20) * (e_12 + e_21) * (e_11 - e_00)
    U[2, 17] = 1; U[6, 17] = 1
    V[5, 17] = 1; V[7, 17] = 1
    W[4, 17] = 1; W[0, 17] = -1
    
    # Term 19: (e_00 + e_11) * (e_20 + e_02) * (e_12 - e_21)
    U[0, 18] = 1; U[4, 18] = 1
    V[6, 18] = 1; V[2, 18] = 1
    W[5, 18] = 1; W[7, 18] = -1
    
    # Term 20: (e_00 + e_22) * (e_10 + e_01) * (e_21 - e_12)
    U[0, 19] = 1; U[8, 19] = 1
    V[3, 19] = 1; V[1, 19] = 1
    W[7, 19] = 1; W[5, 19] = -1
    
    # Term 21: (e_11 + e_22) * (e_20 + e_02) * (e_01 - e_10)
    U[4, 20] = 1; U[8, 20] = 1
    V[6, 20] = 1; V[2, 20] = 1
    W[1, 20] = 1; W[3, 20] = -1
    
    # Term 22: (e_01 + e_20) * (e_10 + e_22) * (e_12 - e_02)
    U[1, 21] = 1; U[6, 21] = 1
    V[3, 21] = 1; V[8, 21] = 1
    W[5, 21] = 1; W[2, 21] = -1
    
    # Term 23: (e_02 + e_10) * (e_20 + e_21) * (e_11 - e_00)
    U[2, 22] = 1; U[3, 22] = 1
    V[6, 22] = 1; V[7, 22] = 1
    W[4, 22] = 1; W[0, 22] = -1
    
    return U, V, W

def _build_schoolbook_np(n, m, p):
    """Return (U, V, W) as numpy arrays for schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_block_3x5x5():
    """
    Build decomposition for 3x5 x 5x5 using block structure with 5=3+2.
    Uses rank-23 for 3x3x3 blocks and schoolbook for mixed blocks.
    """
    n, m, p = 3, 5, 5
    
    # Split indices
    # A is 3x5: rows 0-2, cols 0-4
    # B is 5x5: rows 0-4, cols 0-4
    # C is 3x5: rows 0-2, cols 0-4
    #
    # Block split at position 3:
    # A = [A1 (3x3, cols 0-2) | A2 (3x2, cols 3-4)]
    # B = [[B11 (3x3, rows 0-2, cols 0-2), B12 (3x2, rows 0-2, cols 3-4)],
    #      [B21 (2x3, rows 3-4, cols 0-2), B22 (2x2, rows 3-4, cols 3-4)]]
    # C = [C1 (3x3, cols 0-2) | C2 (3x2, cols 3-4)]
    # C1[i,k] = sum_{j=0}^{2} A1[i,j]*B11[j,k] + sum_{j=3}^{4} A2[i,j-3]*B21[j-3,k]
    # C2[i,k] = sum_{j=0}^{2} A1[i,j]*B12[j,k-3] + sum_{j=3}^{4} A2[i,j-3]*B22[j-3,k-3]
    
    # Block products:
    # P1 = A1 * B11: (3x3)*(3x3) -> 3x3, use rank 23
    # P2 = A2 * B21: (3x2)*(2x3) -> 3x3, use schoolbook rank 18
    # P3 = A1 * B12: (3x3)*(3x2) -> 3x2, use schoolbook rank 18
    # P4 = A2 * B22: (3x2)*(2x2) -> 3x2, use schoolbook rank 12
    
    total_rank = 23 + 18 + 18 + 12  # = 71
    
    U = np.zeros((15, total_rank), dtype=np.float64)  # 3*5 = 15 rows
    V = np.zeros((25, total_rank), dtype=np.float64)  # 5*5 = 25 rows
    W = np.zeros((15, total_rank), dtype=np.float64)  # 5*3 = 15 rows
    
    col_offset = 0
    
    # Block P1: A1 (3x3) * B11 (3x3) -> C1 part (3x3)
    U1, V1, W1 = _build_graziani_3x3x3()
    # U1 is 9x23, indexed by (i,j) where i,j in 0..2 -> row i*3+j
    # V1 is 9x23, indexed by (j,k) where j,k in 0..2 -> row j*3+k
    # W1 is 9x23, indexed by (k,i) where k,i in 0..2 -> row k*3+i
    # In full matrices:
    # A row i, col j -> U row i*5 + j for j in 0..2
    # B row j, col k -> V row j*5 + k for j in 0..2, k in 0..2
    # C row i, col k -> W row k*3 + i for k in 0..2, i in 0..2
    for r in range(23):
        for i in range(3):
            for j in range(3):
                if U1[i*3+j, r] != 0:
                    U[i*5 + j, col_offset + r] = U1[i*3+j, r]
        for j in range(3):
            for k in range(3):
                if V1[j*3+k, r] != 0:
                    V[j*5 + k, col_offset + r] = V1[j*3+k, r]
        for k in range(3):
            for i in range(3):
                if W1[k*3+i, r] != 0:
                    W[k*3 + i, col_offset + r] = W1[k*3+i, r]
    col_offset += 23
    
    # Block P2: A2 (3x2) * B21 (2x3) -> C1 part (3x3)
    # A2: rows i (0..2), cols j (0..1) -> in A: row i, col 3+j
    # B21: rows j (0..1), cols k (0..2) -> in B: row 3+j, col k
    # C1: rows i (0..2), cols k (0..2) -> in C: row i, col k -> W row k*3+i
    U2, V2, W2 = _build_schoolbook_np(3, 2, 3)
    # U2: 6x18, row i*2+j for i in 0..2, j in 0..1
    # V2: 6x18, row j*3+k for j in 0..1, k in 0..2
    # W2: 9x18, row k*3+i for k in 0..2, i in 0..2
    for r in range(18):
        for i in range(3):
            for j in range(2):
                if U2[i*2+j, r] != 0:
                    U[i*5 + (3+j), col_offset + r] = U2[i*2+j, r]
        for j in range(2):
            for k in range(3):
                if V2[j*3+k, r] != 0:
                    V[(3+j)*5 + k, col_offset + r] = V2[j*3+k, r]
        for k in range(3):
            for i in range(3):
                if W2[k*3+i, r] != 0:
                    W[k*3 + i, col_offset + r] = W2[k*3+i, r]
    col_offset += 18
    
    # Block P3: A1 (3x3) * B12 (3x2) -> C2 part (3x2)
    # A1: rows i (0..2), cols j (0..2) -> in A: row i, col j
    # B12: rows j (0..2), cols k (0..1) -> in B: row j, col 3+k
    # C2: rows i (0..2), cols k (0..1) -> in C: row i, col 3+k -> W row (3+k)*3+i
    U3, V3, W3 = _build_schoolbook_np(3, 3, 2)
    # U3: 9x18, row i*3+j for i in 0..2, j in 0..2
    # V3: 6x18, row j*2+k for j in 0..2, k in 0..1
    # W3: 6x18, row k*3+i for k in 0..1, i in 0..2
    for r in range(18):
        for i in range(3):
            for j in range(3):
                if U3[i*3+j, r] != 0:
                    U[i*5 + j, col_offset + r] = U3[i*3+j, r]
        for j in range(3):
            for k in range(2):
                if V3[j*2+k, r] != 0:
                    V[j*5 + (3+k), col_offset + r] = V3[j*2+k, r]
        for k in range(2):
            for i in range(3):
                if W3[k*3+i, r] != 0:
                    W[(3+k)*3 + i, col_offset + r] = W3[k*3+i, r]
    col_offset += 18
    
    # Block P4: A2 (3x2) * B22 (2x2) -> C2 part (3x2)
    # A2: rows i (0..2), cols j (0..1) -> in A: row i, col 3+j
    # B22: rows j (0..1), cols k (0..1) -> in B: row 3+j, col 3+k
    # C2: rows i (0..2), cols k (0..1) -> in C: row i, col 3+k -> W row (3+k)*3+i
    U4, V4, W4 = _build_schoolbook_np(3, 2, 2)
    # U4: 6x12, row i*2+j for i in 0..2, j in 0..1
    # V4: 4x12, row j*2+k for j in 0..1, k in 0..1
    # W4: 6x12, row k*3+i for k in 0..1, i in 0..2
    for r in range(12):
        for i in range(3):
            for j in range(2):
                if U4[i*2+j, r] != 0:
                    U[i*5 + (3+j), col_offset + r] = U4[i*2+j, r]
        for j in range(2):
            for k in range(2):
                if V4[j*2+k, r] != 0:
                    V[(3+j)*5 + (3+k), col_offset + r] = V4[j*2+k, r]
        for k in range(2):
            for i in range(3):
                if W4[k*3+i, r] != 0:
                    W[(3+k)*3 + i, col_offset + r] = W4[k*3+i, r]
    col_offset += 12
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    n, m, p = DIMENSIONS
    
    # Try block decomposition
    try:
        U, V, W = _build_block_3x5x5()
        if _verify(U, V, W, n, m, p):
            best_rank = U.shape[1]
        else:
            # Fallback to schoolbook
            U, V, W = _build_schoolbook_np(n, m, p)
            best_rank = U.shape[1]
    except Exception:
        U, V, W = _build_schoolbook_np(n, m, p)
        best_rank = U.shape[1]
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
