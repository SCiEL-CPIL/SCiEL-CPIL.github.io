import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def _build_rank44():
    """
    Build rank-44 decomposition for 3x4 * 4x4 matrix multiplication.
    Split A into A_top (2x4) and A_bot (1x4).
    Use Strassen-based rank-28 for 2x4x4 and rank-16 schoolbook for 1x4x4.
    """
    n, m, p = 3, 4, 4
    rank = 44
    
    U = np.zeros((n * m, rank))  # (12, 44)
    V = np.zeros((m * p, rank))  # (16, 44)
    W = np.zeros((p * n, rank))  # (12, 44)
    
    # Part 1: 1x4x4 schoolbook (rank 16) for A_bot (row 2 of A)
    col_offset = 0
    for j in range(m):  # 4 columns of A_bot
        for k in range(p):  # 4 columns of B
            U[2 * m + j, col_offset] = 1  # A[2, j]
            V[j * p + k, col_offset] = 1  # B[j, k]
            W[k * n + 2, col_offset] = 1  # C[k, 2]
            col_offset += 1
    # col_offset = 16
    
    # Part 2: 2x4x4 using Strassen (rank 28) for A_top (rows 0,1 of A)
    # Split 2x4 * 4x4 into four 2x2 * 2x2 block multiplications
    # A_top = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    # C_top = A_top * B = [[C11, C12], [C21, C22]]
    
    # Strassen for 2x2 * 2x2 (rank 7):
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each 2x2*2x2 block multiplication, we need 7 rank-1 terms.
    # We have 4 block multiplications, so 4*7 = 28 terms.
    
    def add_strassen(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        """Add Strassen rank-7 decomposition for one 2x2*2x2 block."""
        nonlocal col_offset
        # M1 = (A11 + A22) * (B11 + B22)
        U[a00, col_offset] = 1
        U[a11, col_offset] = 1
        V[b00, col_offset] = 1
        V[b11, col_offset] = 1
        W[c00, col_offset] = 1
        W[c11, col_offset] = 1
        col_offset += 1
        
        # M2 = (A21 + A22) * B11
        U[a10, col_offset] = 1
        U[a11, col_offset] = 1
        V[b00, col_offset] = 1
        W[c10, col_offset] = 1
        W[c11, col_offset] = -1
        col_offset += 1
        
        # M3 = A11 * (B12 - B22)
        U[a00, col_offset] = 1
        V[b01, col_offset] = 1
        V[b11, col_offset] = -1
        W[c01, col_offset] = 1
        W[c11, col_offset] = 1
        col_offset += 1
        
        # M4 = A22 * (B21 - B11)
        U[a11, col_offset] = 1
        V[b10, col_offset] = 1
        V[b00, col_offset] = -1
        W[c00, col_offset] = 1
        W[c10, col_offset] = 1
        col_offset += 1
        
        # M5 = (A11 + A12) * B22
        U[a00, col_offset] = 1
        U[a01, col_offset] = 1
        V[b11, col_offset] = 1
        W[c00, col_offset] = -1
        W[c01, col_offset] = 1
        col_offset += 1
        
        # M6 = (A21 - A11) * (B11 + B12)
        U[a10, col_offset] = 1
        U[a00, col_offset] = -1
        V[b00, col_offset] = 1
        V[b01, col_offset] = 1
        W[c11, col_offset] = 1
        col_offset += 1
        
        # M7 = (A12 - A22) * (B21 + B22)
        U[a01, col_offset] = 1
        U[a11, col_offset] = -1
        V[b10, col_offset] = 1
        V[b11, col_offset] = 1
        W[c00, col_offset] = 1
        col_offset += 1
    
    # Block 1: A11 * B11 -> C[0:2, 0:2]
    # A11: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    # B11: B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    # C: C[0,0]=0, C[0,1]=1, C[1,0]=4, C[1,1]=5
    add_strassen(0, 1, 4, 5, 0, 1, 4, 5, 0, 1, 4, 5)
    
    # Block 2: A12 * B21 -> C[0:2, 0:2]
    # A12: A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
    # B21: B[2,0]=8, B[2,1]=9, B[3,0]=12, B[3,1]=13
    # C: same as block 1
    add_strassen(2, 3, 6, 7, 8, 9, 12, 13, 0, 1, 4, 5)
    
    # Block 3: A11 * B12 -> C[0:2, 2:4]
    # A11: same as block 1
    # B12: B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7
    # C: C[0,2]=2, C[0,3]=3, C[1,2]=6, C[1,3]=7
    add_strassen(0, 1, 4, 5, 2, 3, 6, 7, 2, 3, 6, 7)
    
    # Block 4: A12 * B22 -> C[0:2, 2:4]
    # A12: same as block 2
    # B22: B[2,2]=10, B[2,3]=11, B[3,2]=14, B[3,3]=15
    # C: same as block 3
    add_strassen(2, 3, 6, 7, 10, 11, 14, 15, 2, 3, 6, 7)
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    
    # Try rank-44 decomposition
    try:
        U, V, W = _build_rank44()
        return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook rank 48
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
