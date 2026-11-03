"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
_start_time = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def _time_left():
    return TIME_LIMIT - (time.time() - _start_time) - SAFETY_MARGIN

def _verify(U, V, W):
    """Verify that U, V, W form an exact tensor decomposition of matrix multiplication.
    Returns True if valid, False otherwise.
    """
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _schoolbook():
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

def _strassen_2x2():
    """Return Strassen decomposition for 2x2 * 2x2 -> 2x2.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products for 2x2 matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e)
    # M5 = (a+b)h, M6 = (c-d)(f+h), M7 = (b-c)(e+g)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # Index mapping:
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=4, (0,1)=5, (1,0)=6, (1,1)=7
    # C indices: (0,0)=8, (0,1)=9, (1,0)=10, (1,1)=11
    # But in our convention:
    # U row index = i*m + j = i*2 + j
    # V row index = j*p + k = j*2 + k  
    # W row index = k*n + i = k*2 + i
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a+d)(e+h): A[0,0], A[1,1], B[0,0], B[1,1]
    # C[0,0] += M1, C[1,1] += M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # W[k*2+i]: C[0,0] -> k=0,i=0 -> idx 0; C[1,1] -> k=1,i=1 -> idx 3
    
    # M2 = (c+d)e: A[1,0], A[1,1], B[0,0]
    # C[1,0] += M2, C[1,1] -= M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1  # C[1,0] -> k=0,i=1 -> idx 2; C[1,1] -> k=1,i=1 -> idx 3
    
    # M3 = a(f-h): A[0,0], B[0,1], B[1,1]
    # C[0,1] += M3, C[1,1] += M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1  # C[0,1] -> k=1,i=0 -> idx 1; C[1,1] -> k=1,i=1 -> idx 3
    
    # M4 = d(g-e): A[1,1], B[1,0], B[0,0]
    # C[0,0] += M4, C[1,0] += M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1  # C[0,0] -> k=0,i=0 -> idx 0; C[1,0] -> k=0,i=1 -> idx 2
    
    # M5 = (a+b)h: A[0,0], A[0,1], B[1,1]
    # C[0,0] -= M5, C[0,1] += M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1  # C[0,0] -> idx 0; C[0,1] -> idx 1
    
    # M6 = (c-d)(f+h): A[1,0], A[1,1], B[0,1], B[1,1]
    # C[1,1] += M6
    U[2][5] = 1; U[3][5] = -1
    V[1][5] = 1; V[3][5] = 1
    W[3][5] = 1  # C[1,1] -> idx 3
    
    # M7 = (b-c)(e+g): A[0,1], A[1,0], B[0,0], B[1,0]
    # C[0,0] += M7
    U[1][6] = 1; U[2][6] = -1
    V[0][6] = 1; V[2][6] = 1
    W[0][6] = 1  # C[0,0] -> idx 0
    
    return {"U": U, "V": V, "W": W}

def _build_3x3x3_strassen():
    """Build a decomposition for 3x3 * 3x3 using Strassen on 2x2 blocks.
    Split 3x3 into overlapping 2x2 blocks.
    Use Strassen for top-left 2x2, then handle the remaining parts.
    """
    n, m, p = 3, 3, 3
    # We'll use a structured approach:
    # Decompose 3x3 multiplication using the fact that we can embed 2x2 Strassen
    # and handle the extra row/column separately.
    
    # Strategy: 
    # (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # For i,k in {0,1}: use Strassen on the 2x2 submatrices
    # For i=2 or k=2: handle separately
    
    # Actually, let's use a different approach: use the known rank-23 decomposition
    # or try to construct one using Strassen embedding.
    
    # For now, let's try a simpler block decomposition:
    # Split into 2x2 + 1x1 blocks
    # Top-left 2x2: use Strassen (rank 7)
    # Top-right 2x1: schoolbook (rank 2)
    # Bottom-left 1x2: schoolbook (rank 2)
    # Bottom-right 1x1: schoolbook (rank 1)
    # Total: 7 + 2 + 2 + 1 = 12? But there are cross terms...
    
    # Actually, this doesn't work directly because the blocks share indices.
    
    # Let me try a different approach: use the fact that the 3x3 multiplication tensor
    # can be decomposed using 23 terms (known result). I'll try to construct it.
    
    # For now, return None to indicate we need a different approach.
    return None

def _build_strassen_embedding_3x3x8():
    """Build a decomposition for 3x3 * 3x8 using Strassen embedding.
    
    Approach: Split the 3x3 matrix A and 3x8 matrix B into 2x2 blocks where possible.
    Use Strassen's algorithm for the 2x2 * 2x2 parts, and handle the rest with schoolbook.
    
    A (3x3):
    [a00 a01 a02]
    [a10 a11 a12]
    [a20 a21 a22]
    
    B (3x8):
    [b00 b01 b02 b03 b04 b05 b06 b07]
    [b10 b11 b12 b13 b14 b15 b16 b17]
    [b20 b21 b22 b23 b24 b25 b26 b27]
    
    C (3x8):
    [c00 c01 ... c07]
    [c10 c11 ... c17]
    [c20 c21 ... c27]
    
    We can split B into B1 (3x2), B2 (3x2), B3 (3x2), B4 (3x2) for columns 0-1, 2-3, 4-5, 6-7.
    For each 3x2 block, we can further decompose.
    
    Actually, let's try a simpler approach: use Strassen on the top-left 2x2 of A 
    and corresponding 2x8 of B, then handle the remaining parts.
    
    For the top-left 2x2 of A times top 2x8 of B:
    This gives us 2x8 output. We can split the 2x8 into four 2x2 blocks.
    Each 2x2 * 2x2 can be done with Strassen (rank 7).
    So 4 * 7 = 28 for this part.
    
    Then handle:
    - Row 2 of A times all of B: 1x3 * 3x8 = 1x8, schoolbook rank 24
    - All of A times row 2 of B: but this is already handled above for columns 0-7
    
    Wait, I'm overcomplicating this. Let me use a cleaner block decomposition.
    
    Split A into:
    A11 (2x2), A12 (2x1), A21 (1x2), A22 (1x1)
    
    Split B into:
    B1 (2x8), B2 (1x8)
    
    Then:
    C11 = A11 * B1_top (2x2 * 2x8) -> 2x8
    C12 = A12 * B2 (2x1 * 1x8) -> 2x8  
    C21 = A21 * B1_bottom (1x2 * 2x8) -> 1x8
    C22 = A22 * B2_bottom (1x1 * 1x8) -> 1x8
    
    For C11: A11 (2x2) * B1_top (2x8). Split B1_top into four 2x2 blocks.
    Each 2x2 * 2x2 uses Strassen (rank 7). Total: 4 * 7 = 28.
    
    For C12: A12 (2x1) * B2 (1x8). This is outer product-like.
    (A12 * B2)_{i,k} = A12_{i,0} * B2_{0,k}
    This is rank 16 in tensor terms (2 * 1 * 8 = 16).
    
    For C21: A21 (1x2) * B1_bottom (2x8).
    (A21 * B1_bottom)_{0,k} = sum_j A21_{0,j} * B1_bottom_{j,k}
    This is rank 16 (1 * 2 * 8 = 16).
    
    For C22: A22 (1x1) * B2_bottom (1x8).
    This is rank 8 (1 * 1 * 8 = 8).
    
    Total: 28 + 16 + 16 + 8 = 68.
    
    Let me construct this decomposition.
    """
    n, m, p = 3, 3, 8
    total_rank = 68
    
    U = [[0] * total_rank for _ in range(n * m)]  # 9 x 68
    V = [[0] * total_rank for _ in range(m * p)]  # 24 x 68
    W = [[0] * total_rank for _ in range(p * n)]  # 24 x 68
    
    col = 0
    
    # C11: A11 (rows 0-1, cols 0-1) * B1_top (rows 0-1, cols 0-7)
    # Split into 4 2x2 blocks: cols (0,1), (2,3), (4,5), (6,7)
    for block in range(4):
        k_start = block * 2
        k_end = k_start + 2
        # For each 2x2 * 2x2, use Strassen
        strassen = _strassen_2x2()
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col + c] += strassen["U"][i * 2 + j][c]
            for j in range(2):
                for k in range(2):
                    V[j * p + (k_start + k)][col + c] += strassen["V"][j * 2 + k][c]
            for k in range(2):
                for i in range(2):
                    W[(k_start + k) * n + i][col + c] += strassen["W"][k * 2 + i][c]
        col += 7
    
    # C12: A12 (rows 0-1, col 2) * B2 (row 2, cols 0-7)
    # (A12 * B2)_{i,k} = A12_{i,0} * B2_{0,k}
    # U index: i*m + 2 = i*3 + 2
    # V index: 2*p + k = 16 + k
    # W index: k*n + i = k*3 + i
    for i in range(2):
        for k in range(8):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # C21: A21 (row 2, cols 0-1) * B1_bottom (rows 0-1, cols 0-7)
    # (A21 * B1_bottom)_{0,k} = sum_{j=0}^{1} A21_{0,j} * B1_bottom_{j,k}
    for j in range(2):
        for k in range(8):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
    
    # C22: A22 (row 2, col 2) * B2_bottom (row 2, cols 0-7)
    # (A22 * B2_bottom)_{0,k} = A22_{0,0} * B2_bottom_{0,k}
    for k in range(8):
        U[2 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 2][col] = 1
        col += 1
    
    result = {"U": U, "V": V, "W": W}
    
    if _verify(U, V, W):
        return result
    else:
        return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_result = _schoolbook()
    
    best = schoolbook_result
    best_rank = n * m * p
    
    if _time_left() < 5:
        return best
    
    # Try the Strassen embedding approach
    strassen_result = _build_strassen_embedding_3x3x8()
    if strassen_result is not None:
        rank = len(strassen_result["U"][0])
        if rank < best_rank:
            best = strassen_result
            best_rank = rank
    
    return best

# EVOLVE-BLOCK-END
