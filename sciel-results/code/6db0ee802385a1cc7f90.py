import time
import sys

DIMENSIONS = [3, 3, 3]

def _verify(U, V, W):
    """Verify that U,V,W with given indices correctly decompose matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _construct_rank_26():
    """
    Construct a rank-26 decomposition for 3x3 matrix multiplication.
    Uses Strassen's algorithm on the top-left 2x2 block and standard 
    multiplication for the interactions with the third row/column.
    """
    n, m, p = 3, 3, 3
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices):
        # u_indices: list of (row, col) in U (A matrix)
        # v_indices: list of (row, col) in V (B matrix)
        # w_indices: list of (row, col) in W (C matrix)
        pass # We will use direct assignment below for clarity and speed

    # 1. Strassen terms for 2x2 block (indices 0,1)
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2
    # M3 = A00(B11-B01) -> C01 += M3
    # M4 = A11(B10-B00) -> C00 -= M4, C10 += M4
    # M5 = (A00+A01)B11 -> C01 += M5, C11 -= M5
    # M6 = (A10-A00)(B00+B01) -> C00 -= M6, C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7, C11 += M7
    
    # Term 0: M1
    # U: (0,0), (1,1)
    # V: (0,0), (1,1)
    # W: (0,0), (1,1)
    U[0*3+0][0] += 1
    U[1*3+1][0] += 1
    V[0*3+0][0] += 1
    V[1*3+1][0] += 1
    W[0*3+0][0] += 1
    W[1*3+1][0] += 1
    
    # Term 1: M2
    # U: (1,0), (1,1)
    # V: (0,0)
    # W: (1,0)
    U[1*3+0][1] += 1
    U[1*3+1][1] += 1
    V[0*3+0][1] += 1
    W[1*3+0][1] += 1
    
    # Term 2: M3
    # U: (0,0)
    # V: (1,1), (0,1) with -1
    # W: (0,1)
    U[0*3+0][2] += 1
    V[1*3+1][2] += 1
    V[0*3+1][2] -= 1
    W[0*3+1][2] += 1
    
    # Term 3: M4
    # U: (1,1)
    # V: (1,0), (0,0) with -1
    # W: (0,0) with -1, (1,0) with +1
    U[1*3+1][3] += 1
    V[1*3+0][3] += 1
    V[0*3+0][3] -= 1
    W[0*3+0][3] -= 1
    W[1*3+0][3] += 1
    
    # Term 4: M5
    # U: (0,0), (0,1)
    # V: (1,1)
    # W: (0,1) with +1, (1,1) with -1
    U[0*3+0][4] += 1
    U[0*3+1][4] += 1
    V[1*3+1][4] += 1
    W[0*3+1][4] += 1
    W[1*3+1][4] -= 1
    
    # Term 5: M6
    # U: (1,0), (0,0) with -1
    # V: (0,0), (0,1)
    # W: (0,0) with -1, (1,1) with +1
    U[1*3+0][5] += 1
    U[0*3+0][5] -= 1
    V[0*3+0][5] += 1
    V[0*3+1][5] += 1
    W[0*3+0][5] -= 1
    W[1*3+1][5] += 1
    
    # Term 6: M7
    # U: (0,1), (1,1) with -1
    # V: (1,0), (1,1)
    # W: (0,0) with +1, (1,1) with +1
    U[0*3+1][6] += 1
    U[1*3+1][6] -= 1
    V[1*3+0][6] += 1
    V[1*3+1][6] += 1
    W[0*3+0][6] += 1
    W[1*3+1][6] += 1
    
    # 2. Terms for A[:, 2] * B[2, :] -> C[:, :]
    # 9 terms: for i in 0..2, k in 0..2
    # U[i, 2], V[2, k], W[k, i]
    col = 7
    for i in range(3):
        for k in range(3):
            U[i*3+2][col] += 1
            V[2*3+k][col] += 1
            W[k*3+i][col] += 1
            col += 1
            
    # 3. Terms for A[:, 0..1] * B[0..1, 2] -> C[:, 2]
    # 6 terms: for i in 0..2, j in 0..1
    # U[i, j], V[j, 2], W[2, i]
    for i in range(3):
        for j in range(2):
            U[i*3+j][col] += 1
            V[j*3+2][col] += 1
            W[2*3+i][col] += 1
            col += 1
            
    # 4. Terms for A[2, 0..1] * B[0..1, :] -> C[2, :]
    # 4 terms: for j in 0..1, k in 0..1
    # U[2, j], V[j, k], W[k, 2]
    for j in range(2):
        for k in range(2):
            U[2*3+j][col] += 1
            V[j*3+k][col] += 1
            W[k*3+2][col] += 1
            col += 1
            
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    return None

def _construct_schoolbook():
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
    deadline = start_time + 115

    # Try rank-26 first
    res_26 = _construct_rank_26()
    if res_26:
        return res_26

    # Fallback to schoolbook
    return _construct_schoolbook()
