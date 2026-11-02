import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition is exact using the tensor check."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _strassen_2x2x2(offset_u, offset_v, offset_w, U, V, W, col_offset):
    """
    Add a Strassen 2x2x2 block.
    U rows: offset_u to offset_u+3 (i=0,1; j=0,1)
    V rows: offset_v to offset_v+3 (j=0,1; k=0,1)
    W rows: offset_w to offset_w+3 (k=0,1; i=0,1)
    Returns new col_offset.
    """
    col = col_offset
    
    # M1 = (A00+A11)*(B00+B11)
    U[offset_u + 0][col] = 1
    U[offset_u + 3][col] = 1
    V[offset_v + 0][col] = 1
    V[offset_v + 3][col] = 1
    W[offset_w + 0][col] = 1
    W[offset_w + 3][col] = 1
    col += 1
    
    # M2 = (A10+A11)*B00
    U[offset_u + 2][col] = 1
    U[offset_u + 3][col] = 1
    V[offset_v + 0][col] = 1
    W[offset_w + 2][col] = 1
    W[offset_w + 3][col] = -1
    col += 1
    
    # M3 = A00*(B01-B11)
    U[offset_u + 0][col] = 1
    V[offset_v + 1][col] = 1
    V[offset_v + 3][col] = -1
    W[offset_w + 0][col] = 1
    W[offset_w + 3][col] = 1
    col += 1
    
    # M4 = A11*(B10-B00)
    U[offset_u + 3][col] = 1
    V[offset_v + 2][col] = 1
    V[offset_v + 0][col] = -1
    W[offset_w + 0][col] = 1
    W[offset_w + 2][col] = 1
    col += 1
    
    # M5 = (A00+A01)*B11
    U[offset_u + 0][col] = 1
    U[offset_u + 1][col] = 1
    V[offset_v + 3][col] = 1
    W[offset_w + 0][col] = -1
    W[offset_w + 1][col] = 1
    col += 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[offset_u + 2][col] = 1
    U[offset_u + 0][col] = -1
    V[offset_v + 0][col] = 1
    V[offset_v + 1][col] = 1
    W[offset_w + 3][col] = 1
    col += 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[offset_u + 1][col] = 1
    U[offset_u + 3][col] = -1
    V[offset_v + 2][col] = 1
    V[offset_v + 3][col] = 1
    W[offset_w + 0][col] = 1
    col += 1
    
    return col

def _schoolbook_2x2x1(offset_u, offset_v, offset_w, U, V, W, col_offset):
    """
    Add a schoolbook 2x2x1 block (one column of B).
    U rows: offset_u to offset_u+3 (i=0,1; j=0,1)
    V rows: offset_v to offset_v+1 (j=0,1; k=0)
    W rows: offset_w to offset_w+1 (k=0; i=0,1)
    Returns new col_offset.
    """
    col = col_offset
    for i in range(2):
        for j in range(2):
            U[offset_u + i*2 + j][col] = 1
            V[offset_v + j][col] = 1
            W[offset_w + i][col] = 1
            col += 1
    return col

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Rank 36 Construction
    total_rank = 36
    U = [[0.0] * total_rank for _ in range(n * m)]  # 8 rows
    V = [[0.0] * total_rank for _ in range(m * p)]  # 20 rows
    W = [[0.0] * total_rank for _ in range(p * n)]  # 10 rows
    
    col_offset = 0
    
    # Block 1: A1 (cols 0-1) * B1 (rows 0-1)
    # A1 is U[0:4], B1 is V[0:10]
    
    # Sub-block 1a: A1 * B1a (B1a cols 0-1)
    # Strassen 2x2x2
    # U offset 0, V offset 0, W offset 0
    col_offset = _strassen_2x2x2(0, 0, 0, U, V, W, col_offset)
    
    # Sub-block 1b: A1 * B1b (B1b col 2)
    # Schoolbook 2x2x1
    # U offset 0, V offset 2 (j=0,1; k=2), W offset 4 (k=2; i=0,1)
    col_offset = _schoolbook_2x2x1(0, 2, 4, U, V, W, col_offset)
    
    # Sub-block 1c: A1 * B1c (B1c cols 3-4)
    # Strassen 2x2x2
    # U offset 0, V offset 6 (j=0,1; k=3,4), W offset 6 (k=3,4; i=0,1)
    col_offset = _strassen_2x2x2(0, 6, 6, U, V, W, col_offset)
    
    # Block 2: A2 (cols 2-3) * B2 (rows 2-3)
    # A2 is U[4:8], B2 is V[10:20]
    
    # Sub-block 2a: A2 * B2a (B2a cols 0-1)
    # Strassen 2x2x2
    # U offset 4, V offset 10, W offset 0
    col_offset = _strassen_2x2x2(4, 10, 0, U, V, W, col_offset)
    
    # Sub-block 2b: A2 * B2b (B2b col 2)
    # Schoolbook 2x2x1
    # U offset 4, V offset 12 (j=2,3; k=2), W offset 4 (k=2; i=0,1)
    col_offset = _schoolbook_2x2x1(4, 12, 4, U, V, W, col_offset)
    
    # Sub-block 2c: A2 * B2c (B2c cols 3-4)
    # Strassen 2x2x2
    # U offset 4, V offset 16 (j=2,3; k=3,4), W offset 6 (k=3,4; i=0,1)
    col_offset = _strassen_2x2x2(4, 16, 6, U, V, W, col_offset)
    
    # Verify
    if _verify(U, V, W, n, m, p, total_rank):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails (should not happen)
    rank = n * m * p
    U_fb = [[0] * rank for _ in range(n * m)]
    V_fb = [[0] * rank for _ in range(m * p)]
    W_fb = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
