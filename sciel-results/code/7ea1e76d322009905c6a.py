import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def _add_strassen_block(U, V, W, col, i_off, j_off, k_off):
    """Add Strassen 2x2x2 block at given offsets. Returns new col."""
    n, m, p = 2, 3, 5
    # Local indices: i_l in {0,1}, j_l in {0,1}, k_l in {0,1}
    # Global: i = i_off+i_l, j = j_off+j_l, k = k_off+k_l
    
    # M1: (A00+A11)(B00+B11) -> C00 += , C11 +=
    U[i_off * m + j_off][col] = 1
    U[(i_off+1) * m + (j_off+1)][col] = 1
    V[j_off * p + k_off][col] = 1
    V[(j_off+1) * p + (k_off+1)][col] = 1
    W[k_off * n + i_off][col] = 1
    W[(k_off+1) * n + (i_off+1)][col] = 1
    col += 1
    
    # M2: (A10+A11)B00 -> C10 += , C11 -=
    U[(i_off+1) * m + j_off][col] = 1
    U[(i_off+1) * m + (j_off+1)][col] = 1
    V[j_off * p + k_off][col] = 1
    W[k_off * n + (i_off+1)][col] = 1
    W[(k_off+1) * n + (i_off+1)][col] = -1
    col += 1
    
    # M3: A00(B01-B11) -> C01 += , C11 +=
    U[i_off * m + j_off][col] = 1
    V[j_off * p + (k_off+1)][col] = 1
    V[(j_off+1) * p + (k_off+1)][col] = -1
    W[(k_off+1) * n + i_off][col] = 1
    W[(k_off+1) * n + (i_off+1)][col] = 1
    col += 1
    
    # M4: A11(B10-B00) -> C00 += , C10 +=
    U[(i_off+1) * m + (j_off+1)][col] = 1
    V[(j_off+1) * p + k_off][col] = 1
    V[j_off * p + k_off][col] = -1
    W[k_off * n + i_off][col] = 1
    W[k_off * n + (i_off+1)][col] = 1
    col += 1
    
    # M5: (A00+A01)B11 -> C00 -= , C01 +=
    U[i_off * m + j_off][col] = 1
    U[i_off * m + (j_off+1)][col] = 1
    V[(j_off+1) * p + (k_off+1)][col] = 1
    W[k_off * n + i_off][col] = -1
    W[(k_off+1) * n + i_off][col] = 1
    col += 1
    
    # M6: (A10-A00)(B00+B01) -> C11 +=
    U[(i_off+1) * m + j_off][col] = 1
    U[i_off * m + j_off][col] = -1
    V[j_off * p + k_off][col] = 1
    V[j_off * p + (k_off+1)][col] = 1
    W[(k_off+1) * n + (i_off+1)][col] = 1
    col += 1
    
    # M7: (A01-A11)(B10+B11) -> C00 +=
    U[i_off * m + (j_off+1)][col] = 1
    U[(i_off+1) * m + (j_off+1)][col] = -1
    V[(j_off+1) * p + k_off][col] = 1
    V[(j_off+1) * p + (k_off+1)][col] = 1
    W[k_off * n + i_off][col] = 1
    col += 1
    
    return col

def construct():
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 6 x 28
    V = [[0] * rank for _ in range(m * p)]  # 15 x 28
    W = [[0] * rank for _ in range(p * n)]  # 10 x 28
    
    col = 0
    
    # Block 1: j=0, all k (schoolbook 2x1x5) -> 10 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Block 2a: j={1,2}, k={0,1} (Strassen) -> 7 terms
    col = _add_strassen_block(U, V, W, col, 0, 1, 0)
    
    # Block 2b: j={1,2}, k=2 (schoolbook 2x2x1) -> 4 terms
    for i in range(n):
        for j in range(1, 3):
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
            col += 1
    
    # Block 2c: j={1,2}, k={3,4} (Strassen) -> 7 terms
    col = _add_strassen_block(U, V, W, col, 0, 1, 3)
    
    assert col == rank, f"Column mismatch: {col} != {rank}"
    return {"U": U, "V": V, "W": W}
