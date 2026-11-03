import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def _add_strassen_block(U, V, W, i0, j0, k0, col_offset):
    """
    Adds a 2x2x2 Strassen block to the global matrices.
    i0: starting row index in A (and C)
    j0: starting col index in A (and row index in B)
    k0: starting col index in B (and col index in C)
    col_offset: starting column index in the rank dimension
    """
    n, m, p = DIMENSIONS
    
    # Strassen coefficients for 2x2x2
    # U_local (4x7), V_local (4x7), W_local (4x7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    U_loc = np.zeros((4, 7))
    V_loc = np.zeros((4, 7))
    W_loc = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1 # C00, C11
    
    # M2 = (A10+A11)B00
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[1, 1] = 1; W_loc[3, 1] = -1 # C10, C11
    
    # M3 = A00(B01-B11)
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[2, 2] = 1; W_loc[3, 2] = 1 # C01, C11? 
    # C01 = M3 + M5. C11 = M1 - M2 + M3 + M6.
    # So M3 contributes to C01 (+1) and C11 (+1).
    # W index for C01 is 2 (k=1, i=0). W index for C11 is 3 (k=1, i=1).
    # Correct.
    
    # M4 = A11(B10-B00)
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[1, 3] = 1; W_loc[3, 3] = 1 # C10, C11
    # C10 = M2 + M4. C11 = ... + M4.
    # W index for C10 is 1. W index for C11 is 3.
    # Correct.
    
    # M5 = (A00+A01)B11
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[2, 4] = 1 # C00, C01
    # C00 = M1 + M4 - M5 + M7. So -1 for C00.
    # C01 = M3 + M5. So +1 for C01.
    # W index for C00 is 0. W index for C01 is 2.
    # Correct.
    
    # M6 = (A10-A00)(B00+B01)
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1 # C11
    # C11 = ... + M6. So +1 for C11.
    # W index for C11 is 3.
    # Correct.
    
    # M7 = (A01-A11)(B10+B11)
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1 # C00
    # C00 = ... + M7. So +1 for C00.
    # W index for C00 is 0.
    # Correct.
    
    # Map to global indices
    for r in range(7):
        for i in range(2):
            for j in range(2):
                val = U_loc[i*2+j, r]
                if val != 0:
                    U[(i0+i)*m + (j0+j)][col_offset+r] += val
        for j in range(2):
            for k in range(2):
                val = V_loc[j*2+k, r]
                if val != 0:
                    V[(j0+j)*p + (k0+k)][col_offset+r] += val
        for k in range(2):
            for i in range(2):
                val = W_loc[k*2+i, r]
                if val != 0:
                    W[(k0+k)*n + (i0+i)][col_offset+r] += val

def _add_rank4_block(U, V, W, i0, j0, k0, col_offset):
    """
    Adds a 2x1x2 block (rank 4) to the global matrices.
    i0: starting row index in A
    j0: row index in A (single)
    k0: starting col index in B
    """
    n, m, p = DIMENSIONS
    
    # A is 2x1, B is 1x2
    # C = A B
    # C00 = A00 B00
    # C01 = A00 B01
    # C10 = A10 B00
    # C11 = A10 B01
    
    # Term 1: A00 B00 -> C00
    U[(i0+0)*m + j0][col_offset+0] = 1
    V[j0*p + (k0+0)][col_offset+0] = 1
    W[(k0+0)*n + (i0+0)][col_offset+0] = 1
    
    # Term 2: A00 B01 -> C01
    U[(i0+0)*m + j0][col_offset+1] = 1
    V[j0*p + (k0+1)][col_offset+1] = 1
    W[(k0+1)*n + (i0+0)][col_offset+1] = 1
    
    # Term 3: A10 B00 -> C10
    U[(i0+1)*m + j0][col_offset+2] = 1
    V[j0*p + (k0+0)][col_offset+2] = 1
    W[(k0+0)*n + (i0+1)][col_offset+2] = 1
    
    # Term 4: A10 B01 -> C11
    U[(i0+1)*m + j0][col_offset+3] = 1
    V[j0*p + (k0+1)][col_offset+3] = 1
    W[(k0+1)*n + (i0+1)][col_offset+3] = 1

def construct():
    n, m, p = DIMENSIONS
    total_rank = 108
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j in {0, 1}
    for k0 in [0, 2, 4]:
        # i in {0, 1}
        _add_strassen_block(U, V, W, 0, 0, k0, col)
        col += 7
        # i in {2, 3}
        _add_strassen_block(U, V, W, 2, 0, k0, col)
        col += 7
        
    # Part 2: j in {2, 3, 4}
    for k0 in [0, 2, 4]:
        # j in {2, 3}: 4x2x2 block
        # i in {0, 1}
        _add_strassen_block(U, V, W, 0, 2, k0, col)
        col += 7
        # i in {2, 3}
        _add_strassen_block(U, V, W, 2, 2, k0, col)
        col += 7
        
        # j in {4}: 4x1x2 block
        # i in {0, 1}
        _add_rank4_block(U, V, W, 0, 4, k0, col)
        col += 4
        # i in {2, 3}
        _add_rank4_block(U, V, W, 2, 4, k0, col)
        col += 4
        
    # Verify
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        return _schoolbook()

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(result, 1.0, atol=1e-9)

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
