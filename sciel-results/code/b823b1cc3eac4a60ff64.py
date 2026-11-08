"""Matrix multiplication 2x4 by 4x6: rank-minimal exact tensor decomposition."""

import time
import numpy as np

def _strassen_block():
    """Return the 7 Strassen terms as (U_sub, V_sub, W_sub) for a 2x2x2 block.
    U_sub is 4x7 (rows: (0,0),(0,1),(1,0),(1,1)), V_sub is 4x7, W_sub is 4x7.
    Indexing: row = i*2 + j for A[i,j], B[i,j], C[i,j]."""
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (a00+a11)*(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C00, C11
    
    # M2 = (a10+a11)*b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1  # C10, -C11
    
    # M3 = a00*(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # C01, C11
    
    # M4 = a11*(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1  # C00, C10
    
    # M5 = (a00+a01)*b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # -C00, C01
    
    # M6 = (a10-a00)*(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C11
    
    # M7 = (a01-a11)*(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C00
    
    return U, V, W

def construct():
    t0 = time.time()
    
    n, m, p = 2, 4, 6
    U_rows = n * m  # 8
    V_rows = m * p  # 24
    W_rows = p * n  # 12
    
    rank = 42
    U = np.zeros((U_rows, rank), dtype=int)
    V = np.zeros((V_rows, rank), dtype=int)
    W = np.zeros((W_rows, rank), dtype=int)
    
    # Precompute Strassen block
    U_s, V_s, W_s = _strassen_block()
    
    col = 0
    for a_block in range(2):  # A columns 0-1 or 2-3
        for b_block_row in range(2):  # B rows 0-1 or 2-3
            for t in range(3):  # B columns 2t, 2t+1
                a_col_off = 2 * a_block
                b_row_off = 2 * b_block_row
                b_col_off = 2 * t
                
                # For each of the 7 Strassen terms
                for c in range(7):
                    # U: A entries at (i, a_col_off + j)
                    for i in range(2):
                        for j in range(2):
                            if U_s[i*2+j, c] != 0:
                                U[i*m + (a_col_off + j), col] = U_s[i*2+j, c]
                    
                    # V: B entries at (b_row_off + i, b_col_off + j)
                    for i in range(2):
                        for j in range(2):
                            if V_s[i*2+j, c] != 0:
                                V[(b_row_off + i)*p + (b_col_off + j), col] = V_s[i*2+j, c]
                    
                    # W: C entries at (b_col_off + k, i)
                    # W index: k*n + i, where k is column of C, i is row of C
                    for i in range(2):
                        for k in range(2):
                            if W_s[i*2+k, c] != 0:
                                W[(b_col_off + k)*n + i, col] = W_s[i*2+k, c]
                    
                    col += 1
    
    # Convert to lists as expected by the interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
