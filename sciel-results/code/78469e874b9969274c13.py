"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time

def _add_strassen_block(U, V, W, col_offset, a_col_offset, b_col_offset, w_row_offset):
    """
    Adds a Strassen 2x2x2 block to the decomposition.
    U, V, W are lists of lists (or arrays) to be modified in place.
    col_offset: starting column index in U,V,W.
    a_col_offset: starting column index in A (0 or 1, but here always 0 for the 2x2 part).
    b_col_offset: starting column index in B (0 or 2).
    w_row_offset: starting row index in W (0 or 4).
    """
    # Strassen multipliers M1..M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Indices in local 2x2x2 space:
    # A[i,j], B[j,k], C[i,k]
    # Global U index: i*m + (a_col_offset + j)
    # Global V index: (b_row_offset + j)*p + (b_col_offset + k)  -- Wait, V is (m,p). Index is j*p + k.
    # The B matrix is 3x4.
    # In Block 1, B uses rows 0,1 and cols 0,1.
    # In Block 2, B uses rows 0,1 and cols 2,3.
    # So V index is j*p + k.
    # Global W index: (w_row_offset + k)*n + i
    
    n, m, p = 2, 3, 4
    
    # Helper to set U
    def set_u(i, j, val, col):
        U[i*m + j][col] += val
        
    # Helper to set V
    def set_v(j, k, val, col):
        V[j*p + k][col] += val
        
    # Helper to set W
    def set_w(k, i, val, col):
        W[(w_row_offset + k)*n + i][col] += val

    # M1
    col = col_offset
    set_u(0, 0, 1, col)
    set_u(1, 1, 1, col)
    set_v(0, 0, 1, col)
    set_v(1, 1, 1, col)
    set_w(0, 0, 1, col) # C00
    set_w(1, 1, 1, col) # C11

    # M2
    col = col_offset + 1
    set_u(1, 0, 1, col)
    set_u(1, 1, 1, col)
    set_v(0, 0, 1, col)
    set_w(0, 1, 1, col) # C10
    set_w(1, 1, -1, col) # C11

    # M3
    col = col_offset + 2
    set_u(0, 0, 1, col)
    set_v(1, 0, 1, col)
    set_v(1, 1, -1, col)
    set_w(1, 0, 1, col) # C01
    set_w(0, 1, 1, col) # C10

    # M4
    col = col_offset + 3
    set_u(1, 1, 1, col)
    set_v(0, 1, 1, col)
    set_v(0, 0, -1, col)
    set_w(0, 0, 1, col) # C00
    set_w(0, 1, 1, col) # C10

    # M5
    col = col_offset + 4
    set_u(0, 0, 1, col)
    set_u(0, 1, 1, col)
    set_v(1, 1, 1, col)
    set_w(0, 0, -1, col) # C00
    set_w(1, 0, 1, col) # C01

    # M6
    col = col_offset + 5
    set_u(1, 0, 1, col)
    set_u(0, 0, -1, col)
    set_v(0, 0, 1, col)
    set_v(0, 1, 1, col)
    set_w(1, 1, 1, col) # C11

    # M7
    col = col_offset + 6
    set_u(0, 1, 1, col)
    set_u(1, 1, -1, col)
    set_v(1, 0, 1, col)
    set_v(1, 1, 1, col)
    set_w(0, 0, 1, col) # C00
    set_w(1, 1, -1, col) # C11

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 22 construction
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: Strassen for A[:, 0:2] * B[0:2, 0:2]
    # A cols 0,1. B rows 0,1 cols 0,1. W rows 0,1 (k=0,1)
    _add_strassen_block(U, V, W, 0, 0, 0, 0)
    
    # Block 2: Strassen for A[:, 0:2] * B[0:2, 2:4]
    # A cols 0,1. B rows 0,1 cols 2,3. W rows 2,3 (k=2,3)
    _add_strassen_block(U, V, W, 7, 0, 2, 4)
    
    # Block 3: Schoolbook for A[:, 2] * B[2, :]
    # A col 2. B row 2 cols 0,1,2,3. W all rows.
    # Terms: A[i,2] * B[2,k] -> C[i,k]
    col = 14
    for i in range(n):
        for k in range(p):
            U[i*m + 2][col] = 1
            V[2*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    # Convert to floats for output
    U_out = [[float(u) for u in row] for row in U]
    V_out = [[float(v) for v in row] for row in V]
    W_out = [[float(w) for w in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}

# EVOLVE-BLOCK-END
