import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    if time.time() > deadline:
        return schoolbook()
        
    # Rank 16 construction
    rank = 16
    U = [[Fraction(0)] * rank for _ in range(n * m)]  # 6 x 16
    V = [[Fraction(0)] * rank for _ in range(m * p)]  # 12 x 16
    W = [[Fraction(0)] * rank for _ in range(p * n)]  # 8 x 16
    
    # Helper to add a Strassen 2x2x2 decomposition
    # a_offset: row offset in A (0 or 1)
    # b_offset: col offset in B (0 or 1)
    # c_offset: col offset in C (0 or 1)
    # b_row_offset: row offset in B (0 or 1)
    # a_col_offset: col offset in A (0 or 1)
    
    def add_strassen(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off, r_start):
        # Strassen terms for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # Indices in U (A): (a_row_off + i) * m + (a_col_off + j)
        # Indices in V (B): (b_row_off + k) * p + (b_col_off + l)
        # Indices in W (C): (c_col_off + l) * n + (c_row_off + i)
        
        # M1
        r = r_start
        U[(a_row_off)*m + (a_col_off+0)][r] += 1
        U[(a_row_off+1)*m + (a_col_off+1)][r] += 1
        V[(b_row_off+0)*p + (b_col_off+0)][r] += 1
        V[(b_row_off+1)*p + (b_col_off+1)][r] += 1
        W[(c_col_off+0)*n + (c_row_off+0)][r] += 1
        W[(c_col_off+1)*n + (c_row_off+1)][r] += 1
        # M2
        r = r_start + 1
        U[(a_row_off+1)*m + (a_col_off+0)][r] += 1
        U[(a_row_off+1)*m + (a_col_off+1)][r] += 1
        V[(b_row_off+0)*p + (b_col_off+0)][r] += 1
        W[(c_col_off+1)*n + (c_row_off+0)][r] += 1
        W[(c_col_off+1)*n + (c_row_off+1)][r] += 1
        # M3
        r = r_start + 2
        U[(a_row_off+0)*m + (a_col_off+0)][r] += 1
        V[(b_row_off+1)*p + (b_col_off+0)][r] += 1
        V[(b_row_off+1)*p + (b_col_off+1)][r] += -1
        W[(c_col_off+0)*n + (c_row_off+0)][r] += 1
        W[(c_col_off+1)*n + (c_row_off+0)][r] += 1
        # M4
        r = r_start + 3
        U[(a_row_off+1)*m + (a_col_off+1)][r] += 1
        V[(b_row_off+0)*p + (b_col_off+1)][r] += 1
        V[(b_row_off+0)*p + (b_col_off+0)][r] += -1
        W[(c_col_off+0)*n + (c_row_off+0)][r] += 1
        W[(c_col_off+1)*n + (c_row_off+1)][r] += 1
        # M5
        r = r_start + 4
        U[(a_row_off+0)*m + (a_col_off+0)][r] += 1
        U[(a_row_off+0)*m + (a_col_off+1)][r] += 1
        V[(b_row_off+1)*p + (b_col_off+1)][r] += 1
        W[(c_col_off+0)*n + (c_row_off+0)][r] += -1
        W[(c_col_off+1)*n + (c_row_off+0)][r] += 1
        # M6
        r = r_start + 5
        U[(a_row_off+1)*m + (a_col_off+0)][r] += 1
        U[(a_row_off+0)*m + (a_col_off+0)][r] += -1
        V[(b_row_off+0)*p + (b_col_off+0)][r] += 1
        V[(b_row_off+1)*p + (b_col_off+0)][r] += 1
        W[(c_col_off+1)*n + (c_row_off+1)][r] += 1
        # M7
        r = r_start + 6
        U[(a_row_off+0)*m + (a_col_off+1)][r] += 1
        U[(a_row_off+1)*m + (a_col_off+1)][r] += -1
        V[(b_row_off+0)*p + (b_col_off+1)][r] += 1
        V[(b_row_off+1)*p + (b_col_off+1)][r] += 1
        W[(c_col_off+0)*n + (c_row_off+1)][r] += 1
        W[(c_col_off+1)*n + (c_row_off+1)][r] += -1

    # Block 1: C cols 0,1
    # A cols 0,1 (a_col_off=0), B rows 0,1 (b_row_off=0), B cols 0,1 (b_col_off=0)
    add_strassen(0, 0, 0, 0, 0, 0, 0)
    # A col 2 (a_col_off=2? No, a_col_off is offset in the 2x2 block. 
    # Wait, my add_strassen assumes 2x2 block at offset.
    # For A col 2, it's not in a 2x2 block with col 0,1.
    # I need a separate rank 1 term.
    
    # Rank 1 term for A[*, 2] * B[2, 0:2]
    r = 7
    # U: A rows 0,1 col 2
    U[0*m + 2][r] += 1
    U[1*m + 2][r] += 1
    # V: B row 2 cols 0,1
    V[2*p + 0][r] += 1
    V[2*p + 1][r] += 1
    # W: C rows 0,1 cols 0,1
    W[0*n + 0][r] += 1
    W[0*n + 1][r] += 1
    W[1*n + 0][r] += 1
    W[1*n + 1][r] += 1
    
    # Block 2: C cols 2,3
    # A cols 0,1 (a_col_off=0), B rows 0,1 (b_row_off=0), B cols 2,3 (b_col_off=2)
    add_strassen(0, 0, 0, 2, 0, 2, 8)
    
    # Rank 1 term for A[*, 2] * B[2, 2:4]
    r = 15
    # U: A rows 0,1 col 2
    U[0*m + 2][r] += 1
    U[1*m + 2][r] += 1
    # V: B row 2 cols 2,3
    V[2*p + 2][r] += 1
    V[2*p + 3][r] += 1
    # W: C rows 0,1 cols 2,3
    W[2*n + 0][r] += 1
    W[2*n + 1][r] += 1
    W[3*n + 0][r] += 1
    W[3*n + 1][r] += 1

    return {"U": U, "V": V, "W": W}
