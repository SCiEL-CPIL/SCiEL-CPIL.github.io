"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 4
    rank = 28
    
    # U: (n*m, rank) = (8, 28)
    # V: (m*p, rank) = (16, 28)
    # W: (p*n, rank) = (8, 28)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(offset, a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base):
        """
        Adds a rank-7 Strassen decomposition for a 2x2 * 2x2 block.
        
        A sub-block: rows [a_row_base, a_row_base+1], cols [a_col_base, a_col_base+1]
        B sub-block: rows [b_row_base, b_row_base+1], cols [b_col_base, b_col_base+1]
        C sub-block: rows [c_row_base, c_row_base+1], cols [c_col_base, c_col_base+1]
        
        Global Indices:
        A[i, k] -> U[i*m + k]
        B[k, j] -> V[k*p + j]
        C[i, j] -> W[j*n + i]  (Trace convention: W[k*n+i, r] where k is col index j, i is row index i)
        """
        
        # A indices in U
        a00_u = a_row_base * m + a_col_base
        a01_u = a_row_base * m + (a_col_base + 1)
        a10_u = (a_row_base + 1) * m + a_col_base
        a11_u = (a_row_base + 1) * m + (a_col_base + 1)
        
        # B indices in V
        b00_v = b_row_base * p + b_col_base
        b01_v = b_row_base * p + (b_col_base + 1)
        b10_v = (b_row_base + 1) * p + b_col_base
        b11_v = (b_row_base + 1) * p + (b_col_base + 1)
        
        # C indices in W (j*n + i)
        # c_row_base corresponds to i, c_col_base corresponds to j
        c00_w = c_col_base * n + c_row_base
        c01_w = (c_col_base + 1) * n + c_row_base
        c10_w = c_col_base * n + (c_row_base + 1)
        c11_w = (c_col_base + 1) * n + (c_row_base + 1)
        
        # Strassen Products:
        # M1 = (a00 + a11) * (b00 + b11)
        U[a00_u][offset + 0] += 1
        U[a11_u][offset + 0] += 1
        V[b00_v][offset + 0] += 1
        V[b11_v][offset + 0] += 1
        W[c00_w][offset + 0] += 1
        W[c11_w][offset + 0] += 1
        
        # M2 = (a10 + a11) * b00
        U[a10_u][offset + 1] += 1
        U[a11_u][offset + 1] += 1
        V[b00_v][offset + 1] += 1
        W[c10_w][offset + 1] += 1
        W[c11_w][offset + 1] -= 1
        
        # M3 = a00 * (b01 - b11)
        U[a00_u][offset + 2] += 1
        V[b01_v][offset + 2] += 1
        V[b11_v][offset + 2] -= 1
        W[c01_w][offset + 2] += 1
        W[c11_w][offset + 2] += 1
        
        # M4 = a11 * (b10 - b00)
        U[a11_u][offset + 3] += 1
        V[b10_v][offset + 3] += 1
        V[b00_v][offset + 3] -= 1
        W[c00_w][offset + 3] += 1
        W[c10_w][offset + 3] += 1
        
        # M5 = (a00 + a01) * b11
        U[a00_u][offset + 4] += 1
        U[a01_u][offset + 4] += 1
        V[b11_v][offset + 4] += 1
        W[c00_w][offset + 4] -= 1
        W[c01_w][offset + 4] += 1
        
        # M6 = (a10 - a00) * (b00 + b01)
        U[a10_u][offset + 5] += 1
        U[a00_u][offset + 5] -= 1
        V[b00_v][offset + 5] += 1
        V[b01_v][offset + 5] += 1
        W[c11_w][offset + 5] += 1
        
        # M7 = (a01 - a11) * (b10 + b11)
        U[a01_u][offset + 6] += 1
        U[a11_u][offset + 6] -= 1
        V[b10_v][offset + 6] += 1
        V[b11_v][offset + 6] += 1
        W[c00_w][offset + 6] += 1
        
    # Decomposition of 2x4 * 4x4 into four 2x2 * 2x2 blocks
    # A is 2x4: A[:, 0:2] is Left, A[:, 2:4] is Right
    # B is 4x4: B[0:2, :] is Top, B[2:4, :] is Bottom
    # C is 2x4: C[:, 0:2] is Left, C[:, 2:4] is Right
    
    # Block 1: A_Left * B_TopLeft -> C_Left
    # A rows 0-1, cols 0-1; B rows 0-1, cols 0-1; C rows 0-1, cols 0-1
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 2: A_Right * B_BottomLeft -> C_Left
    # A rows 0-1, cols 2-3; B rows 2-3, cols 0-1; C rows 0-1, cols 0-1
    add_strassen_block(7, 0, 2, 2, 0, 0, 0)
    
    # Block 3: A_Left * B_TopRight -> C_Right
    # A rows 0-1, cols 0-1; B rows 0-1, cols 2-3; C rows 0-1, cols 2-3
    add_strassen_block(14, 0, 0, 0, 2, 0, 2)
    
    # Block 4: A_Right * B_BottomRight -> C_Right
    # A rows 0-1, cols 2-3; B rows 2-3, cols 2-3; C rows 0-1, cols 2-3
    add_strassen_block(21, 0, 2, 2, 2, 0, 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
