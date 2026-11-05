import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will construct a decomposition using Strassen's algorithm for the 2x2x2 blocks.
    # This yields a rank of 36 for 2x4x5 (4 blocks of 2x2x2 * 7 + 2 blocks of 2x2x1 * 4).
    # This is better than the schoolbook rank of 40.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We map these to the U, V, W matrices.
    # U index: i*m + j. For 2x2 block starting at row i0, col j0:
    # A00 -> (i0, j0), A01 -> (i0, j0+1), A10 -> (i0+1, j0), A11 -> (i0+1, j0+1)
    # V index: j*p + k. For 2x2 block starting at row j0, col k0:
    # B00 -> (j0, k0), B01 -> (j0, k0+1), B10 -> (j0+1, k0), B11 -> (j0+1, k0+1)
    # W index: k*n + i. For 2x2 block starting at row k0, col i0:
    # C00 -> (k0, i0), C01 -> (k0, i0+1), C10 -> (k0+1, i0), C11 -> (k0+1, i0+1)
    
    def add_strassen_block(i0, j0, k0, col_offset):
        # i0, j0, k0 are the top-left indices of the 2x2 blocks in A, B, C.
        # A is 2x2 at (i0, j0)
        # B is 2x2 at (j0, k0)
        # C is 2x2 at (k0, i0)
        
        # M1
        U[i0*m + j0][col_offset] += 1
        U[(i0+1)*m + (j0+1)][col_offset] += 1
        V[j0*p + k0][col_offset] += 1
        V[(j0+1)*p + (k0+1)][col_offset] += 1
        W[k0*n + i0][col_offset] += 1
        W[(k0+1)*n + (i0+1)][col_offset] += 1
        W[k0*n + (i0+1)][col_offset] += -1
        W[(k0+1)*n + i0][col_offset] += 1
        
        # M2
        U[(i0+1)*m + j0][col_offset+1] += 1
        U[(i0+1)*m + (j0+1)][col_offset+1] += 1
        V[j0*p + k0][col_offset+1] += 1
        W[(k0+1)*n + i0][col_offset+1] += 1
        W[(k0+1)*n + (i0+1)][col_offset+1] += -1
        
        # M3
        U[i0*m + j0][col_offset+2] += 1
        V[j0*p + (k0+1)][col_offset+2] += 1
        V[(j0+1)*p + (k0+1)][col_offset+2] += -1
        W[k0*n + (i0+1)][col_offset+2] += 1
        W[(k0+1)*n + (i0+1)][col_offset+2] += 1
        
        # M4
        U[(i0+1)*m + (j0+1)][col_offset+3] += 1
        V[(j0+1)*p + k0][col_offset+3] += 1
        V[j0*p + k0][col_offset+3] += -1
        W[k0*n + i0][col_offset+3] += 1
        W[(k0+1)*n + i0][col_offset+3] += 1
        
        # M5
        U[i0*m + j0][col_offset+4] += 1
        U[i0*m + (j0+1)][col_offset+4] += 1
        V[(j0+1)*p + (k0+1)][col_offset+4] += 1
        W[k0*n + i0][col_offset+4] += -1
        W[k0*n + (i0+1)][col_offset+4] += 1
        
        # M6
        U[(i0+1)*m + j0][col_offset+5] += 1
        U[i0*m + j0][col_offset+5] += -1
        V[j0*p + k0][col_offset+5] += 1
        V[j0*p + (k0+1)][col_offset+5] += 1
        W[(k0+1)*n + (i0+1)][col_offset+5] += 1
        
        # M7
        U[i0*m + (j0+1)][col_offset+6] += 1
        U[(i0+1)*m + (j0+1)][col_offset+6] += -1
        V[(j0+1)*p + k0][col_offset+6] += 1
        V[(j0+1)*p + (k0+1)][col_offset+6] += 1
        W[k0*n + i0][col_offset+6] += 1
        
        return col_offset + 7

    def add_2x2x1_block(i0, j0, k0, col_offset):
        # A is 2x2 at (i0, j0)
        # B is 2x1 at (j0, k0)
        # C is 2x1 at (k0, i0)
        # C0 = A00 B0 + A10 B1
        # C1 = A01 B0 + A11 B1
        # Rank 4
        
        # Term 1: A00 B0 -> C0
        U[i0*m + j0][col_offset] += 1
        V[j0*p + k0][col_offset] += 1
        W[k0*n + i0][col_offset] += 1
        
        # Term 2: A10 B1 -> C0
        U[(i0+1)*m + j0][col_offset+1] += 1
        V[(j0+1)*p + k0][col_offset+1] += 1
        W[k0*n + i0][col_offset+1] += 1
        
        # Term 3: A01 B0 -> C1
        U[i0*m + (j0+1)][col_offset+2] += 1
        V[j0*p + k0][col_offset+2] += 1
        W[k0*n + (i0+1)][col_offset+2] += 1
        
        # Term 4: A11 B1 -> C1
        U[(i0+1)*m + (j0+1)][col_offset+3] += 1
        V[(j0+1)*p + k0][col_offset+3] += 1
        W[k0*n + (i0+1)][col_offset+3] += 1
        
        return col_offset + 4

    col = 0
    # 4 blocks of 2x2x2 for the first 4 columns of C
    col = add_strassen_block(0, 0, 0, col) # A00, B00, C00
    col = add_strassen_block(0, 2, 0, col) # A01, B01, C01 (Wait, indices)
    # A is 2x4. Split into A_L (cols 0,1) and A_R (cols 2,3).
    # B is 4x4 (first 4 cols). Split into B_LL (rows 0,1 cols 0,1), B_LR (rows 0,1 cols 2,3), B_RL (rows 2,3 cols 0,1), B_RR (rows 2,3 cols 2,3).
    # C is 2x4. Split into C_L (cols 0,1) and C_R (cols 2,3).
    
    # C_L = A_L B_LL + A_R B_RL
    # C_R = A_L B_LR + A_R B_RR
    
    # Block 1: A_L (i0=0, j0=0) * B_LL (j0=0, k0=0) -> C_L (k0=0, i0=0)
    col = add_strassen_block(0, 0, 0, col)
    # Block 2: A_R (i0=0, j0=2) * B_RL (j0=2, k0=0) -> C_L (k0=0, i0=0)
    col = add_strassen_block(0, 2, 0, col)
    # Block 3: A_L (i0=0, j0=0) * B_LR (j0=0, k0=2) -> C_R (k0=2, i0=0)
    col = add_strassen_block(0, 0, 2, col)
    # Block 4: A_R (i0=0, j0=2) * B_RR (j0=2, k0=2) -> C_R (k0=2, i0=0)
    col = add_strassen_block(0, 2, 2, col)
    
    # 2 blocks of 2x2x1 for the last column of C (col 4)
    # C_4 = A_L B_L3 + A_R B_R3
    # B_L3 is rows 0,1 col 4. B_R3 is rows 2,3 col 4.
    
    # Block 5: A_L (i0=0, j0=0) * B_L3 (j0=0, k0=4) -> C_4 (k0=4, i0=0)
    col = add_2x2x1_block(0, 0, 4, col)
    # Block 6: A_R (i0=0, j0=2) * B_R3 (j0=2, k0=4) -> C_4 (k0=4, i0=0)
    col = add_2x2x1_block(0, 2, 4, col)
    
    assert col == rank, f"Rank mismatch: {col} != {rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
