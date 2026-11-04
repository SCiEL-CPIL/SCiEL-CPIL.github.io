import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose m=5 into 2+2+1
    # A = [A1, A2, A3]
    # B = [B1; B2; B3]
    # C = A1 B1 + A2 B2 + A3 B3
    
    # Rank contributions:
    # A1 B1: 2x2x6 -> 3 * (2x2x2 Strassen rank 7) = 21
    # A2 B2: 2x2x6 -> 3 * (2x2x2 Strassen rank 7) = 21
    # A3 B3: 2x1x6 -> 2 * 6 = 12
    # Total Rank = 54
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add a 2x2x2 Strassen block
    # A indices: i in 0..1, j in j_offset..j_offset+1
    # B indices: j in j_offset..j_offset+1, k in k_offset..k_offset+1
    # C indices: i in 0..1, k in k_offset..k_offset+1
    def add_strassen_block(j_offset, k_offset):
        nonlocal col_offset
        # Strassen formulas for 2x2x2
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B11-B21)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # U coefficients for A
        # V coefficients for B
        # W coefficients for C
        
        # M1
        U[j_offset*2 + 0][col_offset] += 1   # A11
        U[j_offset*2 + 1][col_offset] += 1   # A22 (Wait, A22 is i=1, j=1 relative to block)
        # A indices: i*m + j.
        # Block A is rows 0,1. Cols j_offset, j_offset+1.
        # A11: i=0, j=j_offset. Index 0*m + j_offset.
        # A22: i=1, j=j_offset+1. Index 1*m + j_offset+1.
        # A12: i=0, j=j_offset+1.
        # A21: i=1, j=j_offset.
        
        # Let's define indices explicitly
        i0, i1 = 0, 1
        j0, j1 = j_offset, j_offset + 1
        k0, k1 = k_offset, k_offset + 1
        
        # M1 = (A11+A22)(B11+B22)
        U[i0*m + j0][col_offset] += 1
        U[i1*m + j1][col_offset] += 1
        V[j0*p + k0][col_offset] += 1
        V[j1*p + k1][col_offset] += 1
        W[k0*n + i0][col_offset] += 1  # C11 += M1
        W[k1*n + i0][col_offset] += 1  # C12 += M1
        W[k0*n + i1][col_offset] += 1  # C21 += M1
        W[k1*n + i1][col_offset] += 1  # C22 += M1
        col_offset += 1
        
        # M2 = (A21+A22)B11
        U[i1*m + j0][col_offset] += 1
        U[i1*m + j1][col_offset] += 1
        V[j0*p + k0][col_offset] += 1
        W[k0*n + i0][col_offset] += 1  # C11 += M2
        W[k1*n + i0][col_offset] += 1  # C12 += M2
        col_offset += 1
        
        # M3 = A11(B21-B22)
        U[i0*m + j0][col_offset] += 1
        V[j1*p + k0][col_offset] += 1
        V[j1*p + k1][col_offset] -= 1
        W[k0*n + i1][col_offset] += 1  # C21 += M3
        W[k1*n + i1][col_offset] += 1  # C22 += M3
        col_offset += 1
        
        # M4 = A22(B11-B21)
        U[i1*m + j1][col_offset] += 1
        V[j0*p + k0][col_offset] += 1
        V[j1*p + k0][col_offset] -= 1
        W[k0*n + i0][col_offset] += 1  # C11 += M4
        W[k1*n + i0][col_offset] += 1  # C12 += M4
        col_offset += 1
        
        # M5 = (A11+A12)B22
        U[i0*m + j0][col_offset] += 1
        U[i0*m + j1][col_offset] += 1
        V[j1*p + k1][col_offset] += 1
        W[k1*n + i0][col_offset] += 1  # C12 += M5
        W[k1*n + i1][col_offset] += 1  # C22 += M5
        col_offset += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[i1*m + j0][col_offset] += 1
        U[i0*m + j0][col_offset] -= 1
        V[j0*p + k0][col_offset] += 1
        V[j0*p + k1][col_offset] += 1
        W[k0*n + i1][col_offset] += 1  # C21 += M6
        W[k1*n + i1][col_offset] += 1  # C22 += M6
        col_offset += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[i0*m + j1][col_offset] += 1
        U[i1*m + j1][col_offset] -= 1
        V[j1*p + k0][col_offset] += 1
        V[j1*p + k1][col_offset] += 1
        W[k0*n + i0][col_offset] += 1  # C11 += M7
        W[k1*n + i0][col_offset] += 1  # C12 += M7
        col_offset += 1

    # Helper to add 2x1x6 block (Schoolbook)
    # A is 2x1 (col j_offset), B is 1x6 (row j_offset, cols k_offset..k_offset+5)
    def add_2x1x6_block(j_offset, k_offset):
        nonlocal col_offset
        # C_i,k = A_i,j * B_j,k
        for i in range(2):
            for k in range(6):
                U[i*m + j_offset][col_offset] += 1
                V[j_offset*p + k_offset + k][col_offset] += 1
                W[(k_offset + k)*n + i][col_offset] += 1
                col_offset += 1

    # Block 1: A1 (cols 0,1) * B1 (rows 0,1)
    # B1 is 2x6. Split into 3 2x2 blocks.
    add_strassen_block(0, 0)
    add_strassen_block(0, 2)
    add_strassen_block(0, 4)
    
    # Block 2: A2 (cols 2,3) * B2 (rows 2,3)
    # B2 is 2x6. Split into 3 2x2 blocks.
    add_strassen_block(2, 0)
    add_strassen_block(2, 2)
    add_strassen_block(2, 4)
    
    # Block 3: A3 (col 4) * B3 (row 4)
    # B3 is 1x6.
    add_2x1x6_block(4, 0)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
