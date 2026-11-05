import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def _construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=5
    
    # Block partitions
    # J blocks for m=5
    J_blocks = [[0, 1], [2, 3], [4]]
    # K blocks for p=5
    K_blocks = [[0, 1], [2, 3], [4]]
    
    # Initialize matrices with rank 46
    # 18 + 18 + 10 = 46
    rank = 46
    U = [[0] * rank for _ in range(n * m)] # 10 x 46
    V = [[0] * rank for _ in range(m * p)] # 25 x 46
    W = [[0] * rank for _ in range(p * n)] # 10 x 46
    
    col_idx = 0
    
    def add_strassen_2x2x2(i_offset, j_indices, k_indices):
        nonlocal col_idx
        # j_indices, k_indices are lists of 2 global indices
        j0, j1 = j_indices
        k0, k1 = k_indices
        
        # M1 = (A00+A11)(B00+B11)
        # A00 is A[i_offset, j0], A11 is A[i_offset+1, j1]
        # B00 is B[j0, k0], B11 is B[j1, k1]
        # C00 is C[i_offset, k0], C11 is C[i_offset+1, k1]
        
        # Term 1
        U[i_offset * m + j0][col_idx] = 1
        U[(i_offset + 1) * m + j1][col_idx] = 1
        V[j0 * p + k0][col_idx] = 1
        V[j1 * p + k1][col_idx] = 1
        W[k0 * n + i_offset][col_idx] = 1       # C00
        W[k1 * n + (i_offset + 1)][col_idx] = 1 # C11
        col_idx += 1
        
        # Term 2: M2 = (A10+A11)B00
        # A10 is A[i_offset+1, j0], A11 is A[i_offset+1, j1]
        # B00 is B[j0, k0]
        # C10 is C[i_offset+1, k0]
        U[(i_offset + 1) * m + j0][col_idx] = 1
        U[(i_offset + 1) * m + j1][col_idx] = 1
        V[j0 * p + k0][col_idx] = 1
        W[k0 * n + (i_offset + 1)][col_idx] = 1 # C10
        col_idx += 1
        
        # Term 3: M3 = A00(B10-B11)
        # A00 is A[i_offset, j0]
        # B10 is B[j1, k0], B11 is B[j1, k1]
        # C01 is C[i_offset, k1]
        U[i_offset * m + j0][col_idx] = 1
        V[j1 * p + k0][col_idx] = 1
        V[j1 * p + k1][col_idx] = -1
        W[k1 * n + i_offset][col_idx] = 1 # C01
        col_idx += 1
        
        # Term 4: M4 = A11(B01-B00)
        # A11 is A[i_offset+1, j1]
        # B01 is B[j0, k1], B00 is B[j0, k0]
        # C00 is C[i_offset, k0], C10 is C[i_offset+1, k0]
        U[(i_offset + 1) * m + j1][col_idx] = 1
        V[j0 * p + k1][col_idx] = 1
        V[j0 * p + k0][col_idx] = -1
        W[k0 * n + i_offset][col_idx] = 1       # C00
        W[k0 * n + (i_offset + 1)][col_idx] = 1 # C10
        col_idx += 1
        
        # Term 5: M5 = (A00+A01)B11
        # A00 is A[i_offset, j0], A01 is A[i_offset, j1]
        # B11 is B[j1, k1]
        # C00 is C[i_offset, k0], C01 is C[i_offset, k1]
        U[i_offset * m + j0][col_idx] = 1
        U[i_offset * m + j1][col_idx] = 1
        V[j1 * p + k1][col_idx] = 1
        W[k0 * n + i_offset][col_idx] = -1 # C00
        W[k1 * n + i_offset][col_idx] = 1  # C01
        col_idx += 1
        
        # Term 6: M6 = (A10-A00)(B00+B10)
        # A10 is A[i_offset+1, j0], A00 is A[i_offset, j0]
        # B00 is B[j0, k0], B10 is B[j1, k0]
        # C11 is C[i_offset+1, k1]
        U[(i_offset + 1) * m + j0][col_idx] = 1
        U[i_offset * m + j0][col_idx] = -1
        V[j0 * p + k0][col_idx] = 1
        V[j1 * p + k0][col_idx] = 1
        W[k1 * n + (i_offset + 1)][col_idx] = 1 # C11
        col_idx += 1
        
        # Term 7: M7 = (A01-A11)(B11+B01)
        # A01 is A[i_offset, j1], A11 is A[i_offset+1, j1]
        # B11 is B[j1, k1], B01 is B[j0, k1]
        # C00 is C[i_offset, k0], C11 is C[i_offset+1, k1]
        U[i_offset * m + j1][col_idx] = 1
        U[(i_offset + 1) * m + j1][col_idx] = -1
        V[j1 * p + k1][col_idx] = 1
        V[j0 * p + k1][col_idx] = 1
        W[k0 * n + i_offset][col_idx] = 1       # C00
        W[k1 * n + (i_offset + 1)][col_idx] = 1 # C11
        col_idx += 1

    def add_schoolbook_2x1x2(i_offset, j_index, k_indices):
        nonlocal col_idx
        # A is 2x1, B is 1x2
        # A[i_offset, j_index], A[i_offset+1, j_index]
        # B[j_index, k_indices[0]], B[j_index, k_indices[1]]
        # C[i_offset, k0], C[i_offset, k1], C[i_offset+1, k0], C[i_offset+1, k1]
        
        k0, k1 = k_indices
        
        # C00 = A00 B00
        U[i_offset * m + j_index][col_idx] = 1
        V[j_index * p + k0][col_idx] = 1
        W[k0 * n + i_offset][col_idx] = 1
        col_idx += 1
        
        # C01 = A00 B01
        U[i_offset * m + j_index][col_idx] = 1
        V[j_index * p + k1][col_idx] = 1
        W[k1 * n + i_offset][col_idx] = 1
        col_idx += 1
        
        # C10 = A10 B00
        U[(i_offset + 1) * m + j_index][col_idx] = 1
        V[j_index * p + k0][col_idx] = 1
        W[k0 * n + (i_offset + 1)][col_idx] = 1
        col_idx += 1
        
        # C11 = A10 B01
        U[(i_offset + 1) * m + j_index][col_idx] = 1
        V[j_index * p + k1][col_idx] = 1
        W[k1 * n + (i_offset + 1)][col_idx] = 1
        col_idx += 1

    def add_schoolbook_2x2x1(i_offset, j_indices, k_index):
        nonlocal col_idx
        # A is 2x2, B is 2x1
        # A[i_offset, j0], A[i_offset, j1]
        # A[i_offset+1, j0], A[i_offset+1, j1]
        # B[j0, k_index], B[j1, k_index]
        # C[i_offset, k_index], C[i_offset+1, k_index]
        
        j0, j1 = j_indices
        
        # C00 = A00 B00 + A01 B10
        U[i_offset * m + j0][col_idx] = 1
        V[j0 * p + k_index][col_idx] = 1
        W[k_index * n + i_offset][col_idx] = 1
        col_idx += 1
        
        U[i_offset * m + j1][col_idx] = 1
        V[j1 * p + k_index][col_idx] = 1
        W[k_index * n + i_offset][col_idx] = 1
        col_idx += 1
        
        # C10 = A10 B00 + A11 B10
        U[(i_offset + 1) * m + j0][col_idx] = 1
        V[j0 * p + k_index][col_idx] = 1
        W[k_index * n + (i_offset + 1)][col_idx] = 1
        col_idx += 1
        
        U[(i_offset + 1) * m + j1][col_idx] = 1
        V[j1 * p + k_index][col_idx] = 1
        W[k_index * n + (i_offset + 1)][col_idx] = 1
        col_idx += 1

    def add_schoolbook_2x1x1(i_offset, j_index, k_index):
        nonlocal col_idx
        # A is 2x1, B is 1x1
        # C is 2x1
        
        # C00 = A00 B00
        U[i_offset * m + j_index][col_idx] = 1
        V[j_index * p + k_index][col_idx] = 1
        W[k_index * n + i_offset][col_idx] = 1
        col_idx += 1
        
        # C10 = A10 B00
        U[(i_offset + 1) * m + j_index][col_idx] = 1
        V[j_index * p + k_index][col_idx] = 1
        W[k_index * n + (i_offset + 1)][col_idx] = 1
        col_idx += 1

    # Iterate over blocks
    # i_offset is always 0 because n=2, and the blocks cover the full i dimension?
    # Wait, the blocks are for j and k. i is always 0..1.
    # So i_offset is 0 for all blocks.
    
    # K1 block (cols 0,1)
    # J1 (0,1) x K1 (0,1) -> 2x2x2 Strassen
    add_strassen_2x2x2(0, J_blocks[0], K_blocks[0])
    # J2 (2,3) x K1 (0,1) -> 2x2x2 Strassen
    add_strassen_2x2x2(0, J_blocks[1], K_blocks[0])
    # J3 (4) x K1 (0,1) -> 2x1x2 Schoolbook
    add_schoolbook_2x1x2(0, J_blocks[2][0], K_blocks[0])
    
    # K2 block (cols 2,3)
    # J1 (0,1) x K2 (2,3) -> 2x2x2 Strassen
    add_strassen_2x2x2(0, J_blocks[0], K_blocks[1])
    # J2 (2,3) x K2 (2,3) -> 2x2x2 Strassen
    add_strassen_2x2x2(0, J_blocks[1], K_blocks[1])
    # J3 (4) x K2 (2,3) -> 2x1x2 Schoolbook
    add_schoolbook_2x1x2(0, J_blocks[2][0], K_blocks[1])
    
    # K3 block (col 4)
    # J1 (0,1) x K3 (4) -> 2x2x1 Schoolbook
    add_schoolbook_2x2x1(0, J_blocks[0], K_blocks[2][0])
    # J2 (2,3) x K3 (4) -> 2x2x1 Schoolbook
    add_schoolbook_2x2x1(0, J_blocks[1], K_blocks[2][0])
    # J3 (4) x K3 (4) -> 2x1x1 Schoolbook
    add_schoolbook_2x1x1(0, J_blocks[2][0], K_blocks[2][0])
    
    # Verify col_idx is 46
    if col_idx != 46:
        # Fallback to schoolbook if something went wrong
        return _schoolbook(n, m, p)
        
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
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
    try:
        return _construct()
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
