"""Rewritten program: rank-98 decomposition of 4x4x8 matrix multiplication via recursive Strassen."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_2x2x2():
    """Return (U,V,W) each 4x7 with entries in {0, 1/2, 1} for 2x2x2 Strassen."""
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    # Flattened 2x2 matrices: row-major, index = i*2+j for A/B, i*2+j for C output
    # A = [[a00,a01],[a10,a11]] -> [0,1,2,3]
    # B = [[b00,b01],[b10,b11]] -> [0,1,2,3]
    # C = [[c00,c01],[c10,c11]] -> [0,1,2,3]
    
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # U: coefficients for A
    U[0, 0] = 1  # a00
    U[1, 0] = 0
    U[2, 0] = 0
    U[3, 0] = 1  # a11
    
    U[0, 1] = 0
    U[1, 1] = 1  # a10
    U[2, 1] = 0
    U[3, 1] = 1  # a11
    
    U[0, 2] = 1  # a00
    U[1, 2] = 0
    U[2, 2] = 0
    U[3, 2] = 0
    
    U[0, 3] = 0
    U[1, 3] = 0
    U[2, 3] = 0
    U[3, 3] = 1  # a11
    
    U[0, 4] = 1  # a00
    U[1, 4] = 1  # a01
    U[2, 4] = 0
    U[3, 4] = 0
    
    U[0, 5] = 1  # a10 - a00: a10
    U[1, 5] = -1 # a00
    U[2, 5] = 0
    U[3, 5] = 0
    
    U[0, 6] = 0
    U[1, 6] = 1  # a01 - a11: a01
    U[2, 6] = 0
    U[3, 6] = -1 # a11
    
    # V: coefficients for B
    V[0, 0] = 1  # b00
    V[1, 0] = 0
    V[2, 0] = 0
    V[3, 0] = 1  # b11
    
    V[0, 1] = 1  # b00
    V[1, 1] = 0
    V[2, 1] = 0
    V[3, 1] = 0
    
    V[0, 2] = 0
    V[1, 2] = 1  # b01
    V[2, 2] = 0
    V[3, 2] = -1 # b11
    
    V[0, 3] = -1 # b10 - b00: -b00
    V[1, 3] = 0
    V[2, 3] = 1  # b10
    V[3, 3] = 0
    
    V[0, 4] = 0
    V[1, 4] = 0
    V[2, 4] = 0
    V[3, 4] = 1  # b11
    
    V[0, 5] = 1  # b00 + b01
    V[1, 5] = 1
    V[2, 5] = 0
    V[3, 5] = 0
    
    V[0, 6] = 0
    V[1, 6] = 0
    V[2, 6] = 1  # b10 + b11
    V[3, 6] = 1
    
    # W: coefficients for C
    # c00 = M1 + M4 - M5 - M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M7
    
    W[0, 0] = 1  # M1
    W[1, 0] = 0
    W[2, 0] = 0
    W[3, 0] = 1  # M7
    
    W[0, 1] = 0
    W[1, 1] = 0
    W[2, 1] = 1  # M2
    W[3, 1] = -1 # M2 (c11 = -M2 + ...)
    
    W[0, 2] = 0
    W[1, 2] = 1  # M3 (c01)
    W[2, 2] = 0
    W[3, 2] = 1  # M3 (c11)
    
    W[0, 3] = 1  # M4 (c00)
    W[1, 3] = 0
    W[2, 3] = 1  # M4 (c10)
    W[3, 3] = 0
    
    W[0, 4] = -1 # M5 (c00)
    W[1, 4] = 1  # M5 (c01)
    W[2, 4] = 0
    W[3, 4] = 0
    
    W[0, 5] = 0
    W[1, 5] = 0
    W[2, 5] = 0
    W[3, 5] = 0
    
    W[0, 6] = -1 # M7 (c00)
    W[1, 6] = 0
    W[2, 6] = 0
    W[3, 6] = 1  # M7 (c11)
    
    return U, V, W

def _compose_strassen(U1, V1, W1, U2, V2, W2):
    """
    Compose two Strassen decompositions.
    First level: U1 (4x7), V1 (4x7), W1 (4x7) for block-level
    Second level: U2 (4x7), V2 (4x7), W2 (4x7) for scalar-level within each block
    
    Returns U (16x49), V (16x49), W (16x49) for 4x4x4.
    """
    # The first level has 7 terms. Each term involves linear combinations of 2x2 blocks.
    # U1[i_block, t] where i_block is 0..3 (flattened 2x2 block of A) and t is 0..6 (Strassen term)
    # Actually, for 4x4x4, A is 4x4, split into 2x2 blocks.
    # The U matrix for the full 4x4x4 will have rows indexed by (block_row, block_col, inner_row, inner_col)
    # and columns by (outer_term, inner_term).
    
    # Let me think more carefully.
    # 4x4 matrix A has 16 entries. We split into 2x2 blocks: A = [[A00, A01],[A10, A11]]
    # where each Aij is 2x2.
    # The first-level Strassen computes 7 linear combinations of blocks.
    # For term t, the coefficient of block Aij is U1[ij, t] (where ij is flattened 0..3).
    # Then each block product involves multiplying 2x2 matrices, decomposed by second-level Strassen.
    
    # The full U matrix has 16 rows (one per entry of A) and 49 columns (7x7 terms).
    # Row index: (block_idx, entry_in_block) where block_idx = 0..3, entry_in_block = 0..3
    # Column index: (t1, t2) where t1 = 0..6 (outer Strassen term), t2 = 0..6 (inner Strassen term)
    
    U_full = np.zeros((16, 49), dtype=np.float64)
    V_full = np.zeros((16, 49), dtype=np.float64)
    W_full = np.zeros((16, 49), dtype=np.float64)
    
    for t1 in range(7):
        for t2 in range(7):
            col = t1 * 7 + t2
            # For each block Aij, the coefficient in term t1 is U1[ij, t1]
            # Within that block, the inner Strassen term t2 has coefficient U2[e, t2] for entry e
            for ij in range(4):
                for e in range(4):
                    row = ij * 4 + e
                    U_full[row, col] = U1[ij, t1] * U2[e, t2]
                    V_full[row, col] = V1[ij, t1] * V2[e, t2]
            # W: output C has 16 entries. Split into 2x2 blocks Cij.
            # The outer Strassen term t1 contributes to output blocks with coefficient W1[ij, t1]
            # Within each output block, inner term t2 contributes with coefficient W2[e, t2]
            for ij in range(4):
                for e in range(4):
                    row = ij * 4 + e
                    W_full[row, col] = W1[ij, t1] * W2[e, t2]
    
    return U_full, V_full, W_full

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Get 2x2x2 Strassen
    U1, V1, W1 = _strassen_2x2x2()
    
    # Compose for 4x4x4: rank 49
    U44, V44, W44 = _compose_strassen(U1, V1, W1, U1, V1, W1)
    
    # For 4x4x8: split the 8 columns of B into two 4-column halves.
    # B = [B_left | B_right] where each is 4x4.
    # C = A * B = A * B_left + A * B_right
    # We use the same U for both halves (since A is the same), but different V and W.
    # V44 has 16 rows (4*4), W44 has 16 rows (4*4).
    # For the left half: V_left = V44, W_left = W44
    # For the right half: V_right = V44, W_right = W44
    # But W indices: output is 4x8 = 32 entries. W has p*n = 8*4 = 32 rows.
    # The convention is W[k*n + i, r] where k is column of C (0..7), i is row of C (0..3).
    # For the left 4 columns (k=0..3), W entries are at indices k*4+i for k in 0..3.
    # For the right 4 columns (k=4..7), W entries are at indices k*4+i for k in 4..7.
    
    # U: 16 rows (n*m = 4*4), 98 columns
    U = np.zeros((16, 98), dtype=np.float64)
    V = np.zeros((32, 98), dtype=np.float64)  # m*p = 4*8 = 32
    W = np.zeros((32, 98), dtype=np.float64)  # p*n = 8*4 = 32
    
    # Left half (columns 0-3 of B, 0-3 of C)
    U[:, :49] = U44
    V[:16, :49] = V44  # rows 0-15 of V correspond to j*p+k for j in 0..3, k in 0..3 (left half)
    # V row index: j*p + k where p=8. For left half, k=0..3.
    # V44 has 16 rows indexed by (j_inner, k_inner) where j_inner=0..3 (rows of B block), k_inner=0..3.
    # In the full V, row = j*8 + k. For left half, k = 0..3.
    # So V[j*8 + k, col] = V44[j*4 + k, col] for j=0..3, k=0..3.
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, :49] = V44[j * 4 + k, :49]
    
    # W for left half: output rows k*4+i for k=0..3 (left columns of C)
    # W44 has 16 rows indexed by (k_inner, i_inner).
    # In full W, row = k*4 + i where k is column of C (0..7), i is row of C (0..3).
    # For left half, k=0..3.
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, :49] = W44[k * 4 + i, :49]
    
    # Right half (columns 4-7 of B, 4-7 of C)
    U[:, 49:] = U44
    for j in range(4):
        for k in range(4):
            V[j * 8 + (k + 4), 49:] = V44[j * 4 + k, 49:]
    
    for k in range(4):
        for i in range(4):
            W[(k + 4) * 4 + i, 49:] = W44[k * 4 + i, 49:]
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
