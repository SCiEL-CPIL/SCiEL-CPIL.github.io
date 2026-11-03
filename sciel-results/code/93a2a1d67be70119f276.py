import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank is 85 (49 from 4x4x4 Strassen + 36 from linear terms)
    # We will build the matrices dynamically.
    
    # Helper to add a rank-1 term defined by bilinear forms
    # U_idx = i*m + j
    # V_idx = j*p + k
    # W_idx = k*n + i
    # Term: L_A(A) * L_B(B) contributes w[i,k] to C[i,k]
    # L_A is a dict {(i,j): coeff}
    # L_B is a dict {(j,k): coeff}
    # w is a dict {(i,k): coeff}
    
    # We will use a list of terms to accumulate, then flatten
    # Or just build the matrices directly with a fixed rank
    # Max rank estimate: 49 + 16 + 16 + 4 = 85
    
    rank = 85
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(L_A, L_B, W_coeffs):
        nonlocal col
        if col >= rank:
            return
        for (i, j), ca in L_A.items():
            if ca != 0:
                U[i * m + j][col] += ca
        for (j, k), cb in L_B.items():
            if cb != 0:
                V[j * p + k][col] += cb
        for (i, k), cw in W_coeffs.items():
            if cw != 0:
                W[k * n + i][col] += cw
        col += 1

    # 1. 4x4x4 Core using Strassen (Rank 49)
    # Strassen for 2x2 has 7 terms.
    # 4x4 is 4 blocks of 2x2.
    # We apply Strassen to the 4x4 blocks.
    # Block indices:
    # A11 (0:2, 0:2), A12 (0:2, 2:4)
    # A21 (2:4, 0:2), A22 (2:4, 2:4)
    
    # We need to map the 2x2 Strassen terms to the 4x4 indices.
    # For each pair of 2x2 blocks, we generate 7 terms.
    # There are 4 pairs? No, Strassen on 4x4 uses 7 multiplications of 2x2 matrices.
    # The 2x2 matrices are sums of the 4 blocks.
    
    # Let's define the 7 bilinear forms for 2x2 Strassen.
    # M1: (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
    # M2: (A21+A22)B11 -> C21 += 1, C22 -= 1
    # M3: A11(B12-B22) -> C12 += 1, C22 += 1
    # M4: A22(B21-B11) -> C11 += 1, C21 += 1
    # M5: (A11+A12)B22 -> C11 -= 1, C12 += 1
    # M6: (A21-A11)(B11+B12) -> C22 += 1
    # M7: (A12-A22)(B21+B22) -> C11 += 1, C12 -= 1? 
    # Let's verify Strassen formulas.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We will apply this to the 4x4 matrix by treating it as a 2x2 block matrix.
    # The "A11" of the block matrix is the top-left 2x2 submatrix of the 4x4 A.
    # Indices in 4x4:
    # A_block_11: i in 0..1, j in 0..1
    # A_block_12: i in 0..1, j in 2..3
    # A_block_21: i in 2..3, j in 0..1
    # A_block_22: i in 2..3, j in 2..3
    
    # We define the terms for the block multiplication.
    # Each term is a bilinear form on the 4x4 A and 4x4 B.
    
    # Term 1: (A11+A22)(B11+B22)
    # L_A: sum of A[0:2, 0:2] and A[2:4, 2:4]
    # L_B: sum of B[0:2, 0:2] and B[2:4, 2:4]
    # W: C[0:2, 0:2] += 1, C[2:4, 2:4] += 1
    
    # We will generate these programmatically.
    
    def get_block(i_start, i_end, j_start, j_end):
        return [(i, j) for i in range(i_start, i_end) for j in range(j_start, j_end)]

    # Define the 7 Strassen terms for the 4x4 core
    # We use 0-based indexing for the 4x4 blocks
    # B11: 0:2, 0:2
    # B12: 0:2, 2:4
    # B21: 2:4, 0:2
    # B22: 2:4, 2:4
    
    # Term 1: (A11+A22)(B11+B22)
    LA = {}
    for idx in get_block(0, 2, 0, 2): LA[idx] = 1
    for idx in get_block(2, 4, 2, 4): LA[idx] = 1
    LB = {}
    for idx in get_block(0, 2, 0, 2): LB[idx] = 1
    for idx in get_block(2, 4, 2, 4): LB[idx] = 1
    WC = {}
    for idx in get_block(0, 2, 0, 2): WC[idx] = 1 # C11
    for idx in get_block(2, 4, 2, 4): WC[idx] = 1 # C22
    add_term(LA, LB, WC)
    
    # Term 2: (A21+A22)B11
    LA = {}
    for idx in get_block(2, 4, 0, 2): LA[idx] = 1
    for idx in get_block(2, 4, 2, 4): LA[idx] = 1
    LB = {}
    for idx in get_block(0, 2, 0, 2): LB[idx] = 1
    WC = {}
    for idx in get_block(2, 4, 0, 2): WC[idx] = 1 # C21
    for idx in get_block(2, 4, 2, 4): WC[idx] = -1 # C22
    add_term(LA, LB, WC)
    
    # Term 3: A11(B12-B22)
    LA = {}
    for idx in get_block(0, 2, 0, 2): LA[idx] = 1
    LB = {}
    for idx in get_block(0, 2, 2, 4): LB[idx] = 1
    for idx in get_block(2, 4, 2, 4): LB[idx] = -1
    WC = {}
    for idx in get_block(0, 2, 2, 4): WC[idx] = 1 # C12
    for idx in get_block(2, 4, 2, 4): WC[idx] = 1 # C22
    add_term(LA, LB, WC)
    
    # Term 4: A22(B21-B11)
    LA = {}
    for idx in get_block(2, 4, 2, 4): LA[idx] = 1
    LB = {}
    for idx in get_block(2, 4, 0, 2): LB[idx] = 1
    for idx in get_block(0, 2, 0, 2): LB[idx] = -1
    WC = {}
    for idx in get_block(0, 2, 0, 2): WC[idx] = 1 # C11
    for idx in get_block(2, 4, 0, 2): WC[idx] = 1 # C21
    add_term(LA, LB, WC)
    
    # Term 5: (A11+A12)B22
    LA = {}
    for idx in get_block(0, 2, 0, 2): LA[idx] = 1
    for idx in get_block(0, 2, 2, 4): LA[idx] = 1
    LB = {}
    for idx in get_block(2, 4, 2, 4): LB[idx] = 1
    WC = {}
    for idx in get_block(0, 2, 0, 2): WC[idx] = -1 # C11
    for idx in get_block(0, 2, 2, 4): WC[idx] = 1 # C12
    add_term(LA, LB, WC)
    
    # Term 6: (A21-A11)(B11+B12)
    LA = {}
    for idx in get_block(2, 4, 0, 2): LA[idx] = 1
    for idx in get_block(0, 2, 0, 2): LA[idx] = -1
    LB = {}
    for idx in get_block(0, 2, 0, 2): LB[idx] = 1
    for idx in get_block(0, 2, 2, 4): LB[idx] = 1
    WC = {}
    for idx in get_block(2, 4, 2, 4): WC[idx] = 1 # C22
    add_term(LA, LB, WC)
    
    # Term 7: (A12-A22)(B21+B22)
    LA = {}
    for idx in get_block(0, 2, 2, 4): LA[idx] = 1
    for idx in get_block(2, 4, 2, 4): LA[idx] = -1
    LB = {}
    for idx in get_block(2, 4, 0, 2): LB[idx] = 1
    for idx in get_block(2, 4, 2, 4): LB[idx] = 1
    WC = {}
    for idx in get_block(0, 2, 0, 2): WC[idx] = 1 # C11
    for idx in get_block(0, 2, 2, 4): WC[idx] = -1 # C12
    add_term(LA, LB, WC)
    
    # Now the linear terms for the 5th column/row
    # A is 4x5. B is 5x5.
    # A12 is 4x1 (col 4). A21 is 1x4 (row 4). A22 is 1x1.
    # Indices:
    # A[0:4, 4] is the 5th column.
    # B[4, 0:4] is the 5th row.
    # B[4, 4] is the corner.
    
    # Part 2: A[0:4, 4] * B[4, 0:4] (4x1 x 1x4) -> Rank 16
    # C[0:4, 0:4] += A[i,4] * B[4,j]
    for i in range(4):
        for j in range(4):
            LA = {(i, 4): 1}
            LB = {(4, j): 1}
            WC = {(i, j): 1}
            add_term(LA, LB, WC)
            
    # Part 3: A[0:4, 0:4] * B[0:4, 4] (4x4 x 4x1) -> Rank 16
    # C[0:4, 4] += A[i,k] * B[k,4]
    # This is sum over k.
    # We can do this with 16 terms?
    # C[i,4] = sum_k A[i,k] B[k,4].
    # Each term A[i,k] B[k,4] is a bilinear form.
    for i in range(4):
        for k in range(4):
            LA = {(i, k): 1}
            LB = {(k, 4): 1}
            WC = {(i, 4): 1}
            add_term(LA, LB, WC)
            
    # Part 4: A[0:4, 4] * B[4, 4] (4x1 x 1x1) -> Rank 4
    # C[i,4] += A[i,4] * B[4,4]
    for i in range(4):
        LA = {(i, 4): 1}
        LB = {(4, 4): 1}
        WC = {(i, 4): 1}
        add_term(LA, LB, WC)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
