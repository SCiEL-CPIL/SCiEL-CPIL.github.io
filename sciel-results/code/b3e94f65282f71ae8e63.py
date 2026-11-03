import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will construct a decomposition of rank 53.
    # 49 terms from recursive Strassen for 4x4x4 (columns 0-3)
    # 4 terms for the 5th column (column 4)
    
    # Initialize U, V, W with zeros
    # U: (16, R), V: (20, R), W: (20, R)
    R = 53
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    # Define the 7 Strassen terms for 2x2x2
    # u, v, w are vectors of length 4
    # Indices: 00, 01, 10, 11
    # w index: k*n + i. For 2x2, n=2.
    # C_00 -> w[0], C_01 -> w[1], C_10 -> w[2], C_11 -> w[3]
    
    strassen_u = [
        [1, 0, 0, 1], # M1: A00 + A11
        [0, 0, 1, 1], # M2: A10 + A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00 + A01
        [-1, 0, 1, 0],# M6: A10 - A00
        [0, 1, 0, -1] # M7: A01 - A11
    ]
    strassen_v = [
        [1, 0, 0, 1], # M1: B00 + B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01 - B11
        [-1, 0, 1, 0],# M4: B10 - B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00 + B01
        [0, 0, 1, 1]  # M7: B10 + B11
    ]
    strassen_w = [
        [1, 0, 0, 1], # M1: C00 + C11
        [0, 0, 1, -1],# M2: C10 - C11
        [0, 1, 0, 1], # M3: C01 + C11
        [1, 0, 1, 0], # M4: C00 + C10
        [-1, 1, 0, 0],# M5: -C00 + C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    # Check M7: C11 = M1 - M2 + M3 + M6.
    # M7 contributes to C11?
    # Formula: C11 = M1 + M4 - M5 + M7.
    # So M7 contributes to C11.
    # My w7 was [1,0,0,0] which is C00.
    # Let's re-verify w7.
    # M7 = (A01-A11)(B10+B11).
    # Contributes to C11.
    # So w7 should be [0,0,0,1].
    # Let's check M4: C11 = ... + M4.
    # M4 = A11(B10-B00). Contributes to C10 and C11?
    # Formula: C11 = M1 + M4 - M5 + M7.
    # C21 = M2 + M4.
    # So M4 contributes to C11 and C21.
    # w4 should be [1, 0, 1, 0] (C00? No, C11 and C10).
    # Indices: 00, 01, 10, 11.
    # C11 is index 3. C10 is index 2.
    # So w4 = [0, 0, 1, 1].
    
    # Let's redo w vectors carefully.
    # C11 (idx 3): M1, M4, -M5, M7
    # C12 (idx 1): M3, M5
    # C21 (idx 2): M2, M4
    # C22 (idx 0? No, C22 is idx 3? No.
    # C is 2x2.
    # C[0,0] -> idx 0.
    # C[0,1] -> idx 1.
    # C[1,0] -> idx 2.
    # C[1,1] -> idx 3.
    
    # C00 (idx 0): M1, -M5, M7?
    # Formula: C00 = M1 + M4 - M5 + M7?
    # No, C11 in 1-based is C[1,1] in 0-based?
    # Strassen uses 1..2.
    # C11 (top-left) -> C[0,0].
    # C12 (top-right) -> C[0,1].
    # C21 (bottom-left) -> C[1,0].
    # C22 (bottom-right) -> C[1,1].
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # w1 (M1): C[0,0] + C[1,1] -> [1, 0, 0, 1]
    # w2 (M2): C[1,0] - C[1,1] -> [0, 0, 1, -1]
    # w3 (M3): C[0,1] + C[1,1] -> [0, 1, 0, 1]
    # w4 (M4): C[0,0] + C[1,0] -> [1, 0, 1, 0]
    # w5 (M5): -C[0,0] + C[0,1] -> [-1, 1, 0, 0]
    # w6 (M6): C[1,1] -> [0, 0, 0, 1]
    # w7 (M7): C[0,0] -> [1, 0, 0, 0]
    
    strassen_w = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Now embed into 4x4x4
    # 4x4 is 2x2 blocks of 2x2.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # For each of the 7 Strassen terms (m_idx),
    # we have a linear combination of A-blocks and B-blocks.
    # This linear combination is applied to the 4x4 A and B.
    # The result is a 2x2 matrix product, which is decomposed into 7 terms.
    # So for each m_idx, we have 7 sub-terms.
    # Total 49 terms.
    
    # We need to map the 2x2 indices to 4x4 indices.
    # Block (bi, bj) of A corresponds to rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
    
    # For a given Strassen term m_idx, the input A' is a linear combination of A-blocks.
    # A'_{i,j} = sum_{bi,bj} coeff_A[m_idx, bi, bj] * A[2*bi+i, 2*bj+j]
    # Similarly for B'.
    # The output C' is a 2x2 matrix.
    # C'_{i,j} is added to C[2*ci+i, 2*cj+j] with coefficient coeff_C[m_idx, ci, cj].
    
    # The 7 sub-terms of the 2x2x2 decomposition handle the internal structure.
    # Let's define the coefficients for the block combinations.
    
    # M1: A' = A00 + A11, B' = B00 + B11
    # M2: A' = A10 + A11, B' = B00
    # M3: A' = A00, B' = B01 - B11
    # M4: A' = A11, B' = B10 - B00
    # M5: A' = A00 + A01, B' = B11
    # M6: A' = A10 - A00, B' = B00 + B01
    # M7: A' = A01 - A11, B' = B10 + B11
    
    # Coefficients for A blocks (2x2 grid of blocks)
    # bi, bj in {0,1}
    coeff_A = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 1]], # M2
        [[1, 0], [0, 0]], # M3
        [[0, 0], [0, 1]], # M4
        [[1, 1], [0, 0]], # M5
        [[-1, 0], [1, 0]],# M6
        [[0, 1], [0, -1]] # M7
    ]
    coeff_B = [
        [[1, 0], [0, 1]], # M1
        [[1, 0], [0, 0]], # M2
        [[0, 1], [0, -1]],# M3
        [[-1, 0], [1, 0]],# M4
        [[0, 0], [0, 1]], # M5
        [[1, 1], [0, 0]], # M6
        [[0, 0], [1, 1]]  # M7
    ]
    coeff_C = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, -1]],# M2
        [[0, 1], [0, 1]], # M3
        [[1, 0], [1, 0]], # M4
        [[-1, 1], [0, 0]],# M5
        [[0, 0], [0, 1]], # M6
        [[1, 0], [0, 0]]  # M7
    ]
    
    # Now, for each m_idx in 0..6, and each sub_idx in 0..6:
    # The term is a rank-1 term in the 4x4x4 tensor.
    # U vector:
    # U_{i,j} = sum_{bi,bj} coeff_A[m_idx, bi, bj] * strassen_u[sub_idx][i', j']
    # where i' = i % 2, j' = j % 2?
    # No, the 2x2 decomposition is for the 2x2 blocks.
    # The internal indices of the 2x2 block are i', j'.
    # The global indices are i = 2*bi + i', j = 2*bj + j'.
    
    # So for a fixed m_idx and sub_idx:
    # U_global[i, j] = coeff_A[m_idx, i//2, j//2] * strassen_u[sub_idx][i%2, j%2]
    
    # Similarly for V.
    # V_global[j, k] = coeff_B[m_idx, j//2, k//2] * strassen_v[sub_idx][j%2, k%2]
    
    # And for W.
    # W_global[k, i] = coeff_C[m_idx, k//2, i//2] * strassen_w[sub_idx][k%2, i%2]
    
    term_idx = 0
    for m_idx in range(7):
        for sub_idx in range(7):
            # U
            for i in range(4):
                for j in range(4):
                    bi = i // 2
                    bj = j // 2
                    ii = i % 2
                    jj = j % 2
                    U[i * 4 + j, term_idx] = coeff_A[m_idx][bi][bj] * strassen_u[sub_idx][ii * 2 + jj]
            
            # V
            for j in range(4):
                for k in range(4):
                    bj = j // 2
                    bk = k // 2
                    jj = j % 2
                    kk = k % 2
                    V[j * 5 + k, term_idx] = coeff_B[m_idx][bj][bk] * strassen_v[sub_idx][jj * 2 + kk]
            
            # W
            # W index is k*n + i. n=4.
            # W is for 4x4 output (columns 0-3).
            for k in range(4):
                for i in range(4):
                    bk = k // 2
                    bi = i // 2
                    kk = k % 2
                    ii = i % 2
                    W[k * 4 + i, term_idx] = coeff_C[m_idx][bk][bi] * strassen_w[sub_idx][kk * 2 + ii]
            
            term_idx += 1
    
    # Now add 4 terms for k=4
    # C_{i,4} = sum_j A_{i,j} B_{j,4}
    # Terms for j=0,1,2,3
    for j in range(4):
        # U
        for i in range(4):
            U[i * 4 + j, term_idx] = 1
        # V
        V[j * 5 + 4, term_idx] = 1
        # W
        for i in range(4):
            W[4 * 4 + i, term_idx] = 1
        
        term_idx += 1
    
    return {"U": U, "V": V, "W": W}

