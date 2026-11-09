"""Strassen-based tensor decomposition for 4x4 * 4x8 matrix multiplication."""

import time

DIMENSIONS = [4, 4, 8]

def construct():
    """
    Construct a rank-minimal (within Strassen family) exact tensor decomposition
    for multiplying a 4x4 matrix by a 4x8 matrix.
    
    Uses Strassen's algorithm for 2x2 blocks. The 4x4 matrices are divided into
    2x2 blocks of 2x2 submatrices. Each 2x2 block multiplication uses Strassen's
    7-term decomposition.
    
    For 4x4 * 4x8:
    - A is 4x4: divided into 4 blocks of 2x2
    - B is 4x8: divided into 8 blocks of 2x2 (4 rows x 8 cols = 2x2 blocks)
    - C is 4x8: divided into 8 blocks of 2x2
    
    Each of the 4x4 block multiplications (2x2 blocks) uses Strassen with rank 7.
    Total rank = 4 (block positions in A) * 4 (block positions in B) * 7 (Strassen terms) = 112?
    
    Wait, let me think more carefully.
    
    Actually, for 4x4 * 4x8:
    A has shape (4,4), B has shape (4,8), C has shape (4,8).
    
    Divide A into 2x2 blocks of 2x2: A = [[A11, A12], [A21, A22]]
    Divide B into 2x2 blocks of 2x2: B = [[B11, B12, B13, B14], [B21, B22, B23, B24]]
    Divide C into 2x2 blocks of 2x2: C = [[C11, C12, C13, C14], [C21, C22, C23, C24]]
    
    C_ij = sum_k A_ik * B_kj
    
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C13 = A11*B13 + A12*B23
    C14 = A11*B14 + A12*B24
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    C23 = A21*B13 + A22*B23
    C24 = A21*B14 + A22*B24
    
    Each C_ij is a 2x2 matrix computed from products of 2x2 matrices.
    
    For each pair (A_ik, B_kj), we need to compute their 2x2 product. Using Strassen,
    each such product has rank 7.
    
    Number of such products: 2*2 * 2*4 = 4 * 8 = 32 pairs? No wait.
    
    Actually, for each output block C_ij (i in {1,2}, j in {1,2,3,4}), we have:
    C_ij = A_i1 * B_1j + A_i2 * B_2j
    
    So for each (i,j), we have 2 products of 2x2 matrices.
    Total number of 2x2 matrix products: 2 * 2 * 2 * 4 = 32? No.
    
    i in {1,2}, j in {1,2,3,4}, and for each (i,j) we have 2 terms (k=1,2).
    So total 2x2 products = 2 * 4 * 2 = 16.
    
    Each 2x2 product uses Strassen rank 7, so total rank = 16 * 7 = 112.
    
    That's worse than 128? No, 112 < 128, so it's an improvement.
    
    Wait, let me reconsider. The schoolbook rank is n*m*p = 4*4*8 = 128.
    Strassen gives 112, which is better.
    
    Let me implement this properly.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # We'll build U, V, W as lists of lists.
    # U has shape (n*m, rank) = (16, rank)
    # V has shape (m*p, rank) = (32, rank)  
    # W has shape (p*n, rank) = (32, rank)
    
    # Strassen 2x2 decomposition terms.
    # For 2x2 matrices A and B, compute C = A*B using 7 products.
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the Strassen terms as (u_indices, v_indices, w_coefficients)
    # Each term is a rank-1 tensor: U_col * V_col * W_col
    # For a 2x2 block at position (block_i, block_j) in A, (block_k, block_l) in B,
    # and output block (block_i, block_m) in C:
    
    # Let's define the 7 Strassen products for a generic 2x2 * 2x2 multiplication.
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    
    # Term 1: M1 = (a00+a11)*(b00+b11)
    # Term 2: M2 = (a10+a11)*b00
    # Term 3: M3 = a00*(b01-b11)
    # Term 4: M4 = a11*(b10-b00)
    # Term 5: M5 = (a00+a01)*b11
    # Term 6: M6 = (a10-a00)*(b00+b01)
    # Term 7: M7 = (a01-a11)*(b10+b11)
    
    # For each term, we need:
    # - Which entries of A (with coefficients) go into U
    # - Which entries of B (with coefficients) go into V
    # - Which entries of C (with coefficients) go into W
    
    # Let's define the Strassen terms as a list of dicts.
    # Each term has:
    #   'a': list of (local_a_row, local_a_col, coeff)
    #   'b': list of (local_b_row, local_b_col, coeff)
    #   'c': list of (local_c_row, local_c_col, coeff)
    
    strassen_terms = [
        # M1 = (a00+a11)*(b00+b11)
        {
            'a': [(0,0,1), (1,1,1)],
            'b': [(0,0,1), (1,1,1)],
            'c': [(0,0,1), (1,1,1)]  # C00 += M1, C11 += M1
        },
        # M2 = (a10+a11)*b00
        {
            'a': [(1,0,1), (1,1,1)],
            'b': [(0,0,1)],
            'c': [(1,0,1), (1,1,-1)]  # C10 += M2, C11 -= M2
        },
        # M3 = a00*(b01-b11)
        {
            'a': [(0,0,1)],
            'b': [(0,1,1), (1,1,-1)],
            'c': [(0,1,1), (1,1,1)]  # C01 += M3, C11 += M3
        },
        # M4 = a11*(b10-b00)
        {
            'a': [(1,1,1)],
            'b': [(1,0,1), (0,0,-1)],
            'c': [(0,0,1), (1,0,1)]  # C00 += M4, C10 += M4
        },
        # M5 = (a00+a01)*b11
        {
            'a': [(0,0,1), (0,1,1)],
            'b': [(1,1,1)],
            'c': [(0,0,-1), (0,1,1)]  # C00 -= M5, C01 += M5
        },
        # M6 = (a10-a00)*(b00+b01)
        {
            'a': [(1,0,1), (0,0,-1)],
            'b': [(0,0,1), (0,1,1)],
            'c': [(1,1,1)]  # C11 += M6
        },
        # M7 = (a01-a11)*(b10+b11)
        {
            'a': [(0,1,1), (1,1,-1)],
            'b': [(1,0,1), (1,1,1)],
            'c': [(0,0,1)]  # C00 += M7
        },
    ]
    
    # Now, for the 4x4 * 4x8 problem:
    # A is 4x4, divided into 2x2 blocks: A[i][j] for i,j in {0,1}
    # B is 4x8, divided into 2x2 blocks: B[i][j] for i in {0,1}, j in {0,1,2,3}
    # C is 4x8, divided into 2x2 blocks: C[i][j] for i in {0,1}, j in {0,1,2,3}
    
    # C[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j]
    
    # For each (i, j, k) where i in {0,1}, j in {0,1,2,3}, k in {0,1}:
    # We compute A[i][k] * B[k][j] using Strassen.
    
    # Total number of such products: 2 * 4 * 2 = 16
    # Each uses 7 Strassen terms, so total rank = 16 * 7 = 112
    
    rank = 16 * 7
    U = [[0] * rank for _ in range(n * m)]  # (16, 112)
    V = [[0] * rank for _ in range(m * p)]  # (32, 112)
    W = [[0] * rank for _ in range(p * n)]  # (32, 112)
    
    col = 0
    for i in range(2):  # block row of A and C
        for j in range(4):  # block col of B and C
            for k in range(2):  # block row of B / block col of A
                for t in range(7):  # Strassen term
                    term = strassen_terms[t]
                    
                    # Map local 2x2 indices to global indices
                    # A[i][k] is at global rows 2*i..2*i+1, cols 2*k..2*k+1
                    # B[k][j] is at global rows 2*k..2*k+1, cols 2*j..2*j+1
                    # C[i][j] is at global rows 2*i..2*i+1, cols 2*j..2*j+1
                    
                    # Fill U
                    for (lr, lc, coeff) in term['a']:
                        global_row = 2 * i + lr
                        global_col = 2 * k + lc
                        U[global_row * m + global_col][col] = coeff
                    
                    # Fill V
                    for (lr, lc, coeff) in term['b']:
                        global_row = 2 * k + lr
                        global_col = 2 * j + lc
                        V[global_row * p + global_col][col] = coeff
                    
                    # Fill W
                    for (lr, lc, coeff) in term['c']:
                        global_row = 2 * j + lr
                        global_col = 2 * i + lc
                        W[global_row * n + global_col][col] = coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
