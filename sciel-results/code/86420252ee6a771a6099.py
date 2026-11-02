"""
Tensor decomposition for 2x4 by 4x4 matrix multiplication.

Uses a block-structured Strassen-like approach: split the 4x4 B matrix into four
2x2 blocks, and the 2x4 A matrix into two 2x2 blocks (left and right). Each of
the four sub-blocks of B multiplies with the appropriate block of A to produce
a 2x2 sub-block of C. Each 2x2*2x2->2x2 multiplication is done with Strassen's
7-product algorithm, giving 4*7 = 28 rank-1 terms total.
"""
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-28 exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x4 matrix using Strassen's algorithm on 2x2 blocks.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    A is 2x4, split into A_L (2x2, cols 0,1) and A_R (2x2, cols 2,3).
    B is 4x4, split into four 2x2 blocks:
        B_TL (rows 0,1; cols 0,1)
        B_TR (rows 0,1; cols 2,3)
        B_BL (rows 2,3; cols 0,1)
        B_BR (rows 2,3; cols 2,3)
    
    C is 2x4, split into C_L (2x2, cols 0,1) and C_R (2x2, cols 2,3).
    
    C_L = A_L @ B_TL + A_R @ B_BL
    C_R = A_L @ B_TR + A_R @ B_BR
    
    Each 2x2*2x2 product uses Strassen's 7 multiplications.
    Total rank = 4 * 7 = 28.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    
    # U has shape (n*m, rank) = (8, 28)
    # V has shape (m*p, rank) = (16, 28)
    # W has shape (p*n, rank) = (8, 28)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products for 2x2 * 2x2:
    # Given X (2x2) and Y (2x2), compute Z = X @ Y using:
    # M1 = (X00+X11) * (Y00+Y11)
    # M2 = (X10+X11) * Y00
    # M3 = X00 * (Y01-Y11)
    # M4 = X11 * (Y10-Y00)
    # M5 = (X00+X01) * Y11
    # M6 = (X10-X00) * (Y00+Y01)
    # M7 = (X01-X11) * (Y10+Y11)
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # For each Strassen term, we need to express the linear combinations
    # in terms of the global A and B indices, and the output in terms of C.
    
    # We'll define the 7 Strassen terms as (x_coeffs, y_coeffs, z_coeffs)
    # where x_coeffs is a dict from local (i,j) to coefficient,
    # y_coeffs is a dict from local (j,k) to coefficient,
    # z_coeffs is a dict from local (i,k) to coefficient.
    
    # Local 2x2 indices: i in {0,1}, j in {0,1}, k in {0,1}
    
    strassen_terms = [
        # M1: (X00+X11) * (Y00+Y11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2: (X10+X11) * Y00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3: X00 * (Y01-Y11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4: X11 * (Y10-Y00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(1,0): 1, (1,1): -1}),
        # M5: (X00+X01) * Y11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6: (X10-X00) * (Y00+Y01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7: (X01-X11) * (Y10+Y11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Define the four blocks:
    # Block 0: A_L (cols 0,1) * B_TL (rows 0,1; cols 0,1) -> C_L (cols 0,1)
    # Block 1: A_R (cols 2,3) * B_BL (rows 2,3; cols 0,1) -> C_L (cols 0,1)
    # Block 2: A_L (cols 0,1) * B_TR (rows 0,1; cols 2,3) -> C_R (cols 2,3)
    # Block 3: A_R (cols 2,3) * B_BR (rows 2,3; cols 2,3) -> C_R (cols 2,3)
    
    # For each block, we define:
    # a_col_offset: offset for A's column index (0 or 2)
    # b_row_offset: offset for B's row index (0 or 2)
    # b_col_offset: offset for B's column index (0 or 2)
    # c_col_offset: offset for C's column index (0 or 2)
    
    blocks = [
        # (a_col_offset, b_row_offset, b_col_offset, c_col_offset)
        (0, 0, 0, 0),   # Block 0: A_L * B_TL -> C_L
        (2, 2, 0, 0),   # Block 1: A_R * B_BL -> C_L
        (0, 0, 2, 2),   # Block 2: A_L * B_TR -> C_R
        (2, 2, 2, 2),   # Block 3: A_R * B_BR -> C_R
    ]
    
    col = 0
    for a_col_off, b_row_off, b_col_off, c_col_off in blocks:
        for x_coeffs, y_coeffs, z_coeffs in strassen_terms:
            # Process x_coeffs: A[i, a_col_off + j]
            for (li, lj), coeff_x in x_coeffs.items():
                # Global A index: i * m + (a_col_off + lj)
                global_a_idx = li * m + (a_col_off + lj)
                U[global_a_idx][col] += coeff_x
            
            # Process y_coeffs: B[b_row_off + lj, b_col_off + lk]
            for (lj, lk), coeff_y in y_coeffs.items():
                # Global B index: (b_row_off + lj) * p + (b_col_off + lk)
                global_b_idx = (b_row_off + lj) * p + (b_col_off + lk)
                V[global_b_idx][col] += coeff_y
            
            # Process z_coeffs: C[li, c_col_off + lk]
            # W index: k * n + i (trace convention), where k = c_col_off + lk, i = li
            for (li, lk), coeff_z in z_coeffs.items():
                global_c_i = li
                global_c_k = c_col_off + lk
                # W index: k * n + i
                global_w_idx = global_c_k * n + global_c_i
                W[global_w_idx][col] += coeff_z
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
