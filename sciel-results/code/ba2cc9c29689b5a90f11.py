import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-189 exact tensor decomposition for 6x6 matrix multiplication.
    Uses Block-Strassen: 2x2 blocks of 3x3, with schoolbook multiplication for the 3x3 blocks.
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strassen rank-7 for 2x2 blocks
    # Each of the 7 Strassen products uses schoolbook 3x3 multiplication (rank 27)
    # Total rank = 7 * 27 = 189
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189
    
    # Strassen formulas for 2x2 block multiplication:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = A*B where C_ij = sum_k A_ik * B_kj
    #
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each Strassen term, we need to expand the 3x3 schoolbook multiplication
    # Each 3x3 block product requires 27 scalar multiplications (rank 27)
    
    # Strassen terms: (X_blocks, Y_blocks, C_blocks)
    # X_blocks: list of (block_row, block_col, coeff) for A
    # Y_blocks: list of (block_row, block_col, coeff) for B
    # C_blocks: list of (block_row, block_col, coeff) for C
    
    strassen_terms = [
        {
            'X': [(0, 0, 1), (1, 1, 1)],
            'Y': [(0, 0, 1), (1, 1, 1)],
            'C': [(0, 0, 1), (1, 1, 1)]
        },
        {
            'X': [(1, 0, 1), (1, 1, 1)],
            'Y': [(0, 0, 1)],
            'C': [(1, 0, 1), (1, 1, -1)]
        },
        {
            'X': [(0, 0, 1)],
            'Y': [(0, 1, 1), (1, 1, -1)],
            'C': [(0, 1, 1), (1, 1, 1)]
        },
        {
            'X': [(1, 1, 1)],
            'Y': [(1, 0, 1), (0, 0, -1)],
            'C': [(0, 0, 1), (1, 0, 1)]
        },
        {
            'X': [(0, 0, 1), (0, 1, 1)],
            'Y': [(1, 1, 1)],
            'C': [(0, 0, -1), (0, 1, 1)]
        },
        {
            'X': [(1, 0, 1), (0, 0, -1)],
            'Y': [(0, 0, 1), (0, 1, 1)],
            'C': [(1, 1, 1)]
        },
        {
            'X': [(0, 1, 1), (1, 1, -1)],
            'Y': [(1, 0, 1), (1, 1, 1)],
            'C': [(0, 0, 1)]
        }
    ]
    
    col = 0
    for term in strassen_terms:
        X_coeffs = term['X']
        Y_coeffs = term['Y']
        C_coeffs = term['C']
        
        # For each of the 27 schoolbook multiplications within the 3x3 blocks
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    # This scalar multiplication corresponds to:
                    # (sum of A blocks)_{a,b} * (sum of B blocks)_{b,c}
                    # contributing to (sum of C blocks)_{a,c}
                    
                    # Fill U: coefficients for A matrix entries
                    for (br, bs, coeff) in X_coeffs:
                        # A entry at row = 3*br + a, col = 3*bs + b
                        idx_a_row = 3 * br + a
                        idx_a_col = 3 * bs + b
                        u_idx = idx_a_row * 6 + idx_a_col
                        U[u_idx][col] += coeff
                    
                    # Fill V: coefficients for B matrix entries
                    for (br, bs, coeff) in Y_coeffs:
                        # B entry at row = 3*br + b, col = 3*bs + c
                        idx_b_row = 3 * br + b
                        idx_b_col = 3 * bs + c
                        v_idx = idx_b_row * 6 + idx_b_col
                        V[v_idx][col] += coeff
                    
                    # Fill W: coefficients for C matrix entries
                    # W index convention: k*n + i (col*6 + row)
                    for (br, bs, coeff) in C_coeffs:
                        # C entry at row = 3*br + a, col = 3*bs + c
                        idx_c_row = 3 * br + a
                        idx_c_col = 3 * bs + c
                        w_idx = idx_c_col * 6 + idx_c_row
                        W[w_idx][col] += coeff
                    
                    col += 1
    
    # Deadline guard: verify we haven't exceeded time
    # (This construction is fast, ~milliseconds, so this is just a safety check)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
