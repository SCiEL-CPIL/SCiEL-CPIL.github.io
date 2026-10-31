import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition using Strassen's algorithm recursively
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Base Strassen 7 decomposition for 2x2 matrices
    # Indices for 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # u: coeff for A, v: coeff for B, w: coeff for C
    strassen_2x2 = [
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (A10+A11)B00 -> C10
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3 = A00(B01-B11) -> C01
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        # M4 = A11(B10-B00) -> C00, C10
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # M5 = (A00+A01)B11 -> C00(-1), C01(1)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6 = (A10-A00)(B00+B01) -> C11
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7 = (A01-A11)(B10+B11) -> C00
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Block level Strassen 7 decomposition for 2x2 blocks of 2x2 matrices
    # Blocks: 11, 12, 21, 22
    # u_block: coeff for block A, v_block: coeff for block B, w_block: coeff for block C
    # w_block indicates which block of C receives the result
    strassen_4x4_blocks = [
        # M1 = (A11+A22)(B11+B22) -> C11, C22
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (A21+A22)B11 -> C21
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3 = A11(B12-B22) -> C12
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        # M4 = A22(B21-B11) -> C11, C21
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # M5 = (A11+A12)B22 -> C11(-1), C12(1)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6 = (A21-A11)(B11+B12) -> C22
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7 = (A12-A22)(B21+B22) -> C11
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    col = 0
    for u_b, v_b, w_b in strassen_4x4_blocks:
        for u_s, v_s, w_s in strassen_2x2:
            # u_b is length 4 (blocks 11,12,21,22)
            # u_s is length 4 (elements 00,01,10,11 within a block)
            # We need to map these to the full 16-dimensional vectors
            
            # Helper to expand block coefficients to full vector
            def expand(u_vec, block_coeffs):
                # block_coeffs: [c11, c12, c21, c22]
                # u_vec: [c00, c01, c10, c11]
                # Returns 16-element vector
                res = [0] * 16
                # Block 11 is indices 0..3
                # Block 12 is indices 4..7
                # Block 21 is indices 8..11
                # Block 22 is indices 12..15
                for b_idx in range(4):
                    coeff = block_coeffs[b_idx]
                    if coeff == 0:
                        continue
                    base = b_idx * 4
                    for e_idx in range(4):
                        res[base + e_idx] = coeff * u_vec[e_idx]
                return res
            
            U_full = expand(u_s, u_b)
            V_full = expand(v_s, v_b)
            
            # W expansion is slightly different because w_b tells us WHICH block of C
            # and w_s tells us WHICH element in that block.
            # But wait, the Strassen formulas for blocks give linear combinations of blocks.
            # The result of a block step is a sum of blocks.
            # For example, C11 gets contribution from M1, M4, M5, M7.
            # In the tensor decomposition, each rank-1 term contributes to a specific output location.
            # The output location is determined by (block_idx, elem_idx).
            # w_b is a vector of 4 coefficients, one for each block of C.
            # w_s is a vector of 4 coefficients, one for each element within a block.
            # So for a specific (b_out, e_out), the coefficient is w_b[b_out] * w_s[e_out].
            
            W_full = [0] * 16
            for b_out in range(4):
                coeff_b = w_b[b_out]
                if coeff_b == 0:
                    continue
                base = b_out * 4
                for e_out in range(4):
                    coeff_e = w_s[e_out]
                    W_full[base + e_out] = coeff_b * coeff_e
            
            # Fill U, V, W
            for i in range(16):
                U[i][col] = U_full[i]
                V[i][col] = V_full[i]
                W[i][col] = W_full[i]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
