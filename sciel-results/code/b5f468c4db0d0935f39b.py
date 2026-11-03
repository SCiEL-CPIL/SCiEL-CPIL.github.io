import numpy as np

def construct():
    n, m, p = 6, 6, 6
    rank = 189
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Indices for 3x3 blocks
    # Block (bi, bj) of a 6x6 matrix (partitioned into 2x2 blocks of 3x3)
    # Row range: [bi*3, bi*3+2]
    # Col range: [bj*3, bj*3+2]
    
    # Strassen components
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B00)
    # M4 = A11(B11 - B01)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We iterate over the 7 Strassen products
    # For each product k, we have coefficients for A-blocks and B-blocks
    # A_block_coeffs: list of (block_row, block_col, coeff)
    # B_block_coeffs: list of (block_row, block_col, coeff)
    # C_block_coeffs: list of (block_row, block_col, coeff)
    
    strassen_products = [
        # M1
        (
            [(0,0,1), (1,1,1)], # A blocks
            [(0,0,1), (1,1,1)], # B blocks
            [(0,0,1)]           # C block contribution (added to C00)
        ),
        # M2
        (
            [(1,0,1), (1,1,1)],
            [(0,0,1)],
            [(1,0,1)]           # Added to C10
        ),
        # M3
        (
            [(0,0,1)],
            [(1,0,1), (0,0,-1)],
            [(0,1,1)]           # Added to C01
        ),
        # M4
        (
            [(1,1,1)],
            [(1,1,1), (0,1,-1)],
            [(0,0,1), (1,0,1)]  # Added to C00 and C10
        ),
        # M5
        (
            [(0,0,1), (0,1,1)],
            [(1,1,1)],
            [(0,0,-1), (0,1,1)] # Added to C00 (-1) and C01 (1)
        ),
        # M6
        (
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (0,1,1)],
            [(1,1,1)]           # Added to C11
        ),
        # M7
        (
            [(0,1,1), (1,1,-1)],
            [(1,0,1), (1,1,1)],
            [(0,0,1)]           # Added to C00
        )
    ]
    
    col_offset = 0
    for a_coeffs, b_coeffs, c_coeffs in strassen_products:
        # For this Strassen product, we decompose the 3x3 multiplication into 27 schoolbook terms
        for r in range(3):
            for c in range(3):
                for d in range(3):
                    # This term contributes to X[r, c] * Y[c, d] = Z[r, d]
                    # We need to add this term to U, V, W with appropriate signs based on block coefficients
                    
                    # For each combination of A block and B block in this Strassen product,
                    # we add a rank-1 term to the global decomposition.
                    # However, the Strassen product itself is a linear combination of block products.
                    # Actually, Strassen reduces the number of multiplications.
                    # Each "multiplication" M_k is a product of two linear combinations of blocks.
                    # Let A_k = sum a_i A_{b_i}, B_k = sum b_j B_{c_j}.
                    # M_k = A_k B_k.
                    # This M_k contributes to C.
                    
                    # The rank of M_k (3x3 product) is 27.
                    # So we add 27 columns for M_k.
                    # For each of these 27 columns (indexed by r,c,d), the values in U, V, W are determined by:
                    # U: coefficient for A_{b_i}[r,c]
                    # V: coefficient for B_{c_j}[c,d]
                    # W: coefficient for C_{out}[r,d]
                    
                    # Wait, the linearity is in the matrix entries.
                    # U[row_A, col_k] = sum_{(bi, bj, coef)} coef * delta_{row_A, (bi*3+r, bj*3+c)}
                    # V[row_B, col_k] = sum_{(bi, bj, coef)} coef * delta_{row_B, (bi*3+c, bj*3+d)}
                    # W[row_C, col_k] = sum_{(bi, bj, coef)} coef * delta_{row_C, (bj*3+d, bi*3+r)} ??
                    
                    # Let's trace indices carefully.
                    # A is 6x6. Block (bi, bj) is rows [3bi, 3bi+2], cols [3bj, 3bj+2].
                    # Entry (r, c) in block (bi, bj) is at global row 3bi+r, col 3bj+c.
                    # U index: row_A * 6 + col_A.
                    # So U index for A_{bi,bj}[r,c] is (3bi+r)*6 + (3bj+c).
                    
                    # B is 6x6. Block (bi, bj) is rows [3bi, 3bi+2], cols [3bj, 3bj+2].
                    # Entry (c, d) in block (bi, bj) is at global row 3bi+c, col 3bj+d.
                    # V index: row_B * 6 + col_B.
                    # So V index for B_{bi,bj}[c,d] is (3bi+c)*6 + (3bj+d).
                    
                    # C is 6x6. Block (bi, bj) is rows [3bi, 3bi+2], cols [3bj, 3bj+2].
                    # Entry (r, d) in block (bi, bj) is at global row 3bi+r, col 3bj+d.
                    # W index: col_C * 6 + row_C (Trace convention W[k*n+i] where k=col, i=row).
                    # So W index for C_{bi,bj}[r,d] is (3bj+d)*6 + (3bi+r).
                    
                    # For a specific Strassen product k, with A-blocks coeffs `a_coeffs` and B-blocks coeffs `b_coeffs`:
                    # The product is M_k = (sum_a a_coeff_a A_a) * (sum_b b_coeff_b B_b).
                    # This is NOT simply sum of products. It is (linear combo) * (linear combo).
                    # The rank of this specific 3x3 multiplication instance is 27.
                    # The 27 basis vectors for the 3x3 tensor are fixed.
                    # For the r,c,d term in the 3x3 local coordinates:
                    # The local A matrix has entry 1 at (r,c) and 0 elsewhere? No.
                    # The tensor decomposition $T_{3x3} = \sum_{r,c,d} u_{r,c} \otimes v_{c,d} \otimes w_{r,d}$
                    # $u_{r,c}$ is the 3x3 matrix with 1 at (r,c).
                    # $v_{c,d}$ is the 3x3 matrix with 1 at (c,d).
                    # $w_{r,d}$ is the 3x3 matrix with 1 at (r,d).
                    
                    # So for the global decomposition:
                    # For each Strassen product k:
                    #   For each r,c,d in 0..2:
                    #     U_col = 0
                    #     For each (bi, bj, ca) in a_coeffs:
                    #        row_a = 3*bi + r
                    #        col_a = 3*bj + c
                    #        U[row_a * 6 + col_a, col_offset] += ca
                    #        
                    #     V_col = 0
                    #     For each (bi, bj, cb) in b_coeffs:
                    #        row_b = 3*bi + c
                    #        col_b = 3*bj + d
                    #        V[row_b * 6 + col_b, col_offset] += cb
                    #        
                    #     W_col = 0
                    #     For each (bi, bj, cc) in c_coeffs:
                    #        # This c_coeffs represents how M_k contributes to C_blocks.
                    #        # C_block (bi, bj) gets contribution cc * M_k[r, d]
                    #        # So the local entry (r, d) of M_k maps to global C_block (bi, bj) entry (r, d)
                    #        # Global row in C = 3*bi + r
                    #        # Global col in C = 3*bj + d
                    #        # W index = col_C * 6 + row_C = (3*bj + d) * 6 + (3*bi + r)
                    #        w_idx = (3*bj + d) * 6 + (3*bi + r)
                    #        W[w_idx, col_offset] += cc
                    #        
                    #     col_offset += 1
                    
                    for (bi_a, bj_a, ca) in a_coeffs:
                        row_a = 3 * bi_a + r
                        col_a = 3 * bj_a + c
                        U[row_a * 6 + col_a, col_offset] += ca

                    for (bi_b, bj_b, cb) in b_coeffs:
                        row_b = 3 * bi_b + c
                        col_b = 3 * bj_b + d
                        V[row_b * 6 + col_b, col_offset] += cb

                    for (bi_c, bj_c, cc) in c_coeffs:
                        # M_k is added to C block (bi_c, bj_c) with coefficient cc
                        # The local value is at [r, d]
                        # Global C row = 3 * bi_c + r
                        # Global C col = 3 * bj_c + d
                        # W index = col * 6 + row
                        w_idx = (3 * bj_c + d) * 6 + (3 * bi_c + r)
                        W[w_idx, col_offset] += cc

                    col_offset += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
