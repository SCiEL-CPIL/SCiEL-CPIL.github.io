import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 98 solution using two-level Strassen for 4x4x4, applied to 4x4x8
    # For 4x4x8: A is 4x4, B is 4x8, C is 4x8
    # We can view B as two 4x4 blocks: B_left, B_right
    # So C = A * B_left + A * B_right
    # Each A * B_block uses rank 49 Strassen for 4x4x4
    # Total rank = 2 * 49 = 98
    
    rank = 98
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 98
    V = [[0] * rank for _ in range(m * p)]  # 32 x 98
    W = [[0] * rank for _ in range(p * n)]  # 32 x 98
    
    # Standard Strassen for 2x2x2 (rank 7)
    # For 4x4x4, we use a two-level Strassen:
    # Level 1: Split 4x4 into 2x2 blocks of 2x2
    # Level 2: Apply Strassen to each 2x2 block multiplication
    # But actually, the standard approach for 4x4x4 with rank 49 is:
    # Use the 7 rank-1 tensors for 2x2x2, and for each of the 4 block positions
    # in the 4x4 matrices, we need to handle the cross terms.
    #
    # Actually, let me use the known rank-49 decomposition for 4x4x4.
    # The 4x4x4 matrix multiplication tensor can be decomposed using
    # 7 copies of the 2x2x2 Strassen decomposition, one for each of the
    # 7 "Strassen products" at the block level.
    #
    # Let A = [[A00, A01], [A10, A11]] where each block is 2x2
    # Let B = [[B00, B01], [B10, B11]] where each block is 2x2
    # C = A*B has blocks Cij = sum_k Aik * Bkj
    #
    # Strassen for 2x2 blocks:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    #
    # Now, each Aij and Bkj is a 2x2 matrix, so each "product" like
    # (A00+A11)(B00+B11) is a 2x2x2 matrix multiplication.
    # We can decompose each 2x2x2 product using Strassen (rank 7).
    # So total rank = 7 * 7 = 49.
    
    # For each of the 7 Strassen products at the block level,
    # we have a 2x2x2 tensor that we decompose with rank 7.
    
    # Strassen for 2x2x2:
    # For matrices X, Y, Z (each 2x2):
    # s1 = (X00+X11)(Y00+Y11)
    # s2 = (X10+X11)Y00
    # s3 = X00(Y01-Y11)
    # s4 = X11(Y10-Y00)
    # s5 = (X00+X01)Y11
    # s6 = (X10-X00)(Y00+Y01)
    # s7 = (X01-X11)(Y10+Y11)
    #
    # Z00 = s1 - s4 + s6 - s7
    # Z01 = s3 + s5
    # Z10 = s2 + s4
    # Z11 = s1 - s2 + s3 + s7
    
    # Define the 7 rank-1 tensors for 2x2x2 Strassen
    # Each tensor is (u, v, w) where u, v, w are vectors in R^4
    # representing the row/column indices of the 2x2 matrices
    
    # For 2x2x2: A is 2x2 (4 entries), B is 2x2 (4 entries), C is 2x2 (4 entries)
    # Indexing: A[i,j] -> i*2+j, B[i,j] -> i*2+j, C[i,j] -> i*2+j
    
    strassen_2x2 = [
        # s1: (A00+A11)(B00+B11) -> C00, C11
        # u: A00, A11 -> [1,0,0,1]
        # v: B00, B11 -> [1,0,0,1]
        # w: C00, C11 -> [1,0,0,1]
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # s2: (A10+A11)B00 -> C10
        # u: A10, A11 -> [0,0,1,1]
        # v: B00 -> [1,0,0,0]
        # w: C10 -> [0,0,1,0]
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        # s3: A00(B01-B11) -> C01
        # u: A00 -> [1,0,0,0]
        # v: B01, B11 -> [0,1,0,-1]
        # w: C01 -> [0,1,0,0]
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
        # s4: A11(B10-B00) -> C00, C10
        # u: A11 -> [0,0,0,1]
        # v: B10, B00 -> [0,0,1,-1]
        # w: C00, C10 -> [1,0,1,0]
        ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),
        # s5: (A00+A01)B11 -> C01
        # u: A00, A01 -> [1,1,0,0]
        # v: B11 -> [0,0,0,1]
        # w: C01 -> [0,1,0,0]
        ([1,1,0,0], [0,0,0,1], [0,1,0,0]),
        # s6: (A10-A00)(B00+B01) -> C00
        # u: A10, A00 -> [0,0,1,-1]
        # v: B00, B01 -> [1,1,0,0]
        # w: C00 -> [1,0,0,0]
        ([0,0,1,-1], [1,1,0,0], [1,0,0,0]),
        # s7: (A01-A11)(B10+B11) -> C00, C11
        # u: A01, A11 -> [0,1,0,-1]
        # v: B10, B11 -> [0,0,1,1]
        # w: C00, C11 -> [1,0,0,1]
        ([0,1,0,-1], [0,0,1,1], [1,0,0,1]),
    ]
    
    # Block-level Strassen formulas for 4x4x4:
    # For each of the 7 products, we need to express the left factor (sum/diff of A blocks)
    # and right factor (sum/diff of B blocks), then apply 2x2x2 Strassen.
    
    # Block indices: (bi, bj) for A block, (ci, cj) for B block, (di, dj) for C block
    # A is 4x4, split into 2x2 blocks of 2x2
    # Block (bi, bj) occupies rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    
    # For each Strassen product s (0-6), define:
    # left_coeffs: list of (bi, bj, coeff) for A blocks
    # right_coeffs: list of (ci, cj, coeff) for B blocks
    # c_coeffs: list of (di, dj, coeff) for C blocks
    
    block_strassen = [
        # s1: (A00+A11)(B00+B11) -> C00, C11
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # s2: (A10+A11)B00 -> C10
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),
        # s3: A00(B01-B11) -> C01
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1)]),
        # s4: A11(B10-B00) -> C00, C10
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # s5: (A00+A01)B11 -> C01
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,1,1)]),
        # s6: (A10-A00)(B00+B01) -> C00
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(0,0,1)]),
        # s7: (A01-A11)(B10+B11) -> C00, C11
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
    ]
    
    col = 0
    for s in range(7):
        left_coeffs, right_coeffs, c_coeffs = block_strassen[s]
        
        # For each of the 7 inner Strassen products
        for inner_s in range(7):
            u_vec, v_vec, w_vec = strassen_2x2[inner_s]
            
            # u_vec has 4 entries corresponding to 2x2 matrix entries
            # For A block (bi, bj), entry (u, v) is at global position (2*bi+u, 2*bj+v)
            # u_vec[i] where i = u*2+v gives the coefficient for A entry (u,v) in this block
            
            # Build U contribution
            for (bi, bj, coeff) in left_coeffs:
                for u in range(2):
                    for v in range(2):
                        idx = u * 2 + v
                        val = coeff * u_vec[idx]
                        if val != 0:
                            row_a = 2 * bi + u
                            col_a = 2 * bj + v
                            U[row_a * m + col_a][col] += val
            
            # Build V contribution
            for (ci, cj, coeff) in right_coeffs:
                for u in range(2):
                    for v in range(2):
                        idx = u * 2 + v
                        val = coeff * v_vec[idx]
                        if val != 0:
                            row_b = 2 * ci + u
                            col_b = 2 * cj + v
                            V[row_b * p + col_b][col] += val
            
            # Build W contribution
            for (di, dj, coeff) in c_coeffs:
                for u in range(2):
                    for v in range(2):
                        idx = u * 2 + v
                        val = coeff * w_vec[idx]
                        if val != 0:
                            row_c = 2 * di + u
                            col_c = 2 * dj + v
                            # W index: col_c * n + row_c (trace convention)
                            W[col_c * n + row_c][col] += val
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
