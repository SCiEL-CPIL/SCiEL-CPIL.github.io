import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank for 4x4x4 is 49, for 4x4x1 is 16. Total 65.
    rank = 65
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # 1. 4x4x4 Strassen part (k=0..3)
    # We need to map the 7x7 block structure to the global indices.
    # The 4x4 matrix A is divided into 2x2 blocks of 2x2 submatrices.
    # A_block[bi][bj] corresponds to rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
    
    # Strassen 2x2 decomposition coefficients
    # Each of the 7 products is a 2x2 matrix multiplication.
    # We need the rank-7 decomposition for 2x2 matrix multiplication.
    
    # 2x2 Strassen terms (u, v, w) for X*Y=Z
    # X, Y, Z are 2x2.
    # Indices for 2x2:
    # X: 00, 01, 10, 11
    # Y: 00, 01, 10, 11
    # Z: 00, 01, 10, 11
    
    # The 7 terms:
    # 1: (X00+X11)(Y00+Y11) -> Z00+=1, Z11+=1
    # 2: (X10+X11)Y00 -> Z10+=1
    # 3: X00(Y01-Y11) -> Z01+=1
    # 4: X11(Y10-Y00) -> Z00+=1, Z10+=1
    # 5: (X00+X01)Y11 -> Z00+=1, Z01+=1
    # 6: (X10-X00)(Y00+Y01) -> Z11+=1
    # 7: (X01-X11)(Y10+Y11) -> Z00+=1, Z11+=1
    
    # Let's define the 2x2 decomposition explicitly.
    # We will iterate over the 7 Strassen products for the 4x4 blocks.
    
    # Map global indices to block indices
    # i, j for A. Block bi = i//2, bj = j//2. Local li = i%2, lj = j%2.
    # j, k for B. Block bj = j//2, bk = k//2. Local lj = j%2, lk = k%2.
    # k, i for C. Block bk = k//2, bi = i//2. Local lk = k%2, li = i%2.
    
    # We will generate the 49 terms for the 4x4x4 part.
    # For each of the 7 Strassen products (s=0..6):
    #   For each of the 7 terms in the 2x2 decomposition (t=0..6):
    #     Add a term to U, V, W.
    
    # First, define the 2x2 rank-7 decomposition terms.
    # Each term is (u_coeffs, v_coeffs, w_coeffs) for the 2x2 matrices.
    # u_coeffs: dict of (li, lj) -> coeff
    # v_coeffs: dict of (lj, lk) -> coeff
    # w_coeffs: dict of (lk, li) -> coeff
    
    # 2x2 Strassen decomposition:
    # We need 7 terms.
    # Let's hardcode them.
    
    # Term 1: (X00+X11)(Y00+Y11)
    # u: {00:1, 11:1}
    # v: {00:1, 11:1}
    # w: {00:1, 11:1}  # Z00 += M1, Z11 += M1
    
    # Term 2: (X10+X11)Y00
    # u: {10:1, 11:1}
    # v: {00:1}
    # w: {10:1} # Z10 += M2
    
    # Term 3: X00(Y01-Y11)
    # u: {00:1}
    # v: {01:1, 11:-1}
    # w: {01:1} # Z01 += M3
    
    # Term 4: X11(Y10-Y00)
    # u: {11:1}
    # v: {10:1, 00:-1}
    # w: {00:1, 10:1} # Z00 += M4, Z10 += M4
    
    # Term 5: (X00+X01)Y11
    # u: {00:1, 01:1}
    # v: {11:1}
    # w: {00:1, 01:1} # Z00 += M5, Z01 += M5
    
    # Term 6: (X10-X00)(Y00+Y01)
    # u: {10:1, 00:-1}
    # v: {00:1, 01:1}
    # w: {11:1} # Z11 += M6
    
    # Term 7: (X01-X11)(Y10+Y11)
    # u: {01:1, 11:-1}
    # v: {10:1, 11:1}
    # w: {00:1, 11:1} # Z00 += M7, Z11 += M7
    
    # Wait, the Strassen formulas for Z are:
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # My w_coeffs above must match these.
    # T1: w {00:1, 11:1} -> Z00+=1, Z11+=1. Matches M1 in Z00, Z11.
    # T2: w {10:1} -> Z10+=1. Matches M2 in Z10.
    # T3: w {01:1} -> Z01+=1. Matches M3 in Z01.
    # T4: w {00:1, 10:1} -> Z00+=1, Z10+=1. Matches M4 in Z00, Z10.
    # T5: w {00:1, 01:1} -> Z00+=1, Z01+=1. Matches M5 in Z00, Z01.
    # T6: w {11:1} -> Z11+=1. Matches M6 in Z11.
    # T7: w {00:1, 11:1} -> Z00+=1, Z11+=1. Matches M7 in Z00, Z11.
    
    # But wait, Z00 = M1 + M4 - M5 + M7.
    # My T5 has w {00:1}. So Z00 += M5. But formula says -M5.
    # So T5 w should be {00:-1, 01:1}?
    # Let's check Strassen formula again.
    # M5 = (A11+A12)B22.
    # C11 = M1 + M4 - M5 + M7.
    # C12 = M3 + M5.
    # So for M5, coefficient in C11 is -1, in C12 is 1.
    # So T5 w: {00:-1, 01:1}.
    
    # Let's re-verify T6.
    # M6 = (A21-A11)(B11+B12).
    # C22 = M1 - M2 + M3 + M6.
    # So M6 coeff in C22 is 1.
    # T6 w: {11:1}. Correct.
    
    # Let's re-verify T7.
    # M7 = (A12-A22)(B21+B22).
    # C11 = ... + M7.
    # C22 = ... + M6 (no M7).
    # Wait, C22 = M1 - M2 + M3 + M6.
    # So M7 is not in C22?
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So M7 is in C11 only?
    # Let's check.
    # C11 = (A11+A22)(B11+B22) + A22(B21-B11) - (A11+A12)B22 + (A12-A22)(B21+B22)
    # = A11B11 + A11B22 + A22B11 + A22B22 + A22B21 - A22B11 - A11B22 - A12B22 + A12B21 + A12B22 - A22B21 - A22B22
    # = A11B11 + A12B21.
    # Correct.
    
    # So T7 w: {00:1}. (Only C11).
    # My previous T7 w was {00:1, 11:1}. That was wrong.
    
    # Let's update the w_coeffs:
    # T1: {00:1, 11:1}
    # T2: {10:1}
    # T3: {01:1}
    # T4: {00:1, 10:1}
    # T5: {00:-1, 01:1}
    # T6: {11:1}
    # T7: {00:1}
    
    # Now, we iterate over the 7 block products (s=0..6) for the 4x4 matrix.
    # Each block product s corresponds to a specific combination of A and B blocks.
    # We need to map the 2x2 local indices to the global 4x4 indices.
    
    # Block mapping for A (4x4):
    # Block 00: rows 0,1 cols 0,1
    # Block 01: rows 0,1 cols 2,3
    # Block 10: rows 2,3 cols 0,1
    # Block 11: rows 2,3 cols 2,3
    
    # Block mapping for B (4x4):
    # Block 00: rows 0,1 cols 0,1
    # Block 01: rows 0,1 cols 2,3
    # Block 10: rows 2,3 cols 0,1
    # Block 11: rows 2,3 cols 2,3
    
    # Block mapping for C (4x4):
    # Same as A.
    
    # The 7 Strassen products for 4x4 blocks:
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B01-B11)
    # S4 = A11(B10-B00)
    # S5 = (A00+A01)B11
    # S6 = (A10-A00)(B00+B01)
    # S7 = (A01-A11)(B10+B11)
    
    # We will define the block coefficients for each S.
    # For each S, we have a dict of A_blocks -> coeff, B_blocks -> coeff.
    # Then for each local term t (0..6) in the 2x2 decomposition, we add to U, V, W.
    
    # This is getting complex. Let's just write the code.
    
    # 2x2 decomposition terms (u, v, w)
    # u: {(li, lj): coeff}
    # v: {(lj, lk): coeff}
    # w: {(lk, li): coeff}
    
    terms_2x2 = [
        # T1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # T2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # T3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
        # T4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # T5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # T6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # T7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # 4x4 Block Strassen products
    # S1: (A00+A11)(B00+B11)
    # S2: (A10+A11)B00
    # S3: A00(B01-B11)
    # S4: A11(B10-B00)
    # S5: (A00+A01)B11
    # S6: (A10-A00)(B00+B01)
    # S7: (A01-A11)(B10+B11)
    
    # Block indices: 00, 01, 10, 11
    # Map block (bi, bj) to global row/col ranges.
    # A block (bi, bj): rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
    # B block (bj, bk): rows 2*bj..2*bj+1, cols 2*bk..2*bk+1.
    # C block (bk, bi): rows 2*bk..2*bk+1, cols 2*bi..2*bi+1.
    
    block_products = [
        # (A_blocks, B_blocks)
        # S1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # S2
        ({(1,0):1, (1,1):1}, {(0,0):1}),
        # S3
        ({(0,0):1}, {(0,1):1, (1,1):-1}),
        # S4
        ({(1,1):1}, {(1,0):1, (0,0):-1}),
        # S5
        ({(0,0):1, (0,1):1}, {(1,1):1}),
        # S6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        # S7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
    ]
    
    # Iterate over S (0..6) and t (0..6)
    for s_idx, (a_blocks, b_blocks) in enumerate(block_products):
        for t_idx, (u_local, v_local, w_local) in enumerate(terms_2x2):
            # For each non-zero entry in u_local, v_local, w_local, we add a term.
            # But wait, the term is a product of sums.
            # U term is sum of A entries.
            # V term is sum of B entries.
            # W term is sum of C entries.
            
            # We need to add one row to U, V, W for this (s, t) combination?
            # No, the rank is 49. Each (s, t) pair is one rank term.
            # But u_local has multiple entries.
            # This means U[row, col] gets contributions from multiple global indices.
            
            # So for this specific rank term (s_idx, t_idx):
            # U[:, col] will have non-zeros at indices corresponding to a_blocks and u_local.
            # V[:, col] will have non-zeros at indices corresponding to b_blocks and v_local.
            # W[:, col] will have non-zeros at indices corresponding to w_local and the block mapping.
            
            # Let's populate U, V, W for this column.
            
            # U
            for (bi, bj), a_coeff in a_blocks.items():
                for (li, lj), u_coeff in u_local.items():
                    # Global row i = 2*bi + li
                    # Global col j = 2*bj + lj
                    i = 2*bi + li
                    j = 2*bj + lj
                    u_idx = i*m + j
                    U[u_idx, col] += a_coeff * u_coeff
            
            # V
            for (bj, bk), b_coeff in b_blocks.items():
                for (lj, lk), v_coeff in v_local.items():
                    # Global row j = 2*bj + lj
                    # Global col k = 2*bk + lk
                    j = 2*bj + lj
                    k = 2*bk + lk
                    v_idx = j*p + k
                    V[v_idx, col] += b_coeff * v_coeff
            
            # W
            # w_local keys are (lk, li)
            # We need to map (bk, bi) block to global indices.
            # But w_local is local to the 2x2 product result.
            # The result of S is a 2x2 matrix of 2x2 blocks?
            # No, S is a product of two 2x2 matrices of scalars?
            # No, S is a product of two 4x4 matrices?
            # No, S is one of the 7 Strassen products for the 4x4 matrix.
            # Each S is a 4x4 matrix?
            # No, S is a sum of products of 2x2 blocks.
            # The result of the 2x2 Strassen decomposition is a 2x2 matrix of 2x2 blocks.
            # So w_local (lk, li) refers to the local indices within the 2x2 block result.
            # We need to know which 4x4 block of C this corresponds to.
            # The block structure of C is determined by the Strassen formula.
            # C_block(bk, bi) = sum of S contributions.
            
            # Wait, the 4x4 Strassen formula gives:
            # C00 = S1 + S4 - S5 + S7
            # C01 = S3 + S5
            # C10 = S2 + S4
            # C11 = S1 - S2 + S3 + S6
            
            # So for a given S_idx, we know which C_blocks it contributes to.
            # Let's define c_blocks for each S_idx.
            
            # S1 (idx 0): C00, C11
            # S2 (idx 1): C10
            # S3 (idx 2): C01
            # S4 (idx 3): C00, C10
            # S5 (idx 4): C00, C01
            # S6 (idx 5): C11
            # S7 (idx 6): C00
            
            c_block_map = [
                {(0,0):1, (1,1):1}, # S1
                {(1,0):1}, # S2
                {(0,1):1}, # S3
                {(0,0):1, (1,0):1}, # S4
                {(0,0):-1, (0,1):1}, # S5
                {(1,1):1}, # S6
                {(0,0):1}, # S7
            ]
            
            # Now populate W
            for (bk, bi), c_coeff in c_block_map[s_idx].items():
                for (lk, li), w_coeff in w_local.items():
                    # Global row k = 2*bk + lk
                    # Global col i = 2*bi + li
                    k = 2*bk + lk
                    i = 2*bi + li
                    w_idx = k*n + i
                    W[w_idx, col] += c_coeff * w_coeff
            
            col += 1
    
    # 2. 4x4x1 Schoolbook part (k=4)
    # C[i, 4] = sum_{j=0}^3 A[i, j] B[j, 4]
    # This adds 16 terms.
    for i in range(n):
        for j in range(m):
            u_idx = i*m + j
            v_idx = j*p + 4
            w_idx = 4*n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
