import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Total rank is 49 (for 4x4x4) + 8 (for 4x4x2) = 57
    rank = 57
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, col, coeff=1):
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff

    # Part 1: 4x4x4 using Strassen (rank 49)
    # We decompose the 4x4x4 multiplication into 7 block-level Strassen factors,
    # each of which is a 2x2x2 multiplication decomposed by Strassen (rank 7).
    
    # Indices for A (4x4): i*m + j where i,j in 0..1 (block), a,b in 0..1 (element)
    # Block (i,j) corresponds to rows 2i..2i+1, cols 2j..2j+1 of A.
    # Element (a,b) in block (i,j) is at index (2i+a)*4 + (2j+b).
    
    # Indices for B (4x4): j*p + k where j,k in 0..1 (block), c,d in 0..1 (element)
    # Block (j,k) corresponds to rows 2j..2j+1, cols 2k..2k+1 of B.
    # Element (c,d) in block (j,k) is at index (2j+c)*4 + (2k+d).
    # Note: p=6 here, but for the 4x4x4 part, we only use the first 4 columns of B.
    # So B indices are (row, col) where row in 0..3, col in 0..3.
    # V index is row * p + col. Since p=6, V index is row * 6 + col.
    
    # Indices for C (4x4): k*n + i where k,i in 0..1 (block), e,f in 0..1 (element)
    # Block (k,i) corresponds to rows 2k..2k+1, cols 2i..2i+1 of C.
    # Element (e,f) in block (k,i) is at index (2k+e)*4 + (2i+f).
    # W index is row * n + col. Since n=4, W index is row * 4 + col.
    
    # Strassen 2x2x2 factors (rank 7)
    # Returns list of (u_coeff, v_coeff, w_coeff) for the 7 terms?
    # Actually, we need to expand the linear combinations.
    
    # Let's define the 7 Strassen products for 2x2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B22
    # M3 = A11(B12 - B22)
    # M4 = (A22 - A11)(B11 + B21)
    # M5 = (A11 + A12)B21
    # M6 = (A12 - A22)(B11 + B12)
    # M7 = (A21 - A11)(B12 + B22)
    
    # And the reconstruction:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M2 + M3 + M4
    
    # We are doing this for 7 block-level factors.
    # Block-level factors:
    # S1 = (A11 + A22)(B11 + B22)
    # S2 = (A21 + A22)B22
    # S3 = A11(B12 - B22)
    # S4 = (A22 - A11)(B11 + B21)
    # S5 = (A11 + A12)B21
    # S6 = (A12 - A22)(B11 + B12)
    # S7 = (A21 - A11)(B12 + B22)
    
    # Each Sk is a product of two 2x2 matrices.
    # We decompose each Sk using the 2x2x2 Strassen decomposition.
    
    # Let's create a function to add a 2x2x2 Strassen decomposition to the global U, V, W
    # given the linear combinations for A and B blocks.
    
    def add_strassen_2x2(col_offset, a_combos, b_combos, c_combos):
        # a_combos: list of 4 tuples (coeff, block_i, block_j, elem_a, elem_b) for A
        # b_combos: list of 4 tuples (coeff, block_i, block_j, elem_a, elem_b) for B
        # c_combos: list of 4 tuples (coeff, block_i, block_j, elem_a, elem_b) for C
        # This is getting complex. Let's just iterate through the 7 terms of the inner Strassen.
        
        pass

    # Simpler approach:
    # Generate the 49 terms explicitly.
    
    # Inner Strassen terms (for a single 2x2x2 multiplication):
    # Let A_block be 2x2, B_block be 2x2.
    # A_block indices: (0,0), (0,1), (1,0), (1,1)
    # B_block indices: (0,0), (0,1), (1,0), (1,1)
    
    # The 7 inner products are:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B11
    # P3 = A00(B01 - B11)
    # P4 = (A11 - A00)(B00 + B10)
    # P5 = (A00 + A01)B10
    # P6 = (A01 - A11)(B00 + B01)
    # P7 = (A10 - A00)(B01 + B11)
    
    # Reconstruction:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P6
    # C11 = P1 - P2 + P3 + P4
    
    # We need to map these to the global indices.
    
    # Let's define the 7 outer Strassen factors (S1..S7) in terms of A and B blocks.
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    # S1 = (A00 + A11)(B00 + B11)
    # S2 = (A10 + A11)B11
    # S3 = A00(B01 - B11)
    # S4 = (A11 - A00)(B00 + B10)
    # S5 = (A00 + A01)B10
    # S6 = (A01 - A11)(B00 + B01)
    # S7 = (A10 - A00)(B01 + B11)
    
    # Each Sk is a product of two 2x2 matrices.
    # Let's define the linear combinations for the A and B parts of each Sk.
    
    # A_part_S1 = A00 + A11
    # B_part_S1 = B00 + B11
    
    # A_part_S2 = A10 + A11
    # B_part_S2 = B11
    
    # A_part_S3 = A00
    # B_part_S3 = B01 - B11
    
    # A_part_S4 = A11 - A00
    # B_part_S4 = B00 + B10
    
    # A_part_S5 = A00 + A01
    # B_part_S5 = B10
    
    # A_part_S6 = A01 - A11
    # B_part_S6 = B00 + B01
    
    # A_part_S7 = A10 - A00
    # B_part_S7 = B01 + B11
    
    # Now, for each Sk, we apply the inner Strassen decomposition.
    # The inner Strassen decomposition takes A_part and B_part (which are 2x2 matrices)
    # and produces 7 scalar products.
    
    # Let's write a helper to add the inner Strassen terms.
    
    def add_inner_strassen(col_offset, a_coeffs, b_coeffs, c_coeffs):
        # a_coeffs: dict {(bi, bj, ai, aj): coeff} for A block elements
        # b_coeffs: dict {(bi, bj, bi, bj): coeff} for B block elements
        # c_coeffs: dict {(bi, bj, ci, cj): coeff} for C block elements
        # This is not quite right. The inner Strassen operates on the elements of the 2x2 matrices.
        
        # Let's just hardcode the 49 terms.
        pass

    # Given the complexity, I will construct the 49 terms by iterating.
    
    # Outer Strassen Factors
    outer_factors = [
        # (A_combos, B_combos)
        # A_combos: list of (block_i, block_j, coeff)
        # B_combos: list of (block_i, block_j, coeff)
        (
            [(0,0,1), (1,1,1)], # A00 + A11
            [(0,0,1), (1,1,1)]  # B00 + B11
        ),
        (
            [(1,0,1), (1,1,1)], # A10 + A11
            [(1,1,1)]           # B11
        ),
        (
            [(0,0,1)],          # A00
            [(0,1,1), (1,1,-1)] # B01 - B11
        ),
        (
            [(1,1,1), (0,0,-1)], # A11 - A00
            [(0,0,1), (1,0,1)]   # B00 + B10
        ),
        (
            [(0,0,1), (0,1,1)], # A00 + A01
            [(1,0,1)]           # B10
        ),
        (
            [(0,1,1), (1,1,-1)], # A01 - A11
            [(0,0,1), (0,1,1)]   # B00 + B01
        ),
        (
            [(1,0,1), (0,0,-1)], # A10 - A00
            [(0,1,1), (1,1,1)]   # B01 + B11
        )
    ]
    
    # Inner Strassen Factors (for 2x2x2)
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B11
    # P3 = A00(B01 - B11)
    # P4 = (A11 - A00)(B00 + B10)
    # P5 = (A00 + A01)B10
    # P6 = (A01 - A11)(B00 + B01)
    # P7 = (A10 - A00)(B01 + B11)
    
    inner_factors = [
        (
            [(0,0,1), (1,1,1)], # A00 + A11
            [(0,0,1), (1,1,1)]  # B00 + B11
        ),
        (
            [(1,0,1), (1,1,1)], # A10 + A11
            [(1,1,1)]           # B11
        ),
        (
            [(0,0,1)],          # A00
            [(0,1,1), (1,1,-1)] # B01 - B11
        ),
        (
            [(1,1,1), (0,0,-1)], # A11 - A00
            [(0,0,1), (1,0,1)]   # B00 + B10
        ),
        (
            [(0,0,1), (0,1,1)], # A00 + A01
            [(1,0,1)]           # B10
        ),
        (
            [(0,1,1), (1,1,-1)], # A01 - A11
            [(0,0,1), (0,1,1)]   # B00 + B01
        ),
        (
            [(1,0,1), (0,0,-1)], # A10 - A00
            [(0,1,1), (1,1,1)]   # B01 + B11
        )
    ]
    
    col = 0
    for outer_idx, (a_outer, b_outer) in enumerate(outer_factors):
        for inner_idx, (a_inner, b_inner) in enumerate(inner_factors):
            # We need to combine the coefficients.
            # The term is a product of linear combinations of A elements and B elements.
            # We add this term to U, V, W.
            
            # U part: linear combo of A elements
            # V part: linear combo of B elements
            # W part: linear combo of C elements?
            # No, the decomposition is:
            # C = sum_r U_r * V_r * W_r
            # Where U_r, V_r, W_r are vectors.
            # Actually, the rank-1 term is u * v * w.
            # u is a vector of A coefficients.
            # v is a vector of B coefficients.
            # w is a vector of C coefficients.
            
            # For a term like (A00 + A11)(B00 + B11) -> C...
            # The A part is A00 + A11.
            # The B part is B00 + B11.
            # The C part is determined by the reconstruction formula.
            
            # Wait, the inner Strassen decomposition produces 7 terms P1..P7.
            # Each Pk is a product of two linear combinations.
            # The C matrix is a linear combination of P1..P7.
            
            # So for each outer factor Sk, we have 7 inner terms.
            # Each inner term contributes to the C blocks.
            
            # Let's trace one term.
            # Outer S1 = (A00 + A11)(B00 + B11).
            # Inner P1 for S1: ((A00+A11)_00 + (A00+A11)_11) * ((B00+B11)_00 + (B00+B11)_11)
            # This is getting messy.
            
            # Let's use a different approach.
            # Just iterate through all 49 terms and fill U, V, W.
            
            # A element (bi, bj, ai, aj) is at index (2*bi + ai)*4 + (2*bj + aj).
            # B element (bi, bj, ci, cj) is at index (2*bi + ci)*6 + (2*bj + cj).
            # C element (bi, bj, di, dj) is at index (2*bi + di)*4 + (2*bj + dj).
            
            # We need to determine the C coefficients for each inner term.
            # The inner Strassen reconstruction:
            # C00 = P1 + P4 - P5 + P7
            # C01 = P3 + P5
            # C10 = P2 + P6
            # C11 = P1 - P2 + P3 + P4
            
            # So for a given inner term index (0..6), we know which C blocks it contributes to and with what sign.
            
            c_contributions = {
                0: [(0,0,1), (1,1,1)], # P1 -> C00 (+1), C11 (+1)
                1: [(1,0,1), (1,1,-1)], # P2 -> C10 (+1), C11 (-1)
                2: [(0,1,1), (1,1,1)], # P3 -> C01 (+1), C11 (+1)
                3: [(0,0,1), (1,1,1)], # P4 -> C00 (+1), C11 (+1) -- Wait, P4 is C00 (+1), C11 (+1)
                4: [(0,0,-1), (0,1,1)], # P5 -> C00 (-1), C01 (+1)
                5: [(1,0,1), (0,1,1)], # P6 -> C10 (+1), C01 (+1)
                6: [(0,0,1), (1,1,1)]  # P7 -> C00 (+1), C11 (+1) -- Wait, P7 is C00 (+1), C11 (+1)
            }
            # Correction on P4, P7 contributions:
            # C00 = P1 + P4 - P5 + P7
            # C01 = P3 + P5
            # C10 = P2 + P6
            # C11 = P1 - P2 + P3 + P4
            
            # P1: C00(+1), C11(+1)
            # P2: C10(+1), C11(-1)
            # P3: C01(+1), C11(+1)
            # P4: C00(+1), C11(+1)
            # P5: C00(-1), C01(+1)
            # P6: C10(+1), C01(+1)
            # P7: C00(+1), C11(+1)
            
            # Let's re-verify P4 and P7.
            # P4 = (A11 - A00)(B00 + B10).
            # P7 = (A10 - A00)(B01 + B11).
            
            # The C contributions are correct.
            
            # Now, for each outer factor, we apply the inner Strassen.
            # The A and B parts of the inner Strassen are the linear combinations from the outer factor.
            
            # Example: Outer S1.
            # A_part = A00 + A11.
            # B_part = B00 + B11.
            
            # Inner P1 = (A_part_00 + A_part_11)(B_part_00 + B_part_11).
            # A_part_00 is the (0,0) element of the matrix (A00 + A11).
            # A00 is a 2x2 matrix. A11 is a 2x2 matrix.
            # (A00 + A11)_00 = A00_00 + A11_00.
            
            # So we need to expand the linear combinations.
            
            # Let's define a function to expand a linear combination of blocks into a dict of element coefficients.
            
            def expand_block_combos(block_combos):
                # block_combos: list of (bi, bj, coeff)
                # Returns dict {(bi, bj, ai, aj): coeff}
                res = {}
                for bi, bj, coeff in block_combos:
                    for ai in range(2):
                        for aj in range(2):
                            key = (bi, bj, ai, aj)
                            res[key] = res.get(key, 0) + coeff
                return res

            a_expanded = expand_block_combos(a_outer)
            b_expanded = expand_block_combos(b_outer)
            
            # Now, for each inner factor, we need to compute the linear combination of elements.
            # The inner factor uses the same structure as the 2x2x2 Strassen.
            # But the "elements" of the 2x2 matrices are now the elements of the expanded blocks.
            
            # Wait, the inner Strassen operates on the 2x2 matrices A_part and B_part.
            # The elements of A_part are linear combinations of A elements.
            # So the inner Strassen term Pk is a product of two linear combinations of A elements and B elements.
            
            # We can just add the term to U, V, W.
            
            # For a given inner factor index (0..6):
            # We determine the A and B linear combinations from the inner_factors definition.
            # But the inner_factors definition uses block indices (0,0), (0,1), etc.
            # These block indices refer to the elements of the 2x2 matrix A_part.
            # So (0,0) in inner_factors means the (0,0) element of A_part.
            
            # So we need to map the inner block indices to the actual A element indices.
            
            # Let's precompute the expansion for each inner factor.
            
            for inner_idx, (a_inner_blocks, b_inner_blocks) in enumerate(inner_factors):
                # a_inner_blocks is a list of (bi, bj, coeff) where bi, bj are indices of the 2x2 matrix A_part.
                # We need to map these to the actual A element indices.
                
                a_elem_combos = []
                for bi, bj, coeff in a_inner_blocks:
                    # bi, bj are indices in the 2x2 matrix A_part.
                    # A_part is a linear combination of A blocks.
                    # The (bi, bj) element of A_part is the sum of coeff * A_block(bi, bj) for all blocks in a_outer?
                    # No, a_outer is the linear combination of A blocks.
                    # A_part = sum_{k} c_k * A_block_k.
                    # The (bi, bj) element of A_part is sum_{k} c_k * A_block_k(bi, bj).
                    
                    # So we need to iterate over a_outer and add the contributions.
                    
                    for bi_outer, bj_outer, coeff_outer in a_outer:
                        # The (bi, bj) element of A_block(bi_outer, bj_outer) is A_element(bi_outer, bj_outer, bi, bj).
                        a_elem_combos.append((bi_outer, bj_outer, bi, bj, coeff_outer * coeff))
                
                b_elem_combos = []
                for bi, bj, coeff in b_inner_blocks:
                    for bi_outer, bj_outer, coeff_outer in b_outer:
                        b_elem_combos.append((bi_outer, bj_outer, bi, bj, coeff_outer * coeff))
                
                # Now we have the linear combinations of A and B elements for this inner term.
                # We add this term to U, V, W.
                
                # U part:
                for bi, bj, ai, aj, coeff in a_elem_combos:
                    if coeff != 0:
                        u_idx = (2*bi + ai)*4 + (2*bj + aj)
                        U[u_idx][col] += coeff
                
                # V part:
                for bi, bj, ci, cj, coeff in b_elem_combos:
                    if coeff != 0:
                        v_idx = (2*bi + ci)*6 + (2*bj + cj)
                        V[v_idx][col] += coeff
                
                # W part:
                # We need to determine the C contributions for this inner term.
                # The C contributions are determined by the inner_idx (0..6) and the outer_idx?
                # No, the C contributions are the same for all outer factors, because the inner Strassen is the same.
                # Wait, the C matrix is the sum of the outer factors.
                # C = S1 + S2 + ... + S7?
                # No, the Strassen reconstruction for the 2x2 block matrix is:
                # C00 = S1 + S4 - S5 + S7
                # C01 = S3 + S5
                # C10 = S2 + S6
                # C11 = S1 - S2 + S3 + S4
                
                # So for a given outer_idx, the C contributions are fixed.
                
                c_outer_contributions = {
                    0: [(0,0,1), (1,1,1)], # S1
                    1: [(1,0,1), (1,1,-1)], # S2
                    2: [(0,1,1), (1,1,1)], # S3
                    3: [(0,0,1), (1,1,1)], # S4
                    4: [(0,0,-1), (0,1,1)], # S5
                    5: [(1,0,1), (0,1,1)], # S6
                    6: [(0,0,1), (1,1,1)]  # S7
                }
                
                # But wait, the inner Strassen also has C contributions.
                # The inner Strassen produces P1..P7.
                # C_inner = P1 + ...
                # So the total C contribution is the product of the outer C contribution and the inner C contribution?
                # No, the outer factors S1..S7 are themselves 2x2 matrices.
                # So S1 is a 2x2 matrix.
                # The inner Strassen decomposes the multiplication of the two 2x2 matrices that form S1.
                # So the result of the inner Strassen is the 2x2 matrix S1.
                # And S1 contributes to C with the coefficients in c_outer_contributions.
                
                # So for a given outer_idx and inner_idx, the C contribution is:
                # (c_outer_contributions[outer_idx]) * (c_inner_contributions[inner_idx])
                
                # Let's verify this.
                # S1 = A_part * B_part.
                # S1 is a 2x2 matrix.
                # The inner Strassen decomposes A_part * B_part into 7 terms P1..P7.
                # S1 = P1 + P4 - P5 + P7 (for element 00)
                # S1 = P3 + P5 (for element 01)
                # S1 = P2 + P6 (for element 10)
                # S1 = P1 - P2 + P3 + P4 (for element 11)
                
                # And C00 = S1 + S4 - S5 + S7.
                # So C00 = (P1 + P4 - P5 + P7)_from_S1 + (P1 + P4 - P5 + P7)_from_S4 - ...
                
                # So the total C contribution for a specific Pk from Sk is:
                # c_outer[outer_idx][C_element] * c_inner[inner_idx][C_element]
                
                # Let's implement this.
                
                c_inner_contributions = {
                    0: [(0,0,1), (1,1,1)], # P1
                    1: [(1,0,1), (1,1,-1)], # P2
                    2: [(0,1,1), (1,1,1)], # P3
                    3: [(0,0,1), (1,1,1)], # P4
                    4: [(0,0,-1), (0,1,1)], # P5
                    5: [(1,0,1), (0,1,1)], # P6
                    6: [(0,0,1), (1,1,1)]  # P7
                }
                
                for ci, cj, coeff_outer_c in c_outer_contributions[outer_idx]:
                    for ci_inner, cj_inner, coeff_inner_c in c_inner_contributions[inner_idx]:
                        # The C element is (ci, cj) in the block matrix.
                        # The inner contribution is to element (ci_inner, cj_inner) of the block S_outer_idx.
                        # This doesn't make sense. The inner Strassen produces the elements of S_outer_idx.
                        # So the element (ci_inner, cj_inner) of S_outer_idx is a linear combination of P1..P7.
                        # And S_outer_idx contributes to C with the coefficients in c_outer_contributions.
                        
                        # So the total contribution to C(ci, cj) is:
                        # sum_{inner_idx} coeff_inner_c[inner_idx][ci_inner, cj_inner] * coeff_outer_c[outer_idx][ci, cj]?
                        # No.
                        # S_outer_idx(ci_inner, cj_inner) = sum_{inner_idx} coeff_inner_c[inner_idx][ci_inner, cj_inner] * P_inner_idx.
                        # C(ci, cj) = sum_{outer_idx} coeff_outer_c[outer_idx][ci, cj] * S_outer_idx(ci, cj).
                        
                        # So the contribution of P_inner_idx to C(ci, cj) is:
                        # coeff_outer_c[outer_idx][ci, cj] * coeff_inner_c[inner_idx][ci, cj]?
                        # No, the indices are different.
                        # S_outer_idx is a 2x2 matrix.
                        # C is a 4x4 matrix.
                        # S_outer_idx contributes to a 2x2 block of C.
                        # The block is at (ci, cj) in the block matrix.
                        # So the element C(2*ci + r, 2*cj + s) gets a contribution from S_outer_idx(r, s).
                        
                        # So for a given outer_idx, inner_idx:
                        # The term P_inner_idx contributes to S_outer_idx(r, s) with coeff_inner_c[inner_idx][r, s].
                        # And S_outer_idx(r, s) contributes to C(2*ci + r, 2*cj + s) with coeff_outer_c[outer_idx][ci, cj].
                        
                        # So the total contribution to C(2*ci + r, 2*cj + s) is:
                        # coeff_outer_c[outer_idx][ci, cj] * coeff_inner_c[inner_idx][r, s].
                        
                        # We need to add this to W.
                        
                        for ci in range(2):
                            for cj in range(2):
                                coeff_outer_c_val = 0
                                for cci, ccj, cval in c_outer_contributions[outer_idx]:
                                    if cci == ci and ccj == cj:
                                        coeff_outer_c_val = cval
                                        
                                if coeff_outer_c_val == 0:
                                    continue
                                    
                                for r in range(2):
                                    for s in range(2):
                                        coeff_inner_c_val = 0
                                        for cci, ccj, cval in c_inner_contributions[inner_idx]:
                                            if cci == r and ccj == s:
                                                coeff_inner_c_val = cval
                                                
                                        if coeff_inner_c_val == 0:
                                            continue
                                        
                                        total_coeff = coeff_outer_c_val * coeff_inner_c_val
                                        if total_coeff != 0:
                                            # C index: (2*ci + r)*4 + (2*cj + s)
                                            w_idx = (2*ci + r)*4 + (2*cj + s)
                                            W[w_idx][col] += total_coeff
                col += 1
    
    # Part 2: 4x4x2 using Schoolbook (rank 8)
    # A is 4x4, B is 4x2.
    # C is 4x2.
    # C_i0 = sum_j A_ij B_j0
    # C_i1 = sum_j A_ij B_j1
    
    for i in range(4):
        for j in range(4):
            # Term for C_i0
            add_term(i*4 + j, j*6 + 0, 0*4 + i, col)
            col += 1
            # Term for C_i1
            add_term(i*4 + j, j*6 + 1, 1*4 + i, col)
            col += 1
    
    return {"U": U, "V": V, "W": W}
