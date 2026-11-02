"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block size is 3
    block_size = 3
    num_blocks = 2
    
    # Strassen coefficients for 2x2 block multiplication
    # M_s = L_A * L_B
    # L_A is 2x2 block matrix of linear combos of A blocks
    # L_B is 2x2 block matrix of linear combos of B blocks
    
    # Define Strassen terms
    # Each term is a list of (coeff, A_block, B_block) for the linear combinations
    # Then we compute the product of the resulting 3x3 matrices
    
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # M1
        ([ (1, 0, 0), (1, 1, 1) ], [ (1, 0, 0), (1, 1, 1) ]), # (coeff, block_i, block_j)
        # M2
        ([ (1, 1, 0), (1, 1, 1) ], [ (1, 0, 0) ]),
        # M3
        ([ (1, 0, 0) ], [ (1, 0, 1), (-1, 1, 1) ]),
        # M4
        ([ (1, 1, 1) ], [ (1, 1, 0), (-1, 0, 0) ]),
        # M5
        ([ (1, 0, 0), (1, 0, 1) ], [ (1, 1, 1) ]),
        # M6
        ([ (1, 1, 0), (-1, 0, 0) ], [ (1, 0, 0), (1, 0, 1) ]),
        # M7
        ([ (1, 0, 1), (-1, 1, 1) ], [ (1, 1, 0), (1, 1, 1) ])
    ]
    
    reconstruction = {
        0: [(0, 1), (3, 1), (4, -1), (6, 1)],   # C00: M1 + M4 - M5 + M7
        1: [(2, 1), (4, 1)],                    # C01: M3 + M5
        2: [(1, 1), (3, 1)],                    # C10: M2 + M4
        3: [(0, 1), (1, -1), (2, 1), (5, 1)]    # C11: M1 - M2 + M3 + M6
    }
    
    # Initialize matrices
    rank = 189 # 7 * 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for s_idx, (l_a_terms, l_b_terms) in enumerate(strassen_terms):
        # l_a_terms: list of (coeff, block_i, block_j)
        # l_b_terms: list of (coeff, block_i, block_j)
        
        # Construct L_A and L_B as 3x3 matrices of coefficients for each A_block/B_block
        # L_A[i_loc, j_loc] is a sum of coeff * A[block_i, block_j]
        # We represent L_A as a dict mapping (i_loc, j_loc) to list of (coeff, a_block_i, a_block_j)
        # Actually, we can just iterate over the expansion directly.
        
        # For each i_loc, k_loc in 0..2
        for i_loc in range(block_size):
            for k_loc in range(block_size):
                # M_s[i_loc, k_loc] = sum_{j_loc} L_A[i_loc, j_loc] * L_B[j_loc, k_loc]
                # We need to expand this product.
                
                # Get terms for L_A[i_loc, j_loc] and L_B[j_loc, k_loc]
                # Note: L_A and L_B are 3x3 matrices where each entry is a linear combo of blocks.
                # The block index in L_A corresponds to the A block.
                # L_A[i_loc, j_loc] = sum_{(c, bi, bj) in l_a_terms if bi==? and bj==?} ...
                # Wait, l_a_terms defines the linear combination for the ENTIRE 2x2 block matrix.
                # L_A is a 2x2 block matrix. Each block is 3x3.
                # So L_A[bi, bj] is a 3x3 matrix.
                # The entry L_A[bi, bj][i_loc, j_loc] is the coefficient for A[bi, bj][i_loc, j_loc].
                
                # Let's re-evaluate.
                # l_a_terms is a list of (coeff, block_i, block_j).
                # This means L_A[block_i, block_j] += coeff * Identity? No.
                # The standard Strassen formula is:
                # M1 = (A00 + A11)(B00 + B11)
                # This means the matrix L_A has A00 at block (0,0) and A11 at block (1,1).
                # So L_A is a 2x2 block matrix.
                # L_A[0,0] = A00
                # L_A[1,1] = A11
                # L_A[0,1] = 0
                # L_A[1,0] = 0
                
                # So for a specific term (coeff, bi, bj) in l_a_terms:
                # It contributes to the block (bi, bj) of L_A.
                # The entries of this block are A[bi, bj][i_loc, j_loc].
                
                # So, for a fixed i_loc, j_loc:
                # The coefficient of A[bi, bj][i_loc, j_loc] in L_A[i_loc, j_loc] is:
                # Sum of coeff for (coeff, bi, bj) in l_a_terms.
                # BUT, L_A is a 2x2 block matrix. The index i_loc, j_loc is within the 3x3 block.
                # The block index (bi, bj) determines WHICH 3x3 block of A is used.
                # The local indices (i_loc, j_loc) are the same for all blocks in that linear combination.
                
                # So, L_A[i_loc, j_loc] (as an element of the big 6x6 matrix) is:
                # Sum over (c, bi, bj) in l_a_terms of c * A[bi, bj][i_loc, j_loc]
                # Wait, this is only true if the linear combination applies to the same local indices.
                # Yes, A00 and A11 are both 3x3 matrices. A00[i,j] and A11[i,j] are added.
                
                # So for each j_loc:
                for j_loc in range(block_size):
                    # Get coeff_A for A[bi, bj][i_loc, j_loc]
                    # This is sum of c for (c, bi, bj) in l_a_terms
                    # BUT, we need to keep track of which (bi, bj) it comes from to match with B.
                    # And we need to sum over the product.
                    
                    # Let's collect all (c_a, bi, bj) from l_a_terms
                    # and all (c_b, bj2, bk) from l_b_terms
                    # The product term is non-zero only if bj == bj2.
                    
                    for c_a, bi, bj in l_a_terms:
                        for c_b, bj2, bk in l_b_terms:
                            if bj == bj2:
                                # We have a term c_a * c_b * A[bi, bj][i_loc, j_loc] * B[bj, bk][j_loc, k_loc]
                                coeff = c_a * c_b
                                if coeff == 0:
                                    continue
                                
                                # This term contributes to M_s[i_loc, k_loc]
                                # M_s is added to C blocks according to reconstruction[s_idx]
                                # Wait, the reconstruction formula maps M_s to C blocks.
                                # M_s is a 3x3 matrix.
                                # M_s[i_loc, k_loc] contributes to C[bi_target, bk_target][i_loc, k_loc]
                                # where (bi_target, bk_target) is the block index in C.
                                
                                # Find which C block(s) receive M_s
                                # reconstruction[s_idx] is a list of (c_block_idx, coeff)
                                # c_block_idx 0->C00, 1->C01, 2->C10, 3->C11
                                # Wait, my reconstruction dict keys are 0,1,2,3 corresponding to C00, C01, C10, C11.
                                # But I need to map s_idx to the C blocks it contributes to.
                                # My reconstruction dict is structured as:
                                # reconstruction[c_block_idx] = list of (s_idx, coeff)
                                # So I need to iterate over c_block_idx and check if s_idx is present.
                                
                                for c_block_idx, terms in reconstruction.items():
                                    for s_ref, c_coeff in terms:
                                        if s_ref == s_idx:
                                            # This M_s contributes to C block c_block_idx with coefficient c_coeff
                                            # c_block_idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                                            ci_block = c_block_idx // 2
                                            cj_block = c_block_idx % 2
                                            
                                            # Global indices
                                            I = block_size * ci_block + i_loc
                                            J = block_size * bj + j_loc
                                            K = block_size * cj_block + k_loc
                                            
                                            # Wait, K index in C is determined by the column of B.
                                            # B block is (bj, bk). So K block is bk.
                                            # But the reconstruction formula says M_s goes to C block (ci_block, cj_block).
                                            # Does this match bk?
                                            # Let's check M1.
                                            # M1 = (A00+A11)(B00+B11).
                                            # A00 B00 -> C00. (bi=0, bj=0, bk=0). C block 0,0. Match.
                                            # A11 B11 -> C11. (bi=1, bj=1, bk=1). C block 1,1. Match.
                                            # A00 B11 -> 0.
                                            # A11 B00 -> 0.
                                            # So for valid terms, bk should equal cj_block?
                                            # Let's check M2 = (A10+A11)B00.
                                            # A10 B00 -> C10. (bi=1, bj=0, bk=0). C block 1,0.
                                            # reconstruction for C10 includes M2. c_block_idx=2 (1,0).
                                            # bi=1, bj=0, bk=0.
                                            # ci_block=1, cj_block=0.
                                            # bk=0 == cj_block=0. Match.
                                            
                                            # So we must use bk from the B term, not cj_block?
                                            # Actually, if the term is valid, bk == cj_block.
                                            # But to be safe, we should use bk for K block index.
                                            # However, the W matrix index depends on K and I.
                                            # K = block_size * bk + k_loc
                                            # I = block_size * ci_block + i_loc
                                            
                                            # Let's verify if bk always equals cj_block for the terms in reconstruction.
                                            # If not, the Strassen formula would be wrong.
                                            # It is correct, so they should match.
                                            
                                            K = block_size * bk + k_loc
                                            
                                            u_idx = I * m + J
                                            v_idx = J * p + K
                                            w_idx = K * n + I
                                            
                                            val = coeff * c_coeff
                                            
                                            U[u_idx][col] = val
                                            V[v_idx][col] = 1
                                            W[w_idx][col] = 1
                                            col += 1
                                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
