import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank will be 7 * 27 = 189
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen products for 2x2 blocks of 3x3
    # Blocks: 00, 01, 10, 11
    # A00: i in 0..2, j in 0..2
    # A01: i in 0..2, j in 3..5
    # A10: i in 3..5, j in 0..2
    # A11: i in 3..5, j in 3..5
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # We need to map local 3x3 indices (a, c) to global 6x6 indices (i, j)
    # and handle the linear combinations.
    
    # Helper to add a term A[i,j] * B[j,k] with coefficient coeff to U, V, W at column col
    def add_term(i, j, k, coeff, col):
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        # Wait, the decomposition is U * V * W?
        # The tensor is T = sum U_r V_r W_r.
        # If we have a term c * A_ij B_jk, then U_ij = c, V_jk = 1, W_ki = 1?
        # Or U_ij = sqrt(c), etc?
        # The task allows integer/half-integer.
        # If we set U=c, V=1, W=1, then product is c.
        # This works.
        
    col = 0
    # Define the 7 products as lists of (A_block, B_block, sign)
    # Each product is a sum of terms.
    # Since we are using schoolbook 3x3, we iterate over all 3x3=9 terms for each product.
    
    # Product 1: (A00 + A11)(B00 + B11)
    # A terms: A00 (coeff 1), A11 (coeff 1)
    # B terms: B00 (coeff 1), B11 (coeff 1)
    # Expansion: A00*B00 + A00*B11 + A11*B00 + A11*B11
    # Each of these is a 3x3 * 3x3 multiplication.
    # We need to expand each 3x3 * 3x3 into 27 terms.
    
    def expand_3x3(A_block, B_block, sign, col_start):
        # A_block is in {0, 1, 2, 3} representing 00, 01, 10, 11
        # B_block is in {0, 1, 2, 3}
        # Returns number of columns used
        used = 0
        # Local indices a, c for A; c, b for B?
        # A is 3x3. B is 3x3.
        # Product C = A B.
        # C_ab = sum_c A_ac B_cb.
        # But we are decomposing the multiplication tensor.
        # We need to generate terms A_ij B_jk.
        # For a specific 3x3 block multiplication, the indices are local.
        # Let local row of A be r, local col of A be c.
        # Local row of B be c, local col of B be l.
        # Global indices:
        # If A_block is 00 (top-left): i = r, j = c.
        # If A_block is 01 (top-right): i = r, j = c + 3.
        # If A_block is 10 (bottom-left): i = r + 3, j = c.
        # If A_block is 11 (bottom-right): i = r + 3, j = c + 3.
        
        # Similarly for B_block.
        # B_block 00: j = c, k = l.
        # B_block 01: j = c, k = l + 3.
        # B_block 10: j = c + 3, k = l.
        # B_block 11: j = c + 3, k = l + 3.
        
        # Wait, the term is A_ij B_jk.
        # The column index j of A must match the row index j of B.
        # In the block multiplication, the "inner" index is the column of A and row of B.
        # For A_block and B_block to be multiplied, their inner dimensions must align.
        # A_block 00 (cols 0-2) * B_block 00 (rows 0-2) -> OK.
        # A_block 00 (cols 0-2) * B_block 10 (rows 3-5) -> Mismatch!
        
        # Strassen's algorithm works on the block matrices.
        # M1 = (A00 + A11)(B00 + B11).
        # A00 is 3x3, A11 is 3x3. Their sum is 3x3.
        # B00 is 3x3, B11 is 3x3. Their sum is 3x3.
        # The product is 3x3.
        # When we expand (A00 + A11)(B00 + B11), we get A00B00 + A00B11 + A11B00 + A11B11.
        # A00B00: A00 cols 0-2, B00 rows 0-2. Match.
        # A00B11: A00 cols 0-2, B11 rows 3-5. Mismatch!
        
        # Ah, Strassen's algorithm is for matrix multiplication of the block matrices.
        # The blocks are just matrices. The multiplication is standard matrix multiplication.
        # The fact that they come from different positions in the 6x6 matrix doesn't affect the local 3x3 multiplication logic.
        # The "mismatch" is irrelevant because we are treating them as abstract 3x3 matrices X and Y.
        # X = A00 + A11. Y = B00 + B11.
        # Z = X Y.
        # Z is a 3x3 matrix.
        # We then place Z into the correct position in the 6x6 result C.
        
        # So, the decomposition of Z = X Y into rank-1 tensors is independent of where X and Y come from.
        # We just need to express X and Y in terms of A_ij and B_jk.
        # X_ac = sum_{i,j in block} coeff * A_ij.
        # Y_cb = sum_{j',k' in block} coeff * B_j'k'.
        # Z_ab = sum_c X_ac Y_cb.
        # Z_ab = sum_c sum_{i,j} sum_{j',k'} coeff_A coeff_B A_ij B_j'k'.
        # Note: The index j in A and j' in B are local indices in the 3x3 block.
        # They are NOT the same as the global index j.
        # However, the term is A_ij B_j'k'.
        # This is a bilinear term in A and B.
        # It contributes to C_ab (local).
        # The global position of C_ab depends on which block of C it is.
        
        # So, for each of the 7 Strassen products, we:
        # 1. Determine the linear forms X_ac and Y_cb in terms of global A_ij and B_jk.
        # 2. Expand Z_ab = sum_c X_ac Y_cb.
        # 3. For each term A_ij B_jk with coefficient coeff, add it to the decomposition.
        # 4. The W index is determined by the global position of C_ab.
        
        # This logic holds.
        
        # Let's implement the expansion for one product.
        # We need to know the global indices for A and B terms.
        
        # A_block 0: i = r, j = c
        # A_block 1: i = r, j = c + 3
        # A_block 2: i = r + 3, j = c
        # A_block 3: i = r + 3, j = c + 3
        
        # B_block 0: j = c, k = l
        # B_block 1: j = c, k = l + 3
        # B_block 2: j = c + 3, k = l
        # B_block 3: j = c + 3, k = l + 3
        
        # For a product term A_block_a * B_block_b:
        # X is formed from A_block_a. Y is formed from B_block_b.
        # Wait, X and Y are sums of blocks.
        # e.g. X = A00 + A11.
        # X_ac = A00_ac + A11_ac.
        # A00_ac is A_{r, c} where r,c are local indices for block 00.
        # A11_ac is A_{r+3, c+3}.
        
        # So for each local a, c, b:
        # Z_ab += X_ac Y_cb.
        # X_ac = sum_{block_in_X} coeff_X * A_{global_i, global_j}
        # Y_cb = sum_{block_in_Y} coeff_Y * B_{global_j', global_k}
        
        # We iterate over all combinations of blocks in X and Y.
        
        pass

    # Let's hardcode the 7 products' block compositions.
    # X_blocks: list of (block_id, coeff)
    # Y_blocks: list of (block_id, coeff)
    
    products = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]), # (A00+A11)(B00+B11)
        # M2
        ([(2, 1), (3, 1)], [(0, 1)]), # (A10+A11)B00
        # M3
        ([(0, 1)], [(1, 1), (3, -1)]), # A00(B01-B11)
        # M4
        ([(3, 1)], [(2, 1), (0, -1)]), # A11(B10-B00)
        # M5
        ([(0, 1), (1, 1)], [(3, 1)]), # (A00+A01)B11
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]), # (A10-A00)(B00+B01)
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]) # (A01-A11)(B10+B11)
    ]
    
    # C_block mapping for Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    c_block_coeffs = [
        # M1, M2, M3, M4, M5, M6, M7
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # C_block 0: i in 0..2, k in 0..2
    # C_block 1: i in 0..2, k in 3..5
    # C_block 2: i in 3..5, k in 0..2
    # C_block 3: i in 3..5, k in 3..5
    
    # For each product p_idx, we generate 27 terms (schoolbook 3x3).
    # Each term corresponds to a local a, c, b? No.
    # Schoolbook 3x3: Z_ab = sum_c A_ac B_cb.
    # The rank-1 terms are A_ac B_cb for fixed a, c, b?
    # No, the term is A_ac B_cb.
    # The output is Z_ab.
    # So for each a, c, b, we have a term A_ac B_cb that contributes to Z_ab.
    # There are 3*3*3 = 27 such terms.
    
    for p_idx, (x_blocks, y_blocks) in enumerate(products):
        for a in range(3):
            for c in range(3):
                for b in range(3):
                    # This term is A_ac B_cb contributing to Z_ab.
                    # We need to expand A_ac and B_cb in terms of global indices.
                    # A_ac is a sum of global A_ij.
                    # B_cb is a sum of global B_jk.
                    
                    # A_ac = sum_{(bx, cx)} coeff_x * A_global(i, j)
                    # where i, j depend on bx, cx.
                    
                    a_terms = []
                    for bx, cx in x_blocks:
                        # bx is block id, cx is coeff
                        # Local indices a, c
                        # Global i, j
                        if bx == 0:
                            gi, gj = a, c
                        elif bx == 1:
                            gi, gj = a, c + 3
                        elif bx == 2:
                            gi, gj = a + 3, c
                        elif bx == 3:
                            gi, gj = a + 3, c + 3
                        a_terms.append((gi, gj, cx))
                        
                    b_terms = []
                    for by, cy in y_blocks:
                        # by is block id, cy is coeff
                        # Local indices c, b
                        # Global j, k
                        if by == 0:
                            gj, gk = c, b
                        elif by == 1:
                            gj, gk = c, b + 3
                        elif by == 2:
                            gj, gk = c + 3, b
                        elif by == 3:
                            gj, gk = c + 3, b + 3
                        b_terms.append((gj, gk, cy))
                        
                    # Now combine
                    for gi, gj, cx in a_terms:
                        for gk, gk2, cy in b_terms:
                            # Wait, B term is (gj, gk, cy).
                            # The term is A(gi, gj) * B(gj, gk).
                            # The middle index must match: gj from A must equal gj from B.
                            # In Strassen, the blocks are aligned.
                            # But here, the local index c is the same for A and B.
                            # However, the global index gj might be different if the blocks are different?
                            # No, the local index c is the column of A and row of B.
                            # For A_block bx, global col is determined by bx and c.
                            # For B_block by, global row is determined by by and c.
                            # For the product to be valid in the original 6x6 matrix,
                            # the global column of A must equal the global row of B.
                            # This is guaranteed by the Strassen construction?
                            # Let's check M1: (A00+A11)(B00+B11).
                            # Term A00 * B00: A col c (0-2), B row c (0-2). Match.
                            # Term A00 * B11: A col c (0-2), B row c+3 (3-5). Mismatch!
                            
                            # This implies that the term A00 * B11 is NOT a valid term in the 6x6 multiplication?
                            # No, it IS a term in the matrix product of the block matrices.
                            # The block matrix multiplication is an abstract operation.
                            # The result Z is a 3x3 matrix.
                            # The fact that the indices don't match in the global 6x6 matrix is irrelevant
                            # because we are computing the product of the block matrices, not the sub-matrices of the 6x6.
                            # The "mismatch" just means that the term A00*B11 is a product of two 3x3 matrices
                            # that happen to be located at different positions in the 6x6 matrix.
                            # The multiplication is still well-defined.
                            # The resulting Z_ab is a 3x3 matrix.
                            # We then place Z_ab into the correct position in C.
                            
                            # So, we just add the term A(gi, gj) * B(gj, gk) to the decomposition.
                            # Wait, if gj from A != gj from B, is it a valid bilinear term?
                            # Yes, A and B are just matrices. A_ij and B_kl are independent variables.
                            # The term is A_ij B_kl.
                            # It contributes to C_uv.
                            # The decomposition is valid as long as the sum of terms equals the matrix product.
                            
                            coeff = cx * cy
                            if coeff == 0:
                                continue
                            
                            # Which C block does this contribute to?
                            # The product p_idx contributes to C blocks with coefficients c_block_coeffs[cb][p_idx].
                            # The local indices a, b determine the position within the C block.
                            # Global C indices:
                            # For C_block cb:
                            # if cb == 0: gi_c = a, gj_c = b
                            # if cb == 1: gi_c = a, gj_c = b + 3
                            # if cb == 2: gi_c = a + 3, gj_c = b
                            # if cb == 3: gi_c = a + 3, gj_c = b + 3
                            
                            for cb in range(4):
                                c_coeff = c_block_coeffs[cb][p_idx]
                                if c_coeff == 0:
                                    continue
                                total_coeff = coeff * c_coeff
                                if total_coeff == 0:
                                    continue
                                
                                # Global C indices
                                if cb == 0:
                                    ci, ck = a, b
                                elif cb == 1:
                                    ci, ck = a, b + 3
                                elif cb == 2:
                                    ci, ck = a + 3, b
                                elif cb == 3:
                                    ci, ck = a + 3, b + 3
                                
                                # Add term
                                # U index: gi * m + gj (from A)
                                # V index: gj * p + gk (from B)
                                # W index: ck * n + ci (from C, note trace convention W[k*n+i])
                                # Wait, W index is k*n + i.
                                # Here C is C[ci, ck].
                                # So i=ci, k=ck.
                                # W index = ck * n + ci.
                                
                                u_idx = gi * m + gj
                                v_idx = gj * p + gk
                                w_idx = ck * n + ci
                                
                                U[u_idx][col] += total_coeff
                                V[v_idx][col] += total_coeff
                                W[w_idx][col] += total_coeff
                                
                    col += 1
                    
    # Check if col == rank
    # It should be 7 * 27 = 189.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
