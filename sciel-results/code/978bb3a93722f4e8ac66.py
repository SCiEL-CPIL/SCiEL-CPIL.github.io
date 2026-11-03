import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = 145.0 # 150s limit, leave 5s margin
    
    n, m, p = DIMENSIONS
    # We will collect terms as (a_idx, b_idx, c_idx, coeff)
    # a_idx in [0, 15], b_idx in [0, 35], c_idx in [0, 35]
    # Note: C is 4x9. b is 4x9.
    # For the 4x4x4 part, b_idx is in [0, 15] (cols 0-3) and [16, 31] (cols 4-7).
    # Wait, B is 4x9. Flattened index for B[j, k] is j*9 + k.
    # For B_1 (cols 0-3), b_idx = j*9 + k.
    # For B_2 (cols 4-7), b_idx = j*9 + (k+4).
    # For B_3 (col 8), b_idx = j*9 + 8.
    
    terms = []
    
    def add_term(a_idx, b_idx, c_idx, coeff):
        if coeff == 0:
            return
        terms.append((a_idx, b_idx, c_idx, coeff))

    # Function to add terms for 4x4x4 multiplication using Strassen
    # A is 4x4, B is 4x4 (subset of B), C is 4x4 (subset of C)
    # a_base: base index for A (0)
    # b_base: base index for B (0 or 16)
    # c_base: base index for C (0 or 16)
    def strassen_4x4x4_terms(b_base):
        # Strassen steps for 4x4 using 2x2 blocks
        # A blocks: A00 (0-3), A01 (4-7), A10 (8-11), A11 (12-15)
        # B blocks: B00 (b_base + 0-3), B01 (b_base + 4-7), B10 (b_base + 8-11), B11 (b_base + 12-15)
        
        # Helper to get block indices
        def get_block_indices(base, block_row, block_col):
            # Block is 2x2
            # Rows: base_row*2 to base_row*2+1
            # Cols: base_col*2 to base_col*2+1
            start_row = block_row * 2
            start_col = block_col * 2
            indices = []
            for r in range(2):
                for c in range(2):
                    idx = (start_row + r) * 4 + (start_col + c)
                    indices.append(idx)
            return indices

        A_00 = get_block_indices(0, 0, 0)
        A_01 = get_block_indices(0, 0, 1)
        A_10 = get_block_indices(0, 1, 0)
        A_11 = get_block_indices(0, 1, 1)

        B_00 = get_block_indices(b_base, 0, 0)
        B_01 = get_block_indices(b_base, 0, 1)
        B_10 = get_block_indices(b_base, 1, 0)
        B_11 = get_block_indices(b_base, 1, 1)

        # C blocks
        C_00 = get_block_indices(0, 0, 0) # c_base is 0 for C in 4x4x4 context? 
        # Wait, C is 4x9. For the first 4x4x4, C cols 0-3.
        # c_idx = c_base + local_idx.
        # In this function, we assume C is 4x4. The caller handles the offset?
        # No, I'll pass c_base.
        
        # Actually, let's just use local indices 0-15 for C inside this function, 
        # and the caller will add the offset?
        # No, the term function `add_term` needs global indices.
        # So I need to map local C indices to global C indices.
        # Global C index for 4x9 matrix: c_row * 9 + c_col.
        # Local C index for 4x4 matrix: l_row * 4 + l_col.
        # Mapping: c_row = l_row, c_col = c_base_col + l_col.
        # So global_c = l_row * 9 + (c_base_col + l_col).
        
        def add_local_c_term(a_idx, b_idx, local_c_idx, coeff):
            l_row = local_c_idx // 4
            l_col = local_c_idx % 4
            global_c_row = l_row
            global_c_col = c_base_col + l_col
            global_c_idx = global_c_row * 9 + global_c_col
            add_term(a_idx, b_idx, global_c_idx, coeff)
            
        # This function needs c_base_col.
        
        # Let's redefine the inner logic to take c_base_col.
        
        def _strassen_inner(c_base_col):
            def add_lca(a_idx, b_idx, l_c_idx, coeff):
                lr = l_c_idx // 4
                lc = l_c_idx % 4
                gc = lr * 9 + (c_base_col + lc)
                add_term(a_idx, b_idx, gc, coeff)

            # Strassen M1..M7
            # Each M is product of two 2x2 matrices.
            # We will expand each 2x2 product using schoolbook (8 terms).
            
            def process_product(a_lin_comb, b_lin_comb, c_block_row, c_block_col, sign):
                # a_lin_comb: list of (coeff, a_idx)
                # b_lin_comb: list of (coeff, b_idx)
                # c_block: which 2x2 block of C this product contributes to?
                # No, the product Z is 2x2. Each entry Z_pq contributes to C.
                # In Strassen, M1..M7 are 2x2 matrices.
                # We sum them with signs to get C blocks.
                # Here, we process one 2x2 product Z.
                # Z_pq = sum_{i,j} X_pi Y_jq.
                # We need to expand X_pi and Y_jq into sums of A and B elements.
                
                # X is 2x2. X_pi is a linear combination of A elements.
                # Y is 2x2. Y_jq is a linear combination of B elements.
                
                # We have a_lin_comb and b_lin_comb for the WHOLE 2x2 matrices?
                # No, Strassen defines X and Y as 2x2 matrices where each entry is a linear combo.
                # So we need 4 linear combos for X and 4 for Y.
                # Let's change the interface.
                
                pass
            
            # This is getting complicated to code generically.
            # I will hardcode the 7 Strassen steps for 4x4x4.
            
            # M1 = (A00 + A11)(B00 + B11)
            # M2 = (A10 + A11)B00
            # M3 = A00(B01 - B11)
            # M4 = A11(B10 - B00)
            # M5 = (A00 + A01)B11
            # M6 = (A10 - A00)(B00 + B01)
            # M7 = (A01 - A11)(B10 + B11)
            
            # Each M is 2x2.
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # I will define a function to add terms for a 2x2 product of linear combos.
            def add_2x2_product_terms(x_coeffs, y_coeffs, c_row_base, c_col_base, global_sign):
                # x_coeffs: 2x2 list of linear combos (list of (coeff, a_idx))
                # y_coeffs: 2x2 list of linear combos (list of (coeff, b_idx))
                # c_row_base, c_col_base: top-left of the 2x2 block in C
                for p in range(2):
                    for q in range(2):
                        # Z_pq = sum_{i in 0,1} X_pi * Y_iq
                        for i in range(2):
                            x_lin = x_coeffs[p][i] # list of (coeff, a_idx)
                            y_lin = y_coeffs[i][q] # list of (coeff, b_idx)
                            # Expand product of two linear combos
                            for ca, ia in x_lin:
                                for cb, ib in y_lin:
                                    coeff = ca * cb * global_sign
                                    # local c index
                                    lc_row = c_row_base + p
                                    lc_col = c_col_base + q
                                    lc_idx = lc_row * 4 + lc_col
                                    add_lca(ia, ib, lc_idx, coeff)

            # Define blocks as linear combos of A and B
            # A00: [(1, idx) for idx in A_00]
            A00_l = [(1, i) for i in A_00]
            A01_l = [(1, i) for i in A_01]
            A10_l = [(1, i) for i in A_10]
            A11_l = [(1, i) for i in A_11]
            
            B00_l = [(1, i) for i in B_00]
            B01_l = [(1, i) for i in B_01]
            B10_l = [(1, i) for i in B_10]
            B11_l = [(1, i) for i in B_11]
            
            # Combine blocks
            def add_lcomb(l1, l2):
                return l1 + l2
            def sub_lcomb(l1, l2):
                return l1 + [(-c, i) for c, i in l2]
            
            # X and Y for each M are 2x2 matrices of linear combos.
            # M1: X = [[A00, 0], [0, A11]]? No.
            # M1 = (A00 + A11)(B00 + B11).
            # This is product of two 2x2 matrices? No, A00 is 2x2.
            # So X = [[A00+A11, 0], [0, A00+A11]]? No.
            # In Strassen, the "matrices" are the blocks themselves.
            # M1 = (A00 + A11) * (B00 + B11).
            # Here (A00 + A11) is a 2x2 matrix.
            # So X is a 2x2 matrix where X_00 = A00_00 + A11_00, etc.
            # So X_coeffs[p][q] is the linear combo for the (p,q) entry of the X matrix.
            
            def make_2x2_matrix_from_blocks(b1, b2):
                # b1, b2 are lists of indices for 2x2 blocks.
                # We need to combine them entry-wise.
                # b1[0] + b2[0], b1[1] + b2[1], etc.
                # But b1 and b2 are lists of (coeff, idx) for the whole block.
                # We need to map block indices to local (p,q) indices.
                # This is tricky because A_00 is a list of 4 indices.
                # Let's assume A_00[0] is (0,0), A_00[1] is (0,1), A_00[2] is (1,0), A_00[3] is (1,1).
                # My get_block_indices returns row-major.
                # So A_00[0] is (0,0), A_00[1] is (0,1), A_00[2] is (1,0), A_00[3] is (1,1).
                
                # We need to create a 2x2 list of linear combos.
                # X_coeffs[0][0] should contain terms for A_00[0] and A_11[0].
                
                def block_to_2x2_lcomb(block_indices):
                    res = [[[], []], [[], []]]
                    # block_indices[0] -> (0,0)
                    res[0][0].append((1, block_indices[0]))
                    res[0][1].append((1, block_indices[1]))
                    res[1][0].append((1, block_indices[2]))
                    res[1][1].append((1, block_indices[3]))
                    return res

                # Sum two 2x2 matrices of linear combos
                def sum_2x2(M1, M2):
                    res = [[[], []], [[], []]]
                    for r in range(2):
                        for c in range(2):
                            res[r][c] = M1[r][c] + M2[r][c]
                    return res

                # Difference
                def diff_2x2(M1, M2):
                    res = [[[], []], [[], []]]
                    for r in range(2):
                        for c in range(2):
                            res[r][c] = M1[r][c] + [(-c_, i) for c_, i in M2[r][c]]
                    return res

                M1_mat = block_to_2x2_lcomb(b1)
                M2_mat = block_to_2x2_lcomb(b2)
                return sum_2x2(M1_mat, M2_mat)
            
            # Define the 7 M's
            # M1 = (A00+A11)(B00+B11)
            X1 = make_2x2_matrix_from_blocks(A_00, A_11)
            Y1 = make_2x2_matrix_from_blocks(B_00, B_11)
            
            # M2 = (A10+A11)B00
            X2 = make_2x2_matrix_from_blocks(A_10, A_11)
            Y2 = block_to_2x2_lcomb(B_00)
            
            # M3 = A00(B01-B11)
            X3 = block_to_2x2_lcomb(A_00)
            Y3 = diff_2x2(block_to_2x2_lcomb(B_01), block_to_2x2_lcomb(B_11))
            
            # M4 = A11(B10-B00)
            X4 = block_to_2x2_lcomb(A_11)
            Y4 = diff_2x2(block_to_2x2_lcomb(B_10), block_to_2x2_lcomb(B_00))
            
            # M5 = (A00+A01)B11
            X5 = make_2x2_matrix_from_blocks(A_00, A_01)
            Y5 = block_to_2x2_lcomb(B_11)
            
            # M6 = (A10-A00)(B00+B01)
            X6 = diff_2x2(block_to_2x2_lcomb(A_10), block_to_2x2_lcomb(A_00))
            Y6 = make_2x2_matrix_from_blocks(B_00, B_01)
            
            # M7 = (A01-A11)(B10+B11)
            X7 = diff_2x2(block_to_2x2_lcomb(A_01), block_to_2x2_lcomb(A_11))
            Y7 = make_2x2_matrix_from_blocks(B_10, B_11)
            
            # Add terms for C
            # C00 = M1 + M4 - M5 + M7
            add_2x2_product_terms(X1, Y1, 0, 0, 1)
            add_2x2_product_terms(X4, Y4, 0, 0, 1)
            add_2x2_product_terms(X5, Y5, 0, 0, -1)
            add_2x2_product_terms(X7, Y7, 0, 0, 1)
            
            # C01 = M3 + M5
            add_2x2_product_terms(X3, Y3, 0, 2, 1) # c_col_base 2? No, 2x2 block C01 is cols 2-3.
            # Wait, C00 is cols 0-1, C01 is cols 2-3.
            # So c_col_base for C01 is 2.
            add_2x2_product_terms(X5, Y5, 0, 2, 1)
            
            # C10 = M2 + M4
            # C10 is rows 2-3, cols 0-1.
            add_2x2_product_terms(X2, Y2, 2, 0, 1)
            add_2x2_product_terms(X4, Y4, 2, 0, 1)
            
            # C11 = M1 - M2 + M3 + M6
            # C11 is rows 2-3, cols 2-3.
            add_2x2_product_terms(X1, Y1, 2, 2, 1)
            add_2x2_product_terms(X2, Y2, 2, 2, -1)
            add_2x2_product_terms(X3, Y3, 2, 2, 1)
            add_2x2_product_terms(X6, Y6, 2, 2, 1)

        _strassen_inner(0) # for B cols 0-3
        if time.time() > deadline: return {"U": [], "V": [], "W": []}
        _strassen_inner(4) # for B cols 4-7
        if time.time() > deadline: return {"U": [], "V": [], "W": []}
        
        # B cols 8 (1 col)
        # Schoolbook 4x4x1
        for i in range(4):
            for j in range(4):
                for k in range(1):
                    a_idx = i * 4 + j
                    b_idx = j * 9 + (8 + k)
                    c_idx = k * 4 + i # C is 4x9. k is 0. c_row=0, c_col=i.
                    # Wait, C is 4x9.
                    # C[i, k] = sum_j A[i, j] B[j, k].
                    # c_idx = i * 9 + k.
                    c_idx = i * 9 + (8 + k)
                    add_term(a_idx, b_idx, c_idx, 1)
        
        # Now convert terms to U, V, W
        # Merge terms with same (a_idx, b_idx, c_idx)
        term_dict = {}
        for a, b, c, coeff in terms:
            key = (a, b, c)
            if key in term_dict:
                term_dict[key] += coeff
            else:
                term_dict[key] = coeff
        
        # Remove zero terms
        final_terms = [(a, b, c, coeff) for (a, b, c), coeff in term_dict.items() if coeff != 0]
        
        rank = len(final_terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for r, (a, b, c, coeff) in enumerate(final_terms):
            U[a][r] = coeff
            V[b][r] = coeff
            W[c][r] = coeff # Wait, W index is p*n. c is index in C (4x9).
            # W shape is (p*n, rank) = (36, rank).
            # c is index in C (36).
            # This is correct.
            
        return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print("Rank:", len(res["U"][0]))
