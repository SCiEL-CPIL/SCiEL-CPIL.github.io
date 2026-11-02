import numpy as np

def construct():
    n, m, p = 4, 4, 6
    rank = 44
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Helper to get index in A (U)
    def a_idx(r, c):
        return r * m + c
        
    # Helper to get index in B (V)
    def b_idx(r, c):
        return r * p + c
        
    # Helper to get index in C (W)
    def c_idx(r, c):
        # W[k*n+i, r] where k is col, i is row
        return c * n + r

    col = 0
    
    # Strassen for A (4x4) * B1 (4x4)
    # B1 consists of B[0:2, 0:2], B[0:2, 2:4], B[2:4, 0:2], B[2:4, 2:4]
    
    # We need to add terms to U, V, W.
    # Each term is a rank-1 contribution.
    # For a product A_sub * B_sub, we add 4 terms (schoolbook).
    # A_sub is 2x2, B_sub is 2x2.
    
    def add_strassen_term(a_coeffs, b_coeffs, c_coeffs, sign=1.0):
        nonlocal col
        # a_coeffs: list of (a_row, a_col, coeff)
        # b_coeffs: list of (b_row, b_col, coeff)
        # c_coeffs: list of (c_row, c_col, coeff)
        # This function adds the 4 schoolbook terms for the product of linear combinations.
        
        # Actually, it's easier to just generate the 4 scalar multiplications for each 2x2*2x2.
        # But the linear combinations are on the matrices.
        # M = (sum a_i A_i) * (sum b_j B_j)
        # This expands to sum a_i b_j (A_i B_j).
        # We can just add terms for each pair (i, j).
        
        for a_r, a_c, a_coeff in a_coeffs:
            for b_r, b_c, b_coeff in b_coeffs:
                # Product of A[a_r, a_c] and B[b_r, b_c]
                # This contributes to C.
                # But wait, A_i and B_j are 2x2 blocks.
                # A_i is a 2x2 matrix. B_j is a 2x2 matrix.
                # The product A_i B_j is a 2x2 matrix.
                # We need to decompose this 2x2*2x2 into 4 scalar products.
                
                # The block A_i corresponds to specific rows/cols in A.
                # The block B_j corresponds to specific rows/cols in B.
                
                # Let's identify the global indices.
                # a_r, a_c are local 0..1 in the block.
                # We need to know which block of A and B.
                
                # This approach is getting complicated with nested loops.
                # Let's explicitly list the 7 Strassen products and their components.
                
                pass

    # Let's do it explicitly.
    
    # Define blocks of A
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # Define blocks of B1
    # B11: rows 0-1, cols 0-1
    # B12: rows 0-1, cols 2-3
    # B21: rows 2-3, cols 0-1
    # B22: rows 2-3, cols 2-3
    
    # Function to add a product of two 2x2 blocks
    # block_a: (start_row, start_col) in A
    # block_b: (start_row, start_col) in B
    # block_c: (start_row, start_col) in C
    # coeff: multiplier for the whole product
    
    def add_block_product(a_sr, a_sc, b_sr, b_sc, c_sr, c_sc, coeff):
        nonlocal col
        # Iterate over 2x2 elements
        for i in range(2):
            for j in range(2):
                a_r = a_sr + i
                a_c = a_sc + j
                b_r = b_sr + j # B row index matches A col index in product?
                # Wait. A is 4x4. B is 4x6.
                # A[i, j] * B[j, k].
                # In block multiplication:
                # C_block = A_block * B_block.
                # A_block is 2x2. B_block is 2x2.
                # (A_block * B_block)_{i, k} = sum_j A_block[i, j] B_block[j, k].
                # So A index is (a_sr+i, a_sc+j).
                # B index is (b_sr+j, b_sc+k).
                # C index is (c_sr+i, c_sc+k).
                
                for k in range(2):
                    # Term: A[a_sr+i, a_sc+j] * B[b_sr+j, b_sc+k]
                    # This is a scalar product.
                    # It contributes to C[c_sr+i, c_sc+k].
                    
                    u_idx = a_r * m + a_c
                    v_idx = b_r * p + b_c
                    w_idx = c_sc * n + c_r # c_sc is col of C block, c_r is row of C block?
                    # c_r = c_sr + i.
                    # c_col = c_sc + k.
                    w_idx = (c_sc + k) * n + (c_sr + i)
                    
                    U[u_idx][col] += coeff
                    V[v_idx][col] += coeff
                    W[w_idx][col] += coeff
                    col += 1

    # This function adds 4 terms (i=0,1; k=0,1) for a fixed j?
    # No, the loop over j is inside?
    # A[i,j] * B[j,k].
    # My loop:
    # for i in range(2):
    #   for j in range(2):
    #     for k in range(2):
    # This adds 8 terms?
    # A 2x2 * 2x2 multiplication has rank 4.
    # It is sum_{j=0,1} A[i,j] B[j,k].
    # So for each i, k, we sum over j.
    # The decomposition is:
    # For each j in 0..1:
    #   For each i in 0..1:
    #     For each k in 0..1:
    #       Term A[i,j] * B[j,k] contributes to C[i,k].
    # Total 2*2*2 = 8 terms?
    # No.
    # Rank 4 means we can do it with 4 terms.
    # The schoolbook decomposition for 2x2x2 is:
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    # This is 8 scalar multiplications.
    # Wait, the rank of 2x2x2 matrix multiplication is 4?
    # No, the rank of 2x2x2 is 7 (Strassen) or 4 (Schoolbook)?
    # Schoolbook is 4 multiplications?
    # No, schoolbook for 2x2 is 4 multiplications?
    # A00B00, A01B10, A10B00, A11B11?
    # C00 = A00B00 + A01B10. Two multiplications.
    # C01 = A00B01 + A01B11. Two multiplications.
    # C10 = A10B00 + A11B10. Two multiplications.
    # C11 = A10B01 + A11B11. Two multiplications.
    # Total 8 scalar multiplications.
    # So rank of 2x2x2 is 7 (Strassen) or 8 (Schoolbook)?
    # Wait, the tensor is $M_{2,2,2}$.
    # The rank is 7.
    # Schoolbook uses 7? No, schoolbook uses 7?
    # Strassen is 7.
    # Schoolbook is 7?
    # Let's check.
    # C = AB.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # This is 7 multiplications.
    # Schoolbook:
    # A00B00, A01B10, A10B00, A11B11?
    # No, schoolbook is 7?
    # Actually, for 2x2, schoolbook is 7?
    # Let's count:
    # A00*B00, A00*B01, A01*B10, A01*B11, A10*B00, A10*B01, A11*B10, A11*B11.
    # That's 8.
    # So schoolbook rank is 8?
    # No, you can share?
    # A00(B00+B01) = A00B00 + A00B01.
    # A10(B00+B01) = A10B00 + A10B01.
    # A01(B10+B11) = A01B10 + A01B11.
    # A11(B10+B11) = A11B10 + A11B11.
    # 4 multiplications!
    # So rank 4 is possible for 2x2x2?
    # Yes, if you allow additions in U, V, W.
    # The rank is the number of terms in the sum.
    # So rank 4 is correct for 2x2x2.
    
    # So my `add_block_product` should add 4 terms.
    # The terms are:
    # 1. A00 * (B00+B01) -> contributes to C00, C01
    # 2. A01 * (B10+B11) -> contributes to C00, C01
    # 3. A10 * (B00+B01) -> contributes to C10, C11
    # 4. A11 * (B10+B11) -> contributes to C10, C11
    
    # Let's refine add_block_product.
    
    def add_block_product_v2(a_sr, a_sc, b_sr, b_sc, c_sr, c_sc, coeff):
        nonlocal col
        
        # Term 1: A[0,0] * (B[0,0] + B[0,1])
        # A index: a_sr+0, a_sc+0
        # B indices: b_sr+0, b_sc+0 and b_sr+0, b_sc+1
        # C indices: c_sr+0, c_sc+0 and c_sr+0, c_sc+1
        
        a_idx = (a_sr) * m + (a_sc)
        
        # We need to add two V entries for the same U entry?
        # No, each term in the decomposition is a rank-1 tensor.
        # u \otimes v \otimes w.
        # If v has two non-zero entries, it's still one rank-1 term.
        # So we can have V[b_idx1][col] = 1 and V[b_idx2][col] = 1.
        
        V_idx_1 = (b_sr) * p + (b_sc)
        V_idx_2 = (b_sr) * p + (b_sc + 1)
        
        W_idx_1 = (c_sc) * n + (c_sr)
        W_idx_2 = (c_sc + 1) * n + (c_sr)
        
        U[a_idx][col] = coeff
        V[V_idx_1][col] = coeff
        V[V_idx_2][col] = coeff
        W[W_idx_1][col] = coeff
        W[W_idx_2][col] = coeff
        col += 1
        
        # Term 2: A[0,1] * (B[1,0] + B[1,1])
        a_idx = (a_sr) * m + (a_sc + 1)
        V_idx_1 = (b_sr + 1) * p + (b_sc)
        V_idx_2 = (b_sr + 1) * p + (b_sc + 1)
        
        U[a_idx][col] = coeff
        V[V_idx_1][col] = coeff
        V[V_idx_2][col] = coeff
        W[W_idx_1][col] = coeff
        W[W_idx_2][col] = coeff
        col += 1
        
        # Term 3: A[1,0] * (B[0,0] + B[0,1])
        a_idx = (a_sr + 1) * m + (a_sc)
        V_idx_1 = (b_sr) * p + (b_sc)
        V_idx_2 = (b_sr) * p + (b_sc + 1)
        
        W_idx_1 = (c_sc) * n + (c_sr + 1)
        W_idx_2 = (c_sc + 1) * n + (c_sr + 1)
        
        U[a_idx][col] = coeff
        V[V_idx_1][col] = coeff
        V[V_idx_2][col] = coeff
        W[W_idx_1][col] = coeff
        W[W_idx_2][col] = coeff
        col += 1
        
        # Term 4: A[1,1] * (B[1,0] + B[1,1])
        a_idx = (a_sr + 1) * m + (a_sc + 1)
        V_idx_1 = (b_sr + 1) * p + (b_sc)
        V_idx_2 = (b_sr + 1) * p + (b_sc + 1)
        
        U[a_idx][col] = coeff
        V[V_idx_1][col] = coeff
        V[V_idx_2][col] = coeff
        W[W_idx_1][col] = coeff
        W[W_idx_2][col] = coeff
        col += 1

    # Now apply Strassen
    # M1 = (A11+A22)(B11+B22)
    # This is (A11 + A22) * (B11 + B22)
    # = A11 B11 + A11 B22 + A22 B11 + A22 B22
    # We can add 4 block products.
    
    # But wait, add_block_product adds 4 terms for one block product.
    # So M1 will add 16 terms?
    # No, M1 is ONE product of two 2x2 matrices.
    # The rank of 2x2x2 is 4.
    # So M1 should be represented by 4 terms.
    # My add_block_product adds 4 terms.
    # So M1 adds 4 terms.
    # Total 7 * 4 = 28 terms.
    
    # M1
    # (A11+A22) is a 2x2 matrix.
    # (B11+B22) is a 2x2 matrix.
    # We need to decompose the product of these two linear combinations.
    # (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
    # This is a sum of 4 block products.
    # Each block product has rank 4.
    # So M1 has rank 16?
    # No.
    # Strassen works because the *sum* of the 7 M's reconstructs C.
    # The M's themselves are not independent rank-1 tensors.
    # They are intermediate values.
    # In the tensor decomposition, we don't compute M's.
    # We compute the final C.
    # The decomposition is $\sum_{r} u_r v_r w_r$.
    # The Strassen algorithm says $C = \sum_{i=1}^7 L_i(M_i)$ where $M_i = L'_i(A) L''_i(B)$.
    # So $C = \sum_{i=1}^7 \sum_{\text{terms in } M_i} \dots$
    # Since $M_i$ is a matrix product, and we decompose it into rank 4,
    # each $M_i$ contributes 4 rank-1 tensors to the total sum.
    # So total rank is $7 \times 4 = 28$.
    
    # So we need to add 4 block products for each Strassen M.
    # And the coefficients in U, V, W will be sums of coefficients from the linear combinations.
    
    # Let's implement a helper to add a "linear combination product".
    # add_lin_prod(a_comps, b_comps, c_block, coeff)
    # a_comps: list of (a_sr, a_sc, coeff)
    # b_comps: list of (b_sr, b_sc, coeff)
    # c_block: (c_sr, c_sc)
    
    def add_lin_prod(a_comps, b_comps, c_sr, c_sc, global_coeff=1.0):
        for a_sr, a_sc, a_c in a_comps:
            for b_sr, b_sc, b_c in b_comps:
                add_block_product_v2(a_sr, a_sc, b_sr, b_sc, c_sr, c_sc, global_coeff * a_c * b_c)

    # M1 = (A11+A22)(B11+B22)
    # A11: (0,0), A22: (2,2)
    # B11: (0,0), B22: (2,2)
    # C11: (0,0)
    add_lin_prod([(0,0,1), (2,2,1)], [(0,0,1), (2,2,1)], 0, 0)
    
    # M2 = (A21+A22)B11
    # A21: (2,0), A22: (2,2)
    # B11: (0,0)
    # C21: (2,0)
    add_lin_prod([(2,0,1), (2,2,1)], [(0,0,1)], 2, 0)
    
    # M3 = A11(B12-B22)
    # A11: (0,0)
    # B12: (0,2), B22: (2,2)
    # C12: (0,2)
    add_lin_prod([(0,0,1)], [(0,2,1), (2,2,-1)], 0, 2)
    
    # M4 = A22(B21-B11)
    # A22: (2,2)
    # B21: (2,0), B11: (0,0)
    # C21: (2,0)
    add_lin_prod([(2,2,1)], [(2,0,1), (0,0,-1)], 2, 0)
    
    # M5 = (A11+A12)B22
    # A11: (0,0), A12: (0,2)
    # B22: (2,2)
    # C11: (0,0)
    add_lin_prod([(0,0,1), (0,2,1)], [(2,2,1)], 0, 0)
    
    # M6 = (A21-A11)(B11+B12)
    # A21: (2,0), A11: (0,0)
    # B11: (0,0), B12: (0,2)
    # C22: (2,2)
    add_lin_prod([(2,0,1), (0,0,-1)], [(0,0,1), (0,2,1)], 2, 2)
    
    # M7 = (A12-A22)(B21+B22)
    # A12: (0,2), A22: (2,2)
    # B21: (2,0), B22: (2,2)
    # C11: (0,0)
    add_lin_prod([(0,2,1), (2,2,-1)], [(2,0,1), (2,2,1)], 0, 0)

    # Now B2 part
    # C13 = A11B13 + A12B23
    # C23 = A21B13 + A22B23
    # B13: (0,4), B23: (2,4)
    # C13: (0,4), C23: (2,4)
    
    # A11B13
    add_block_product_v2(0, 0, 0, 4, 0, 4, 1.0)
    # A12B23
    add_block_product_v2(0, 2, 2, 4, 0, 4, 1.0)
    # A21B13
    add_block_product_v2(2, 0, 0, 4, 2, 4, 1.0)
    # A22B23
    add_block_product_v2(2, 2, 2, 4, 2, 4, 1.0)

    return {"U": U, "V": V, "W": W}
