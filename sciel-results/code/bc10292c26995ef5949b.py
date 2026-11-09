"""Program for 6x6 matrix multiplication rank minimization."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-189 decomposition for 6x6 matrix multiplication.
    Uses tiling into 2x2 blocks of 3x3 matrices.
    Applies Strassen's algorithm (rank 7) to the 2x2 block structure.
    Uses schoolbook multiplication (rank 27) for the 3x3 blocks.
    Total rank = 7 * 27 = 189.
    """
    n, m, p = DIMENSIONS
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We represent the 6x6 matrix as 2x2 blocks of 3x3 matrices.
    # A = [[A11, A12], [A21, A22]] where each Aij is 3x3.
    # B = [[B11, B12], [B21, B22]] where each Bij is 3x3.
    # C = A * B.
    
    # The multiplication of two 3x3 matrices is done via schoolbook (27 multiplications).
    # Each of the 7 Strassen products involves multiplying two 3x3 matrices.
    # So we need 7 * 27 = 189 scalar multiplications.
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a schoolbook decomposition for a 3x3 block multiplication
    # into the global U, V, W matrices at a specific offset.
    # The 3x3 block corresponds to indices:
    # A block (i, j) in the 2x2 block grid:
    #   Rows: i*3 to i*3+2
    #   Cols: j*3 to j*3+2
    # B block (j, k):
    #   Rows: j*3 to j*3+2
    #   Cols: k*3 to k*3+2
    # C block (i, k):
    #   Rows: k*3 to k*3+2 (Wait, W index is k*n + i? No, W is p*n. Index is k*n + i?
    #   Task says W[k*n+i, r].
    #   Standard C[i, k] = sum A[i, j] B[j, k].
    #   Here i, j, k are block indices.
    #   Element (I, K) in C is sum over J of A[I, J] B[J, K].
    #   I is row index in 0..5, K is col index in 0..5.
    #   Block row i = I // 3, Block col k = K // 3.
    #   W index for element (I, K) is K * n + I?
    #   Task: W[k*n+i, r].
    #   If k is col block index and i is row block index?
    #   Let's check the seed program.
    #   Seed: W[k*n+i][column] = 1.
    #   In seed: i is row, j is col of A, k is col of B (row of B).
    #   C[i, k] = A[i, j] * B[j, k].
    #   So i is row index of C, k is col index of C.
    #   W index is k * n + i.
    #   This matches C[k, i] if we view W as flattened C?
    #   Usually C is stored row-major or col-major.
    #   If W index is k*n + i, and n=6, then for k=0, i=0..5, indices 0..5.
    #   For k=1, i=0..5, indices 6..11.
    #   This is column-major storage of C (col k, row i).
    #   So C[i, k] is at W[k*n + i].
    
    def add_schoolbook_3x3(offset_u, offset_v, offset_w, base_col):
        # offset_u: row index in U for the first element of the A block
        # offset_v: row index in V for the first element of the B block
        # offset_w: row index in W for the first element of the C block
        # base_col: starting column index in U, V, W
        
        # A block is 3x3, indices (r, c) -> offset_u + r*3 + c
        # B block is 3x3, indices (r, c) -> offset_v + r*3 + c
        # C block is 3x3, indices (r, c) -> offset_w + c*3 + r  (since W is col-major)
        # Wait, W index is k*n + i.
        # In the block context:
        # C block row i_c (0..2), col k_c (0..2).
        # Global row I = block_row_i * 3 + i_c.
        # Global col K = block_col_k * 3 + k_c.
        # W index = K * 6 + I = (block_col_k * 3 + k_c) * 6 + (block_row_i * 3 + i_c).
        # This is not simply offset_w + k_c*3 + i_c.
        # We need to map the local 3x3 indices to global W indices.
        
        for i_c in range(3):
            for k_c in range(3):
                # Sum over j_c
                for j_c in range(3):
                    # A element: local (i_c, j_c)
                    u_idx = offset_u + i_c * 3 + j_c
                    # B element: local (j_c, k_c)
                    v_idx = offset_v + j_c * 3 + k_c
                    # C element: global (I, K)
                    I = offset_u // 3 * 3 + i_c # This is wrong, offset_u is not block index.
                    
        # Let's use explicit global indices.
        # U index for A[i, j] is i*6 + j.
        # V index for B[j, k] is j*6 + k.
        # W index for C[i, k] is k*6 + i.
        
        pass

    # It is easier to just iterate over all 189 multiplications directly.
    # We define the 7 Strassen products.
    # Each product is a 3x3 matrix multiplication.
    
    # Strassen terms:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # We need to decompose each Mx = Ax * Bx into 27 scalar multiplications.
    # Ax and Bx are linear combinations of the 3x3 blocks.
    # The entries of Ax are sums/differences of entries of A blocks.
    # The entries of Bx are sums/differences of entries of B blocks.
    
    # We will iterate over the 7 Strassen steps.
    # For each step, we iterate over the 27 schoolbook multiplications.
    
    # Block indices:
    # A11: rows 0-2, cols 0-2
    # A12: rows 0-2, cols 3-5
    # A21: rows 3-5, cols 0-2
    # A22: rows 3-5, cols 3-5
    
    # B11: rows 0-2, cols 0-2
    # B12: rows 0-2, cols 3-5
    # B21: rows 3-5, cols 0-2
    # B22: rows 3-5, cols 3-5
    
    # C11: rows 0-2, cols 0-2
    # C12: rows 0-2, cols 3-5
    # C21: rows 3-5, cols 0-2
    # C22: rows 3-5, cols 3-5
    
    # Helper to get global indices for a block
    def get_block_indices(block_id):
        # block_id: 0=A11, 1=A12, 2=A21, 3=A22
        # Returns (row_start, col_start)
        if block_id == 0: return (0, 0)
        if block_id == 1: return (0, 3)
        if block_id == 2: return (3, 0)
        if block_id == 3: return (3, 3)
        
    # We define the 7 Strassen products as lists of (coeff_A, block_A, coeff_B, block_B)
    # But since the blocks are 3x3, we need to handle each element.
    
    # Let's define the linear combinations for each Strassen term.
    # M1: A = A11 + A22, B = B11 + B22
    # M2: A = A21 + A22, B = B11
    # M3: A = A11, B = B12 - B22
    # M4: A = A22, B = B21 - B11
    # M5: A = A11 + A12, B = B22
    # M6: A = A21 - A11, B = B11 + B12
    # M7: A = A12 - A22, B = B21 + B22
    
    # For each term, we perform a 3x3 matrix multiplication.
    # C_local[i, k] = sum_j A_local[i, j] * B_local[j, k]
    
    # We will generate 27 multiplications for each term.
    
    strassen_terms = [
        # (A_blocks, B_blocks)
        # Each block is a tuple (coeff, block_id)
        # A_local = sum(coeff * A_block)
        # B_local = sum(coeff * B_block)
        ([ (1, 0), (1, 3) ], [ (1, 0), (1, 3) ]), # M1
        ([ (1, 2), (1, 3) ], [ (1, 0) ]),          # M2
        ([ (1, 0) ], [ (1, 1), (-1, 3) ]),         # M3
        ([ (1, 3) ], [ (1, 2), (-1, 0) ]),         # M4
        ([ (1, 0), (1, 1) ], [ (1, 3) ]),          # M5
        ([ (1, 2), (-1, 0) ], [ (1, 0), (1, 1) ]), # M6
        ([ (1, 1), (-1, 3) ], [ (1, 2), (1, 3) ]), # M7
    ]
    
    col = 0
    for a_terms, b_terms in strassen_terms:
        # For each element in the 3x3 result
        for i_c in range(3):
            for k_c in range(3):
                for j_c in range(3):
                    # A_local[i_c, j_c] = sum(coeff * A_block[i_c, j_c])
                    # B_local[j_c, k_c] = sum(coeff * B_block[j_c, k_c])
                    
                    # We need to express the product A_local[i_c, j_c] * B_local[j_c, k_c]
                    # as a sum of products of scalar entries of A and B.
                    # (sum ca * Aa) * (sum cb * Bb) = sum ca*cb * Aa*Bb
                    # This would increase the rank.
                    # Instead, we should pre-compute the linear combinations?
                    # No, the rank is the number of scalar multiplications.
                    # If we compute M1 = (A11+A22)(B11+B22), we can do it in 27 multiplications
                    # IF we treat (A11+A22) and (B11+B22) as the operands.
                    # But the operands are 3x3 matrices with entries that are sums of A entries.
                    # To multiply two 3x3 matrices where entries are linear combinations,
                    # we can either:
                    # 1. Expand the products, leading to more terms.
                    # 2. Use the fact that the tensor is bilinear.
                    
                    # The standard Strassen algorithm works on scalars.
                    # Here, the "scalars" are 3x3 matrices.
                    # The multiplication of two 3x3 matrices is bilinear.
                    # So (A11+A22) * (B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22.
                    # This requires 4 * 27 = 108 multiplications?
                    # No, Strassen reduces the number of matrix multiplications from 8 to 7.
                    # Each matrix multiplication is of 3x3 matrices.
                    # So 7 * 27 = 189 scalar multiplications.
                    # This implies that we compute the 7 products of 3x3 matrices.
                    # The entries of the resulting 3x3 matrices are sums of products of entries.
                    
                    # So for each Strassen term, we perform a full 3x3 matrix multiplication.
                    # The inputs to this multiplication are the linear combinations.
                    # However, we cannot "add" A11 and A22 without using additional operations?
                    # No, the additions are free (not counted in rank).
                    # So we form the matrices A_M1 = A11 + A22 and B_M1 = B11 + B22.
                    # Then we multiply A_M1 * B_M1.
                    # This multiplication takes 27 scalar multiplications.
                    
                    # So for each of the 7 terms, we do 27 scalar multiplications.
                    
                    # We need to map the local indices (i_c, j_c, k_c) to global indices.
                    
                    # A_M1[i_c, j_c] = A11[i_c, j_c] + A22[i_c, j_c]
                    # This is a linear combination.
                    # The product is (A11[i_c, j_c] + A22[i_c, j_c]) * (B11[j_c, k_c] + B22[j_c, k_c])
                    # = A11[i_c, j_c]*B11[j_c, k_c] + A11[i_c, j_c]*B22[j_c, k_c] + ...
                    
                    # This expansion leads to 4 products per term per element.
                    # Total 7 * 27 * 4? No.
                    # The Strassen algorithm replaces 8 matrix mults with 7.
                    # It does NOT expand the entries.
                    # The entries of the result are computed as sums of the 7 matrix products.
                    # C11 = M1 + M4 - M5 + M7.
                    # The entries of M1 are the entries of the 3x3 matrix product (A11+A22)(B11+B22).
                    # So we compute the 3x3 matrix product, getting 9 entries per Mx.
                    # Total 7 * 9 = 63 entries.
                    # Each entry is a sum of 27 scalar products? No.
                    # Each entry Mx[i_c, k_c] is sum_j A_Mx[i_c, j_c] * B_Mx[j_c, k_c].
                    # This sum has 3 terms.
                    # Each term is a product of two linear combinations.
                    # So Mx[i_c, k_c] is a sum of 3 products of linear combinations.
                    # Each product of linear combinations expands to 4 scalar products.
                    # So Mx[i_c, k_c] is a sum of 12 scalar products.
                    # Total scalar products = 7 * 9 * 12 = 756?
                    # This is worse than schoolbook 216.
                    
                    # The error is in how Strassen is applied to blocks.
                    # Strassen applies to the block structure.
                    # The blocks are 3x3 matrices.
                    # The multiplication of two 3x3 matrices is a bilinear map.
                    # The rank of this map is 27.
                    # So 7 * 27 = 189 is correct.
                    # How?
                    # We compute the 7 products of 3x3 matrices.
                    # Each product is a 3x3 matrix.
                    # The entries of these products are computed using the rank 27 decomposition.
                    # So we need 27 scalar multiplications for each of the 7 products.
                    # Total 189.
                    
                    # The linear combinations A11+A22 are formed by addition (free).
                    # Then we multiply the resulting 3x3 matrices.
                    # The multiplication of two 3x3 matrices uses 27 scalar multiplications.
                    # These scalar multiplications are between the entries of the combined matrices.
                    # But the entries of the combined matrices are sums of entries of A.
                    # So the scalar multiplication is between (A11[i,j] + A22[i,j]) and (B11[j,k] + B22[j,k]).
                    # This product is not a single scalar multiplication in the tensor decomposition sense unless we treat the sum as a single variable?
                    # No, the tensor decomposition is over the field.
                    # The variables are the individual entries A[i,j], B[j,k].
                    # So (A11[i,j] + A22[i,j]) * (B11[j,k] + B22[j,k]) is NOT a rank 1 term.
                    # It is a sum of 4 rank 1 terms.
                    
                    # Therefore, the rank is NOT 189 if we use this naive approach.
                    # The rank 189 result implies that there is a way to do it in 189.
                    # This requires that the linear combinations are handled such that the products are rank 1.
                    # This is only possible if we change the basis?
                    # Or if the decomposition of the 3x3 multiplication is used in a way that respects the linear combinations?
                    
                    # Actually, the rank 189 is achieved by viewing the 6x6 multiplication as a 2x2 multiplication of 3x3 matrices.
                    # The tensor is T_6 = T_2 (on blocks) \otimes T_3 (on elements).
                    # Rank(T_6) <= Rank(T_2) * Rank(T_3) = 7 * 27 = 189.
                    # This submultiplicativity holds.
                    # The construction involves taking the rank 7 decomposition of T_2 and the rank 27 decomposition of T_3.
                    # For each term in T_2 (which is a product of linear forms in blocks), we take the corresponding term in T_3 (product of linear forms in elements).
                    # The product of the two linear forms is a linear form in the 6x6 entries.
                    # So we get 7 * 27 = 189 terms.
                    
                    # So the algorithm is:
                    # 1. Get the 7 terms of Strassen for 2x2.
                    #    Each term is u_t (block linear form) * v_t (block linear form) * w_t (block linear form).
                    # 2. For each term, get the 27 terms of schoolbook for 3x3.
                    #    Each term is u'_s * v'_s * w'_s.
                    # 3. The combined term is (u_t \otimes u'_s) * (v_t \otimes v'_s) * (w_t \otimes w'_s).
                    #    The coefficients are products of the coefficients.
                    
                    # This is the correct construction.
                    
                    pass

    # Implementation of the tensor product of decompositions
    # Strassen 2x2 terms
    # We need the explicit coefficients for the 7 terms.
    # M1: A11+A22, B11+B22, C11+=1, C12+=0, C21+=0, C22+=1 (Wait, C11 = M1+M4-M5+M7)
    # The w vector for M1 has 1 at C11 and C22?
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So for M1:
    # u: A11=1, A22=1
    # v: B11=1, B22=1
    # w: C11=1, C22=1
    
    # We represent u, v, w as vectors of length 4 (blocks).
    # Then we tensor with the 3x3 schoolbook terms.
    
    # Schoolbook 3x3 terms:
    # 27 terms.
    # u': A[i,j]=1
    # v': B[j,k]=1
    # w': C[i,k]=1
    
    # We will generate all 189 terms.
    
    strassen_u = [
        [1, 0, 0, 1], # M1: A11, A22
        [0, 0, 1, 1], # M2: A21, A22
        [1, 0, 0, 0], # M3: A11
        [0, 0, 0, 1], # M4: A22
        [1, 1, 0, 0], # M5: A11, A12
        [0, 0, 1, -1],# M6: A21, -A11
        [0, 1, 0, -1],# M7: A12, -A22
    ]
    strassen_v = [
        [1, 0, 0, 1], # M1: B11, B22
        [1, 0, 0, 0], # M2: B11
        [0, 1, 0, -1],# M3: B12, -B22
        [0, 0, 1, -1],# M4: B21, -B11
        [0, 0, 0, 1], # M5: B22
        [1, 1, 0, 0], # M6: B11, B12
        [0, 0, 1, 1], # M7: B21, B22
    ]
    strassen_w = [
        [1, 0, 0, 1], # M1: C11, C22
        [0, 0, 1, 0], # M2: C21
        [0, 1, 0, 0], # M3: C12
        [1, 0, 1, 0], # M4: C11, C21
        [0, 1, 0, -1],# M5: C12, -C22
        [0, 0, 0, 1], # M6: C22
        [1, 0, 0, 0], # M7: C11
    ]
    
    # Schoolbook 3x3 terms
    # We need the u', v', w' vectors of length 9.
    # u'[i*3+j] = 1
    # v'[j*3+k] = 1
    # w'[k*3+i] = 1 (Col-major)
    
    for t in range(7):
        for i_c in range(3):
            for j_c in range(3):
                for k_c in range(3):
                    # u' index
                    u_idx_3 = i_c * 3 + j_c
                    # v' index
                    v_idx_3 = j_c * 3 + k_c
                    # w' index
                    w_idx_3 = k_c * 3 + i_c
                    
                    # Coefficients
                    # The term is sum over block indices b_a, b_b, b_c
                    # coeff = strassen_u[t][b_a] * strassen_v[t][b_b] * strassen_w[t][b_c]
                    # But we need to map to global indices.
                    
                    # For each non-zero entry in strassen_u[t], etc.
                    for b_a in range(4):
                        if strassen_u[t][b_a] == 0: continue
                        for b_b in range(4):
                            if strassen_v[t][b_b] == 0: continue
                            for b_c in range(4):
                                if strassen_w[t][b_c] == 0: continue
                                
                                coeff = strassen_u[t][b_a] * strassen_v[t][b_b] * strassen_w[t][b_c]
                                if coeff == 0: continue
                                
                                # Global indices
                                # A block b_a:
                                row_start_a, col_start_a = get_block_indices(b_a)
                                u_global = (row_start_a + i_c) * 6 + (col_start_a + j_c)
                                
                                # B block b_b:
                                row_start_b, col_start_b = get_block_indices(b_b)
                                v_global = (row_start_b + j_c) * 6 + (col_start_b + k_c)
                                
                                # C block b_c:
                                row_start_c, col_start_c = get_block_indices(b_c)
                                # W index is k*n + i
                                # Global row I = row_start_c + i_c
                                # Global col K = col_start_c + k_c
                                w_global = K * 6 + I
                                w_global = (col_start_c + k_c) * 6 + (row_start_c + i_c)
                                
                                U[u_global][col] += coeff
                                V[v_global][col] += coeff
                                W[w_global][col] += coeff
                                
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
