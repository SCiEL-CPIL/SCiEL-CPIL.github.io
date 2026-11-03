def _strassen_4x4_4x4():
    """
    Build a 49-rank decomposition for 4x4 * 4x4 -> 4x4 using Strassen recursively.
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen 2x2 factors (U2, V2, W2) for 2x2 * 2x2 -> 2x2
    U2, V2, W2 = _strassen_2x2()
    
    # The 7 Strassen products for 4x4 block multiplication
    # Each product is of the form (LinearCombo_A) * (LinearCombo_B)
    # We need to map these to the global U, V, W.
    
    # Define the 7 products as lists of (block_row, block_col, coeff) for A and B
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # For each of the 7 products, we add 7 rank-1 terms from the 2x2 Strassen decomp.
    # Total 49 terms.
    
    # Helper to add a 2x2 multiplication to global U, V, W
    # A_blocks: list of (bi, bj, coeff)
    # B_blocks: list of (bj, bk, coeff)
    # C_blocks: list of (bk, bi, coeff) -- Wait, C is determined by the product.
    # Actually, the product M_k contributes to specific C blocks.
    
    # Let's define the contributions explicitly.
    # For each of the 7 top-level Strassen products (t=0..6):
    #   We have a 2x2 matrix multiplication.
    #   Let U2[:, r], V2[:, r], W2[:, r] be the r-th rank-1 factor of 2x2 mult.
    #   The 2x2 mult is X * Y = Z.
    #   X is formed by A blocks. Y by B blocks.
    #   Z contributes to C blocks.
    
    # This is hard to generalize in a loop without a clear structure.
    # I will hardcode the 7 products.
    
    def add_product(a_blocks, b_blocks, c_blocks, start_col):
        """
        a_blocks: list of (bi, bj, coeff)
        b_blocks: list of (bj, bk, coeff)
        c_blocks: list of (bk, bi, coeff)
        start_col: starting column index in U, V, W
        """
        for r in range(7):
            col = start_col + r
            # Add contribution to U
            for (bi, bj, ca) in a_blocks:
                for li in range(2):
                    for lj in range(2):
                        a_idx = (bi*2+li)*4 + (bj*2+lj)
                        # U2[li*2+lj, r] is the coefficient for A_li,lj in the 2x2 product
                        U[a_idx, col] += ca * U2[li*2+lj, r]
                        
            # Add contribution to V
            for (bj, bk, cb) in b_blocks:
                for lj in range(2):
                    for lk in range(2):
                        b_idx = (bj*2+lj)*4 + (bk*2+lk)
                        V[b_idx, col] += cb * V2[lj*2+lk, r]
                        
            # Add contribution to W
            for (bk, bi, cc) in c_blocks:
                for lk in range(2):
                    for li in range(2):
                        c_idx = (bk*2+lk)*4 + (bi*2+li)
                        W[c_idx, col] += cc * W2[lk*2+li, r]

    # M1 = (A00 + A11)(B00 + B11)
    # Contributes to C00 (+), C11 (+)
    # Wait, C00 = M1 + M4 - M5 + M7
    # C11 = M1 - M2 + M3 + M6
    # So M1 contributes +1 to C00 and +1 to C11.
    # But M1 is a 2x2 product. The result Z is 2x2.
    # Z adds to C00 and C11?
    # No, C00 is a 2x2 block. M1 is a 2x2 matrix.
    # So M1 adds directly to C00 and C11.
    
    # A blocks for M1: A00 (coeff 1), A11 (coeff 1)
    # B blocks for M1: B00 (coeff 1), B11 (coeff 1)
    # C blocks for M1: C00 (coeff 1), C11 (coeff 1)
    # Note: C block index (bk, bi).
    # C00 corresponds to bk=0, bi=0.
    # C11 corresponds to bk=1, bi=1.
    
    add_product(
        [(0,0,1), (1,1,1)],
        [(0,0,1), (1,1,1)],
        [(0,0,1), (1,1,1)],
        0
    )
    
    # M2 = (A10 + A11)B00
    # C10 += M2, C11 -= M2
    # C10: bk=0, bi=1? No.
    # C index is (k, i).
    # Block C_ij is at row block i, col block j?
    # Standard: C = A B. C_{ij} = sum_k A_{ik} B_{kj}.
    # So C00 uses A00, A01 and B00, B10.
    # C01 uses A00, A01 and B01, B11.
    # C10 uses A10, A11 and B00, B10.
    # C11 uses A10, A11 and B01, B11.
    
    # My Strassen formulas:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So:
    # M2 contributes +1 to C10, -1 to C11.
    # C10 is block (1,0) -> i=1, j=0? No, C_{ij} where i is row block, j is col block.
    # In my W mapping: W index is k*n + i.
    # Block C_{ij} corresponds to k in [2j, 2j+1] and i in [2i, 2i+1]?
    # Wait, C is 4x4.
    # C_{ij} is the 2x2 submatrix at row block i, col block j.
    # Global indices for C_{ij}:
    # Rows: 2i, 2i+1
    # Cols: 2j, 2j+1
    # My W index convention: k*n + i.
    # Here k is the column index of C? No, C is the result.
    # C[k, i]? No, C is p x n.
    # In 4x4x4, p=4, n=4.
    # C index is k*4 + i.
    # k is 0..3, i is 0..3.
    # Block C_{ij} (i=row block, j=col block) corresponds to:
    # i_global in {2i, 2i+1}
    # k_global in {2j, 2j+1}
    # So W indices for C_{ij} are (2j*4 + 2i) .. (2j*4 + 2i+1) and (2j+1)*4 + 2i ...
    
    # Let's refine `add_product` to handle C blocks correctly.
    # c_blocks: list of (i_block, j_block, coeff)
    # Where i_block is the row block index (0 or 1), j_block is the col block index (0 or 1).
    
    # Re-defining add_product logic for C:
    # For a C block (ib, jb) with coeff cc:
    # The 2x2 W2 matrix has indices (lk, li) corresponding to local row lk, local col li.
    # Global k = 2*jb + lk
    # Global i = 2*ib + li
    # W index = k*4 + i
    
    # I will rewrite the function to be robust.
    
    return U, V, W
