"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 4x4 matrix multiplication using
    Strassen-like recursive decomposition.
    
    The approach uses the standard Strassen 7-product algorithm recursively:
    - Split 4x4 matrices into 2x2 blocks
    - Apply Strassen's 7 multiplications on 2x2 blocks
    - This gives rank 7 * (rank of 2x2 multiplication)
    
    For 2x2 matrices, we use the base case of Strassen (7 products).
    
    Total rank: 7 * 7 = 49 (vs schoolbook 64)
    
    Actually, let's think more carefully. The tensor for multiplying n×m by m×p
    has rank at most n*m*p (schoolbook). Strassen's algorithm shows that
    matrix multiplication can be done with O(n^log2(7)) multiplications.
    
    For 4x4: we can use the recursive Strassen approach:
    - Split into four 2x2 submatrices
    - Compute 7 products of 2x2 matrices
    - Each 2x2 product can be done with 7 scalar multiplications (Strassen base)
    
    So total rank = 7 * 7 = 49.
    
    However, we need to be careful about the exact indices and the trace convention
    W[k*n+i, r] for the output.
    
    Let's implement this properly.
    """
    n, m, p = DIMENSIONS  # All are 4
    
    # We'll build the decomposition using recursive Strassen.
    # The key insight: a rank-1 term corresponds to one scalar multiplication.
    # Strassen's algorithm expresses C = A*B as a sum of products of linear
    # combinations of entries of A and B.
    
    # For 4x4 matrices, split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bj is 2x2
    # C = [[C11, C12], [C21, C22]] where each Cij is 2x2
    
    # Strassen's 7 products for the 2x2 block matrix multiplication:
    # P1 = (A11 + A22) * (B11 + B22)
    # P2 = (A21 + A22) * B11
    # P3 = A11 * (B12 - B22)
    # P4 = A22 * (B21 - B11)
    # P5 = (A11 + A12) * B22
    # P6 = (A21 - A11) * (B11 + B12)
    # P7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Each Pi is a product of two 2x2 matrices.
    # Each 2x2 matrix product can be done with 7 scalar multiplications.
    
    # So we have 7 * 7 = 49 scalar multiplications total.
    
    # Let's define the 2x2 Strassen decomposition first.
    # For multiplying two 2x2 matrices X (rows 0,1; cols 0,1) by Y:
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    #
    # The 7 Strassen products:
    # s0 = (x00 + x11) * (y00 + y11)
    # s1 = (x10 + x11) * y00
    # s2 = x00 * (y01 - y11)
    # s3 = x11 * (y10 - y00)
    # s4 = (x00 + x01) * y11
    # s5 = (x10 - x00) * (y00 + y01)
    # s6 = (x01 - x11) * (y10 + y11)
    #
    # Result Z = X*Y:
    # z00 = s0 + s3 - s4 + s6
    # z01 = s2 + s4
    # z10 = s1 + s3
    # z11 = s0 - s1 + s2 + s5
    
    # Now, for each of the 7 block-level Strassen products, we need to
    # express the two 2x2 matrices as linear combinations of the original
    # 4x4 matrix entries, then apply the 2x2 Strassen decomposition.
    
    # Let's define the linear forms for the 7 block products.
    
    # Block indices: for 4x4 matrix with 2x2 blocks:
    # A11 = rows 0-1, cols 0-1
    # A12 = rows 0-1, cols 2-3
    # A21 = rows 2-3, cols 0-1
    # A22 = rows 2-3, cols 2-3
    #
    # Similarly for B.
    
    # Define linear forms as dicts: {(row, col): coefficient}
    
    def add_forms(f1, f2):
        """Add two linear forms."""
        result = dict(f1)
        for k, v in f2.items():
            result[k] = result.get(k, 0) + v
        # Remove zeros
        return {k: v for k, v in result.items() if v != 0}
    
    def sub_forms(f1, f2):
        """Subtract f2 from f1."""
        result = dict(f1)
        for k, v in f2.items():
            result[k] = result.get(k, 0) - v
        return {k: v for k, v in result.items() if v != 0}
    
    def scale_form(f, s):
        """Scale a linear form by s."""
        return {k: v * s for k, v in f.items() if v * s != 0}
    
    # Define block extraction forms for a 4x4 matrix
    # A block (br, bc) where br, bc in {0, 1} extracts a 2x2 block
    def block_form(matrix_type, br, bc):
        """
        Returns a dict mapping (row, col) in 4x4 to coefficient for extracting
        a specific 2x2 block. matrix_type is 'A' or 'B'.
        For A: rows are 0-3, cols are 0-3.
        Block (br, bc) has rows 2*br to 2*br+1, cols 2*bc to 2*bc+1.
        """
        form = {}
        for dr in range(2):
            for dc in range(2):
                row = 2 * br + dr
                col = 2 * bc + dc
                form[(row, col)] = 1
        return form
    
    # The 7 Strassen linear forms at the block level:
    # Each is a pair (L_A, L_B) where L_A is a linear form on A's entries
    # and L_B is a linear form on B's entries.
    
    # Define block forms
    A11 = block_form('A', 0, 0)
    A12 = block_form('A', 0, 1)
    A21 = block_form('A', 1, 0)
    A22 = block_form('A', 1, 1)
    
    B11 = block_form('B', 0, 0)
    B12 = block_form('B', 0, 1)
    B21 = block_form('B', 1, 0)
    B22 = block_form('B', 1, 1)
    
    # The 7 Strassen products at block level:
    # Each product is (L_A_block, L_B_block) where each is a 2x2 matrix
    # expressed as linear combination of A/B entries.
    
    # But wait - in Strassen, L_A and L_B are 2x2 matrices, not scalars.
    # So each "block" in the Strassen formula is itself a 2x2 matrix.
    
    # Let me redefine: each of the 7 terms involves multiplying two 2x2
    # matrices, where each 2x2 matrix is a linear combination of blocks.
    
    # Term 1: (A11 + A22) * (B11 + B22)
    # Term 2: (A21 + A22) * B11
    # Term 3: A11 * (B12 - B22)
    # Term 4: A22 * (B21 - B11)
    # Term 5: (A11 + A12) * B22
    # Term 6: (A21 - A11) * (B11 + B12)
    # Term 7: (A12 - A22) * (B21 + B22)
    
    # For each term, we have two 2x2 matrices (each a linear form on the
    # 4x4 matrix entries), and we need to multiply them using 2x2 Strassen.
    
    # Let's build all 49 scalar multiplication terms.
    
    # Each term will be: (L_A_4x4, L_B_4x4) -> contributes to C[i,k]
    # where L_A_4x4 is a linear form on A's 16 entries,
    # L_B_4x4 is a linear form on B's 16 entries.
    
    all_terms = []  # List of (L_A, L_B, weight, (i, k))
    
    # Define the 7 block-level terms as pairs of 2x2 matrix linear forms
    # Each 2x2 matrix linear form is a dict from (local_row, local_col) 
    # to a linear form on the 4x4 entries.
    
    def block_linear_form(block1, coeff1, block2=None, coeff2=0):
        """
        Create a 2x2 matrix where each entry is a linear form on 4x4 entries.
        block1, block2 are dicts from (row,col) in 4x4 to coeff.
        Returns a dict from (dr, dc) in {0,1}^2 to linear form.
        """
        result = {}
        for dr in range(2):
            for dc in range(2):
                lf = {}
                if block1:
                    for (r, c), v in block1.items():
                        lf[(r, c)] = lf.get((r, c), 0) + coeff1 * v
                if block2:
                    for (r, c), v in block2.items():
                        lf[(r, c)] = lf.get((r, c), 0) + coeff2 * v
                if lf:
                    result[(dr, dc)] = {k: v for k, v in lf.items() if v != 0}
        return result
    
    # Actually, let me simplify. Each block like A11 is a 2x2 submatrix.
    # When I write (A11 + A22), I mean a 2x2 matrix where entry (r,c) 
    # is A11[r,c] + A22[r,c].
    
    # Let me define each block as a mapping from (dr,dc) to (global_row, global_col)
    def block_entries(br, bc):
        """Returns dict from (dr,dc) to (global_row, global_col) for block (br,bc)."""
        return {(dr, dc): (2*br + dr, 2*bc + dc) for dr in range(2) for dc in range(2)}
    
    A11_e = block_entries(0, 0)
    A12_e = block_entries(0, 1)
    A21_e = block_entries(1, 0)
    A22_e = block_entries(1, 1)
    
    B11_e = block_entries(0, 0)
    B12_e = block_entries(0, 1)
    B21_e = block_entries(1, 0)
    B22_e = block_entries(1, 1)
    
    # For a 2x2 matrix that is a sum/difference of blocks, 
    # entry (dr,dc) is a linear form on the global entries.
    def make_2x2_matrix(*blocks_with_coeffs):
        """
        Create a 2x2 matrix where each entry is a linear form.
        blocks_with_coeffs is a list of (block_entry_dict, coeff)
        Returns dict from (dr,dc) to linear form dict.
        """
        result = {}
        for dr in range(2):
            for dc in range(2):
                lf = {}
                for block_e, coeff in blocks_with_coeffs:
                    if (dr, dc) in block_e:
                        gr, gc = block_e[(dr, dc)]
                        lf[(gr, gc)] = lf.get((gr, gc), 0) + coeff
                if lf:
                    result[(dr, dc)] = {k: v for k, v in lf.items() if v != 0}
        return result
    
    # The 7 block-level Strassen terms:
    # Each is (M_A, M_B) where M_A and M_B are 2x2 matrices (each entry a linear form)
    
    # Term 1: (A11 + A22) * (B11 + B22)
    M_A1 = make_2x2_matrix((A11_e, 1), (A22_e, 1))
    M_B1 = make_2x2_matrix((B11_e, 1), (B22_e, 1))
    
    # Term 2: (A21 + A22) * B11
    M_A2 = make_2x2_matrix((A21_e, 1), (A22_e, 1))
    M_B2 = make_2x2_matrix((B11_e, 1))
    
    # Term 3: A11 * (B12 - B22)
    M_A3 = make_2x2_matrix((A11_e, 1))
    M_B3 = make_2x2_matrix((B12_e, 1), (B22_e, -1))
    
    # Term 4: A22 * (B21 - B11)
    M_A4 = make_2x2_matrix((A22_e, 1))
    M_B4 = make_2x2_matrix((B21_e, 1), (B11_e, -1))
    
    # Term 5: (A11 + A12) * B22
    M_A5 = make_2x2_matrix((A11_e, 1), (A12_e, 1))
    M_B5 = make_2x2_matrix((B22_e, 1))
    
    # Term 6: (A21 - A11) * (B11 + B12)
    M_A6 = make_2x2_matrix((A21_e, 1), (A11_e, -1))
    M_B6 = make_2x2_matrix((B11_e, 1), (B12_e, 1))
    
    # Term 7: (A12 - A22) * (B21 + B22)
    M_A7 = make_2x2_matrix((A12_e, 1), (A22_e, -1))
    M_B7 = make_2x2_matrix((B21_e, 1), (B22_e, 1))
    
    block_terms = [
        (M_A1, M_B1),  # P1
        (M_A2, M_B2),  # P2
        (M_A3, M_B3),  # P3
        (M_A4, M_B4),  # P4
        (M_A5, M_B5),  # P5
        (M_A6, M_B6),  # P6
        (M_A7, M_B7),  # P7
    ]
    
    # For each block term, apply 2x2 Strassen to get 7 scalar products.
    # The 2x2 Strassen products for X (2x2) * Y (2x2):
    # s0 = (X[0,0] + X[1,1]) * (Y[0,0] + Y[1,1])
    # s1 = (X[1,0] + X[1,1]) * Y[0,0]
    # s2 = X[0,0] * (Y[0,1] - Y[1,1])
    # s3 = X[1,1] * (Y[1,0] - Y[0,0])
    # s4 = (X[0,0] + X[0,1]) * Y[1,1]
    # s5 = (X[1,0] - X[0,0]) * (Y[0,0] + Y[0,1])
    # s6 = (X[0,1] - X[1,1]) * (Y[1,0] + Y[1,1])
    #
    # Result Z = X*Y:
    # z00 = s0 + s3 - s4 + s6
    # z01 = s2 + s4
    # z10 = s1 + s3
    # z11 = s0 - s1 + s2 + s5
    
    # For each scalar product, we need to compute the linear forms.
    
    def combine_2x2_entries(M, entries, coeff=1):
        """
        M is a dict from (dr,dc) to linear form.
        entries is a list of (dr, dc).
        coeff is a scalar.
        Returns a linear form (dict from global (r,c) to coeff).
        """
        result = {}
        for (dr, dc) in entries:
            if (dr, dc) in M:
                for (gr, gc), v in M[(dr, dc)].items():
                    result[(gr, gc)] = result.get((gr, gc), 0) + coeff * v
        return {k: v for k, v in result.items() if v != 0}
    
    # Define the 7 Strassen sub-products for 2x2 matrices
    strassen_2x2 = [
        # (entries_A, entries_B, name)
        ([(0,0), (1,1)], [(0,0), (1,1)], 's0'),
        ([(1,0), (1,1)], [(0,0)], 's1'),
        ([(0,0)], [(0,1), (1,1)], 's2'),  # Need special handling for subtraction
        ([(1,1)], [(1,0), (0,0)], 's3'),  # Need special handling
        ([(0,0), (0,1)], [(1,1)], 's4'),
        ([(1,0), (0,0)], [(0,0), (0,1)], 's5'),  # Special
        ([(0,1), (1,1)], [(1,0), (1,1)], 's6'),  # Special
    ]
    
    # Actually, let me handle the subtractions explicitly.
    # s0 = (X00 + X11) * (Y00 + Y11)
    # s1 = (X10 + X11) * Y00
    # s2 = X00 * (Y01 - Y11)
    # s3 = X11 * (Y10 - Y00)
    # s4 = (X00 + X01) * Y11
    # s5 = (X10 - X00) * (Y00 + Y01)
    # s6 = (X01 - X11) * (Y10 + Y11)
    
    def strassen_products(M_A, M_B):
        """
        Given two 2x2 matrices (as dicts from (dr,dc) to linear form),
        return the 7 Strassen scalar products as pairs of linear forms.
        """
        def entry(M, dr, dc):
            return M.get((dr, dc), {})
        
        products = []
        
        # s0 = (X00 + X11) * (Y00 + Y11)
        X_s0 = {**entry(M_A, 0, 0), **entry(M_A, 1, 1)}
        X_s0 = {k: v for k, v in X_s0.items() if v != 0}
        Y_s0 = {**entry(M_B, 0, 0), **entry(M_B, 1, 1)}
        Y_s0 = {k: v for k, v in Y_s0.items() if v != 0}
        products.append((X_s0, Y_s0))
        
        # s1 = (X10 + X11) * Y00
        X_s1 = {**entry(M_A, 1, 0), **entry(M_A, 1, 1)}
        X_s1 = {k: v for k, v in X_s1.items() if v != 0}
        Y_s1 = entry(M_B, 0, 0)
        products.append((X_s1, Y_s1))
        
        # s2 = X00 * (Y01 - Y11)
        X_s2 = entry(M_A, 0, 0)
        Y_s2_raw = {}
        for k, v in entry(M_B, 0, 1).items():
            Y_s2_raw[k] = Y_s2_raw.get(k, 0) + v
        for k, v in entry(M_B, 1, 1).items():
            Y_s2_raw[k] = Y_s2_raw.get(k, 0) - v
        Y_s2 = {k: v for k, v in Y_s2_raw.items() if v != 0}
        products.append((X_s2, Y_s2))
        
        # s3 = X11 * (Y10 - Y00)
        X_s3 = entry(M_A, 1, 1)
        Y_s3_raw = {}
        for k, v in entry(M_B, 1, 0).items():
            Y_s3_raw[k] = Y_s3_raw.get(k, 0) + v
        for k, v in entry(M_B, 0, 0).items():
            Y_s3_raw[k] = Y_s3_raw.get(k, 0) - v
        Y_s3 = {k: v for k, v in Y_s3_raw.items() if v != 0}
        products.append((X_s3, Y_s3))
        
        # s4 = (X00 + X01) * Y11
        X_s4_raw = {}
        for k, v in entry(M_A, 0, 0).items():
            X_s4_raw[k] = X_s4_raw.get(k, 0) + v
        for k, v in entry(M_A, 0, 1).items():
            X_s4_raw[k] = X_s4_raw.get(k, 0) + v
        X_s4 = {k: v for k, v in X_s4_raw.items() if v != 0}
        Y_s4 = entry(M_B, 1, 1)
        products.append((X_s4, Y_s4))
        
        # s5 = (X10 - X00) * (Y00 + Y01)
        X_s5_raw = {}
        for k, v in entry(M_A, 1, 0).items():
            X_s5_raw[k] = X_s5_raw.get(k, 0) + v
        for k, v in entry(M_A, 0, 0).items():
            X_s5_raw[k] = X_s5_raw.get(k, 0) - v
        X_s5 = {k: v for k, v in X_s5_raw.items() if v != 0}
        Y_s5_raw = {}
        for k, v in entry(M_B, 0, 0).items():
            Y_s5_raw[k] = Y_s5_raw.get(k, 0) + v
        for k, v in entry(M_B, 0, 1).items():
            Y_s5_raw[k] = Y_s5_raw.get(k, 0) + v
        Y_s5 = {k: v for k, v in Y_s5_raw.items() if v != 0}
        products.append((X_s5, Y_s5))
        
        # s6 = (X01 - X11) * (Y10 + Y11)
        X_s6_raw = {}
        for k, v in entry(M_A, 0, 1).items():
            X_s6_raw[k] = X_s6_raw.get(k, 0) + v
        for k, v in entry(M_A, 1, 1).items():
            X_s6_raw[k] = X_s6_raw.get(k, 0) - v
        X_s6 = {k: v for k, v in X_s6_raw.items() if v != 0}
        Y_s6_raw = {}
        for k, v in entry(M_B, 1, 0).items():
            Y_s6_raw[k] = Y_s6_raw.get(k, 0) + v
        for k, v in entry(M_B, 1, 1).items():
            Y_s6_raw[k] = Y_s6_raw.get(k, 0) + v
        Y_s6 = {k: v for k, v in Y_s6_raw.items() if v != 0}
        products.append((X_s6, Y_s6))
        
        return products
    
    # Now, for each of the 7 block terms, get the 7 scalar products.
    # Each scalar product contributes to certain entries of the block result.
    
    # The block result C_block has entries (dr, dc) where dr, dc in {0,1}.
    # The contribution to C_block[dr, dc] from the 7 Strassen products:
    # C[0,0] = s0 + s3 - s4 + s6
    # C[0,1] = s2 + s4
    # C[1,0] = s1 + s3
    # C[1,1] = s0 - s1 + s2 + s5
    
    block_result_contributions = [
        # (product_index, dr, dc, weight)
        (0, 0, 0, 1),   # s0 to C[0,0]
        (2, 0, 0, 1),   # s3 to C[0,0]
        (3, 0, 0, -1),  # s4 to C[0,0]
        (6, 0, 0, 1),   # s6 to C[0,0]
        (2, 0, 1, 1),   # s2 to C[0,1]
        (3, 0, 1, 1),   # s4 to C[0,1]
        (1, 1, 0, 1),   # s1 to C[1,0]
        (2, 1, 0, 1),   # s3 to C[1,0]
        (0, 1, 1, 1),   # s0 to C[1,1]
        (1, 1, 1, -1),  # s1 to C[1,1]
        (2, 1, 1, 1),   # s2 to C[1,1]
        (4, 1, 1, 1),   # s5 to C[1,1]
    ]
    
    # Now we need to map block results to the full 4x4 result.
    # Block (br, bc) of the result C corresponds to rows 2*br to 2*br+1
    # and cols 2*bc to 2*bc+1.
    
    # The 7 block terms contribute to different blocks of C:
    # P1 -> contributes to C11, C12, C21, C22
    # P2 -> contributes to C21, C22
    # P3 -> contributes to C12, C22
    # P4 -> contributes to C11, C21
    # P5 -> contributes to C11, C12
    # P6 -> contributes to C22
    # P7 -> contributes to C11, C22
    
    # From the 4x4 Strassen formula:
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    block_contributions = [
        # (block_term_index, result_block, weight)
        (0, (0, 0), 1),   # P1 to C11
        (3, (0, 0), 1),   # P4 to C11
        (4, (0, 0), -1),  # P5 to C11
        (6, (0, 0), 1),   # P7 to C11
        (2, (0, 1), 1),   # P3 to C12
        (4, (0, 1), 1),   # P5 to C12
        (1, (1, 0), 1),   # P2 to C21
        (3, (1, 0), 1),   # P4 to C21
        (0, (1, 1), 1),   # P1 to C22
        (1, (1, 1), -1),  # P2 to C22
        (2, (1, 1), 1),   # P3 to C22
        (5, (1, 1), 1),   # P6 to C22
    ]
    
    # Now build all scalar multiplication terms.
    # Each term: (L_A, L_B, weight, (global_i, global_k))
    
    all_terms = []
    
    for block_term_idx, (M_A_block, M_B_block) in enumerate(block_terms):
        products = strassen_products(M_A_block, M_B_block)
        
        for product_idx, (L_A_prod, L_B_prod) in enumerate(products):
            if not L_A_prod or not L_B_prod:
                continue
            
            # Find which result blocks this product contributes to
            for result_br, result_bc, block_weight in block_contributions:
                if block_term_idx != block_contributions[0][0]:  # This is wrong
                    pass
            
            # Let me redo this more carefully.
    
    # Let me restructure. For each block term t (0-6), and for each of its
    # 7 Strassen products s (0-6), and for each result block (br, bc) that
    # t contributes to with weight w_block, and for each local position (dr, dc)
    # in the result block that s contributes to with weight w_local:
    # The global position is (2*br + dr, 2*bc + dc)
    # The total weight is w_block * w_local
    
    for block_term_idx, (M_A_block, M_B_block) in enumerate(block_terms):
        products = strassen_products(M_A_block, M_B_block)
        
        for product_idx, (L_A_prod, L_B_prod) in enumerate(products):
            if not L_A_prod or not L_B_prod:
                continue
            
            # Find all (result_block, block_weight) for this block_term
            for result_br, result_bc, block_weight in block_contributions:
                if block_contributions[0][0] != block_term_idx and True:
                    pass
            
            # I'm getting confused. Let me just iterate properly.
    
    # Let me redo the block_contributions lookup.
    block_contrib_lookup = {}
    for block_term_idx, result_block, weight in block_contributions:
        if block_term_idx not in block_contrib_lookup:
            block_contrib_lookup[block_term_idx] = []
        block_contrib_lookup[block_term_idx].append((result_block, weight))
    
    # Similarly for local contributions.
    local_contrib_lookup = {}
    for product_idx, dr, dc, weight in block_result_contributions:
        if product_idx not in local_contrib_lookup:
            local_contrib_lookup[product_idx] = []
        local_contrib_lookup[product_idx].append((dr, dc, weight))
    
    all_terms = []
    
    for block_term_idx, (M_A_block, M_B_block) in enumerate(block_terms):
        products = strassen_products(M_A_block, M_B_block)
        
        for product_idx, (L_A_prod, L_B_prod) in enumerate(products):
            if not L_A_prod or not L_B_prod:
                continue
            
            # Get block contributions for this block term
            if block_term_idx not in block_contrib_lookup:
                continue
            
            for (result_br, result_bc), block_weight in block_contrib_lookup[block_term_idx]:
                # Get local contributions for this product
                if product_idx not in local_contrib_lookup:
                    continue
                
                for dr, dc, local_weight in local_contrib_lookup[product_idx]:
                    total_weight = block_weight * local_weight
                    if total_weight == 0:
                        continue
                    
                    global_i = 2 * result_br + dr
                    global_k = 2 * result_bc + dc
                    
                    all_terms.append((L_A_prod, L_B_prod, total_weight, (global_i, global_k)))
    
    # Now, group terms by (L_A, L_B) to form rank-1 components.
    from collections import defaultdict
    
    groups = defaultdict(lambda: defaultdict(float))
    
    for L_A, L_B, weight, (i, k) in all_terms:
        # Convert to hashable keys
        key_A = tuple(sorted(L_A.items()))
        key_B = tuple(sorted(L_B.items()))
        groups[(key_A, key_B)][(i, k)] += weight
    
    rank = len(groups)
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, ((key_A, key_B), ik_weights) in enumerate(groups.items()):
        # Fill U: U[i*m + j, r] = coeff for A[i,j]
        for (i, j), coeff in key_A:
            if coeff != 0:
                U[i * m + j][r] = coeff
        
        # Fill V: V[j*p + k, r] = coeff for B[j,k]
        for (j, k), coeff in key_B:
            if coeff != 0:
                V[j * p + k][r] = coeff
        
        # Fill W: W[k*n + i, r] = weight for C[i,k]
        for (i, k), weight in ik_weights.items():
            if weight != 0:
                W[k * n + i][r] = weight
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
