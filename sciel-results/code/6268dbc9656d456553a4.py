"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Recursive block partitioning with Strassen-like base cases
    # Strategy: Split into sub-blocks, use Strassen for 2x2x2 base cases
    # where beneficial, schoolbook otherwise
    
    terms = []  # list of (u_idx, v_idx, w_idx) for rank-1 terms
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        """Add a rank-1 term."""
        terms.append((u_idx, v_idx, w_idx, val))
    
    def schoolbook_block(a_i_start, a_j_start, b_j_start, b_k_start, 
                         a_rows, a_cols, b_rows, b_cols):
        """Add schoolbook terms for a block multiplication.
        A block: rows a_i_start..a_i_start+a_rows-1, cols a_j_start..a_j_start+a_cols-1
        B block: rows b_j_start..b_j_start+b_rows-1, cols b_k_start..b_k_start+b_cols-1
        Contributes to C at corresponding positions.
        """
        for di in range(a_rows):
            for dj in range(a_cols):
                for dk in range(b_cols):
                    i = a_i_start + di
                    j = a_j_start + dj
                    k = b_k_start + dk
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    add_term(u_idx, v_idx, w_idx, 1)
    
    def strassen_2x2(a_i_start, a_j_start, b_j_start, b_k_start):
        """Apply Strassen's algorithm for 2x2 x 2x2 -> 2x2 block multiplication.
        A block: 2x2 starting at (a_i_start, a_j_start)
        B block: 2x2 starting at (b_j_start, b_k_start)
        C block: 2x2 at (a_i_start, b_k_start)
        Uses 7 multiplications instead of 4.
        """
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        # C = [[C11, C12], [C21, C22]]
        
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
        
        # Each "multiplication" is actually a sum of products of linear combinations
        # We need to decompose each M into rank-1 terms
        
        # For M1 = (A11+A22)(B11+B22):
        # A11+A22 is a linear combination of A elements
        # B11+B22 is a linear combination of B elements
        # Their product contributes to C11 and C22
        
        # Let's define the linear combinations:
        # For A side:
        # L1_A = A11 + A22 (indices: (a_i_start, a_j_start) and (a_i_start+1, a_j_start+1))
        # L2_A = A21 + A22 (indices: (a_i_start+1, a_j_start) and (a_i_start+1, a_j_start+1))
        # L3_A = A11 (index: (a_i_start, a_j_start))
        # L4_A = A22 (index: (a_i_start+1, a_j_start+1))
        # L5_A = A11 + A12 (indices: (a_i_start, a_j_start) and (a_i_start, a_j_start+1))
        # L6_A = A21 - A11 (indices: (a_i_start+1, a_j_start) and (a_i_start, a_j_start))
        # L7_A = A12 - A22 (indices: (a_i_start, a_j_start+1) and (a_i_start+1, a_j_start+1))
        
        # For B side:
        # L1_B = B11 + B22 (indices: (b_j_start, b_k_start) and (b_j_start+1, b_k_start+1))
        # L2_B = B11 (index: (b_j_start, b_k_start))
        # L3_B = B12 - B22 (indices: (b_j_start, b_k_start+1) and (b_j_start+1, b_k_start+1))
        # L4_B = B21 - B11 (indices: (b_j_start+1, b_k_start) and (b_j_start, b_k_start))
        # L5_B = B22 (index: (b_j_start+1, b_k_start+1))
        # L6_B = B11 + B12 (indices: (b_j_start, b_k_start) and (b_j_start, b_k_start+1))
        # L7_B = B21 + B22 (indices: (b_j_start+1, b_k_start) and (b_j_start+1, b_k_start+1))
        
        # Now each M_t = L_t_A * L_t_B is a product of two linear combinations
        # L_t_A * L_t_B = sum over all pairs of terms in the linear combinations
        # Each pair gives a rank-1 term
        
        # M1: L1_A = A11 + A22, L1_B = B11 + B22
        # = A11*B11 + A11*B22 + A22*B11 + A22*B22
        # Contributes to C11 (+1) and C22 (+1)
        # C11 gets: +A11*B11, +A11*B22, +A22*B11, +A22*B22
        # C22 gets: +A11*B11, +A11*B22, +A22*B11, +A22*B22
        
        # M2: L2_A = A21 + A22, L2_B = B11
        # = A21*B11 + A22*B11
        # Contributes to C21 (+1) and C22 (-1)
        # C21 gets: +A21*B11, +A22*B11
        # C22 gets: -A21*B11, -A22*B11
        
        # M3: L3_A = A11, L3_B = B12 - B22
        # = A11*B12 - A11*B22
        # Contributes to C12 (+1) and C22 (+1)
        # C12 gets: +A11*B12, -A11*B22
        # C22 gets: +A11*B12, -A11*B22
        
        # M4: L4_A = A22, L4_B = B21 - B11
        # = A22*B21 - A22*B11
        # Contributes to C11 (+1), C21 (+1), C22 (-1)
        # C11 gets: +A22*B21, -A22*B11
        # C21 gets: +A22*B21, -A22*B11
        # C22 gets: -A22*B21, +A22*B11
        
        # M5: L5_A = A11 + A12, L5_B = B22
        # = A11*B22 + A12*B22
        # Contributes to C11 (-1) and C12 (+1)
        # C11 gets: -A11*B22, -A12*B22
        # C12 gets: +A11*B22, +A12*B22
        
        # M6: L6_A = A21 - A11, L6_B = B11 + B12
        # = A21*B11 + A21*B12 - A11*B11 - A11*B12
        # Contributes to C22 (+1)
        # C22 gets: +A21*B11, +A21*B12, -A11*B11, -A11*B12
        
        # M7: L7_A = A12 - A22, L7_B = B21 + B22
        # = A12*B21 + A12*B22 - A22*B21 - A22*B22
        # Contributes to C11 (+1) and C22 (+1)
        # C11 gets: +A12*B21, +A12*B22, -A22*B21, -A22*B22
        # C22 gets: +A12*B21, +A12*B22, -A22*B21, -A22*B22
        
        # Now let's collect all contributions to each C element:
        
        # C11 = M1 + M4 - M5 + M7
        # From M1: +A11*B11, +A11*B22, +A22*B11, +A22*B22
        # From M4: +A22*B21, -A22*B11
        # From -M5: -A11*B22, -A12*B22
        # From M7: +A12*B21, +A12*B22, -A22*B21, -A22*B22
        # C11 = A11*B11 + A11*B22 + A22*B11 + A22*B22 + A22*B21 - A22*B11 
        #      - A11*B22 - A12*B22 + A12*B21 + A12*B22 - A22*B21 - A22*B22
        # = A11*B11 + (A11*B22 - A11*B22) + (A22*B11 - A22*B11) + (A22*B22 - A22*B22)
        #   + (A22*B21 - A22*B21) + (-A12*B22 + A12*B22) + A12*B21
        # = A11*B11 + A12*B21
        # Which is correct! C11 = A11*B11 + A12*B21 ✓
        
        # Similarly for other C elements, this should work out.
        # The key insight is that we use 7 "multiplications" (each being a product of linear combinations)
        # instead of 4 direct products.
        
        # Let's add the terms:
        # Each M_t contributes to specific C elements with specific signs
        
        # Helper to add a product of linear combinations
        def add_product_contribution(a_terms, b_terms, c_i, c_k, sign=1):
            """Add sign * (sum a_terms) * (sum b_terms) to C[c_i, c_k]
            a_terms: list of (a_row, a_col, coeff)
            b_terms: list of (b_row, b_col, coeff)
            """
            for (ar, ac, ac_coeff) in a_terms:
                for (br, bc, bc_coeff) in b_terms:
                    u_idx = ar * m + ac
                    v_idx = br * p + bc
                    w_idx = bc * n + ar
                    val = sign * ac_coeff * bc_coeff
                    add_term(u_idx, v_idx, w_idx, val)
        
        # Define A linear combinations
        A11 = [(a_i_start, a_j_start, 1)]
        A12 = [(a_i_start, a_j_start+1, 1)]
        A21 = [(a_i_start+1, a_j_start, 1)]
        A22 = [(a_i_start+1, a_j_start+1, 1)]
        
        # M1: (A11+A22)(B11+B22)
        M1_A = A11 + A22
        M1_B = [(b_j_start, b_k_start, 1), (b_j_start+1, b_k_start+1, 1)]
        # C11 += M1, C22 += M1
        add_product_contribution(M1_A, M1_B, a_i_start, b_k_start, 1)
        add_product_contribution(M1_A, M1_B, a_i_start+1, b_k_start+1, 1)
        
        # M2: (A21+A22)*B11
        M2_A = A21 + A22
        M2_B = [(b_j_start, b_k_start, 1)]
        # C21 += M2, C22 -= M2
        add_product_contribution(M2_A, M2_B, a_i_start+1, b_k_start, 1)
        add_product_contribution(M2_A, M2_B, a_i_start+1, b_k_start+1, -1)
        
        # M3: A11*(B12-B22)
        M3_A = A11
        M3_B = [(b_j_start, b_k_start+1, 1), (b_j_start+1, b_k_start+1, -1)]
        # C12 += M3, C22 += M3
        add_product_contribution(M3_A, M3_B, a_i_start, b_k_start+1, 1)
        add_product_contribution(M3_A, M3_B, a_i_start+1, b_k_start+1, 1)
        
        # M4: A22*(B21-B11)
        M4_A = A22
        M4_B = [(b_j_start+1, b_k_start, 1), (b_j_start, b_k_start, -1)]
        # C11 += M4, C21 += M4, C22 -= M4
        add_product_contribution(M4_A, M4_B, a_i_start, b_k_start, 1)
        add_product_contribution(M4_A, M4_B, a_i_start+1, b_k_start, 1)
        add_product_contribution(M4_A, M4_B, a_i_start+1, b_k_start+1, -1)
        
        # M5: (A11+A12)*B22
        M5_A = A11 + A12
        M5_B = [(b_j_start+1, b_k_start+1, 1)]
        # C11 -= M5, C12 += M5
        add_product_contribution(M5_A, M5_B, a_i_start, b_k_start, -1)
        add_product_contribution(M5_A, M5_B, a_i_start, b_k_start+1, 1)
        
        # M6: (A21-A11)*(B11+B12)
        M6_A = [(a_i_start+1, a_j_start, 1), (a_i_start, a_j_start, -1)]
        M6_B = [(b_j_start, b_k_start, 1), (b_j_start, b_k_start+1, 1)]
        # C22 += M6
        add_product_contribution(M6_A, M6_B, a_i_start+1, b_k_start+1, 1)
        
        # M7: (A12-A22)*(B21+B22)
        M7_A = [(a_i_start, a_j_start+1, 1), (a_i_start+1, a_j_start+1, -1)]
        M7_B = [(b_j_start+1, b_k_start, 1), (b_j_start+1, b_k_start+1, 1)]
        # C11 += M7, C22 += M7
        add_product_contribution(M7_A, M7_B, a_i_start, b_k_start, 1)
        add_product_contribution(M7_A, M7_B, a_i_start+1, b_k_start+1, 1)
    
    # Now let's build the decomposition for 4x5 x 5x5
    # Partition:
    # A (4x5): rows 2+2, cols 3+2
    # B (5x5): rows 3+2, cols 3+2
    
    # Sub-blocks:
    # A11: (0,0) 2x3, A12: (0,3) 2x2
    # A21: (2,0) 2x3, A22: (2,3) 2x2
    # B11: (0,0) 3x3, B12: (0,3) 3x2
    # B21: (3,0) 2x3, B22: (3,3) 2x2
    
    # C11 = A11@B11 + A12@B21 (2x3)
    # C12 = A11@B12 + A12@B22 (2x2)
    # C21 = A21@B11 + A22@B21 (2x3)
    # C22 = A21@B12 + A22@B22 (2x2)
    
    # For C12 and C22 (2x2 blocks), we can potentially use Strassen if we can
    # express them as 2x2 matrix multiplications. But they are sums of products.
    # Let's just use schoolbook for now and see if we can improve.
    
    # Actually, let's try a different approach: pad to make square blocks
    # and use Strassen recursively.
    
    # For now, let's use the 92-rank schoolbook decomposition and see if we
    # can apply Strassen to any sub-blocks.
    
    # Use schoolbook for all sub-blocks:
    # A11@B11: (2x3)@(3x3) -> 18 mults
    schoolbook_block(0, 0, 0, 0, 2, 3, 3, 3)
    
    # A12@B21: (2x2)@(2x3) -> 12 mults
    schoolbook_block(0, 3, 3, 0, 2, 2, 2, 3)
    
    # A11@B12: (2x3)@(3x2) -> 12 mults
    schoolbook_block(0, 0, 0, 3, 2, 3, 3, 2)
    
    # A12@B22: (2x2)@(2x2) -> 4 mults
    schoolbook_block(0, 3, 3, 3, 2, 2, 2, 2)
    
    # A21@B11: (2x3)@(3x3) -> 18 mults
    schoolbook_block(2, 0, 0, 0, 2, 3, 3, 3)
    
    # A22@B21: (2x2)@(2x3) -> 12 mults
    schoolbook_block(2, 3, 3, 0, 2, 2, 2, 3)
    
    # A21@B12: (2x3)@(3x2) -> 12 mults
    schoolbook_block(2, 0, 0, 3, 2, 3, 3, 2)
    
    # A22@B22: (2x2)@(2x2) -> 4 mults
    schoolbook_block(2, 3, 3, 3, 2, 2, 2, 2)
    
    rank = len(terms)
    
    # Build U, V, W with proper coefficients
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r, (u_idx, v_idx, w_idx, val) in enumerate(terms):
        U[u_idx][r] = val
        V[v_idx][r] = 1.0
        W[w_idx][r] = 1.0
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
