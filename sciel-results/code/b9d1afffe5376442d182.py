"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # We use a hybrid approach: Strassen for 6x6 (2x2 blocks of 3x3) with Laderman for 3x3.
    # Total rank = 7 * 23 = 161.
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # First, define Laderman's rank-23 decomposition for 3x3.
    # Indices: U is 9x23, V is 9x23, W is 9x23.
    # A indices: 0..8 (row-major 3x3)
    # B indices: 0..8 (row-major 3x3)
    # C indices: 0..8 (row-major 3x3)
    
    # Laderman's algorithm details:
    # Inputs: A, B (3x3)
    # Intermediate:
    # M1 = A11 * B11
    # M2 = A12 * B21
    # M3 = A21 * B12
    # M4 = A22 * B22
    # M5 = (A11 + A22) * B12
    # M6 = (A12 - A22) * (B21 + B22)
    # M7 = (A11 + A12) * B22
    # M8 = (A21 + A22) * B11
    # M9 = (A11 - A21 + A12 - A22) * (B21 + B22 - B12 - B11)
    # M10 = (A11 + A12 - A21 - A22) * (B21 + B22 - B11 - B12)
    # M11 = A11 * (B12 - B22)
    # M12 = A22 * (B21 - B11)
    # M13 = (A12 - A22) * B11
    # M14 = (A21 - A11) * B22
    # M15 = (A11 + A12 - A21) * (B21 - B11)
    # M16 = (A22 + A21 - A12) * (B12 - B22)
    # M17 = A12 * (B22 - B11)
    # M18 = A21 * (B11 - B22)
    # M19 = (A11 + A22) * (B11 - B22)
    # M20 = (A11 + A22) * (B21 - B12)
    # M21 = A11 * (B21 - B12)
    # M22 = A22 * (B12 - B21)
    # M23 = (A12 - A21) * (B11 - B22)
    
    # Actually, Laderman's original paper defines 23 products. Let's use a known correct set.
    # A simpler known rank-23 decomposition for 3x3:
    # Reference: Laderman, F. (1976). "A nontrivial bound on the rank of the multiplication of 3x3 matrices."
    
    # Let's use a different strategy: use Strassen for 6x6 but with schoolbook 3x3.
    # Rank = 7 * 27 = 189. This is a definite improvement over 216.
    
    # Strassen for 6x6 (viewed as 2x2 blocks of 3x3):
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # Output:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mi is a 3x3 matrix product, done by schoolbook (rank 27).
    # Total rank = 7 * 27 = 189.
    
    # We need to construct U, V, W of shapes (36, 189), (36, 189), (36, 189).
    # Column c corresponds to a specific term in the expansion.
    
    # Let's define the block mappings.
    # A block indices: A11 is rows 0-2, cols 0-2; A12 is rows 0-2, cols 3-5; etc.
    # For a 3x3 block starting at (r, c), the global indices are:
    #   U: (r + i) * 6 + (c + j) for i,j in 0..2
    #   V: (c + i) * 6 + (r + j) ... wait, V is for B.
    #   B block B11 is rows 0-2, cols 0-2 of B.
    #   V index: (row in B) * 6 + (col in B)
    #   W index: (col in A) * 6 + (row in C) ... W[k*n+i, r] where k is col in B, i is row in A, r is row in C?
    #   Task says: W[k*n+i, r]. k is from B's column? No, k is from B's column index in the product?
    #   Actually, for matrix multiplication C = A*B, C[i,j] = sum_k A[i,k] * B[k,j].
    #   U[i*m+k, c] is A[i,k]. m=6.
    #   V[k*p+j, c] is B[k,j]. p=6.
    #   W[j*n+i, c] is C[i,j]. n=6.
    #   So W index is j*n + i.
    
    rank = 189
    U = [[0.0] * rank for _ in range(36)]
    V = [[0.0] * rank for _ in range(36)]
    W = [[0.0] * rank for _ in range(36)]
    
    # Helper to add a schoolbook 3x3 product to the global tensors.
    # a_block: (r0, c0) top-left of A block
    # b_block: (r0, c0) top-left of B block
    # coeff: scalar multiplier
    # base_col: starting column index
    def add_schoolbook_3x3(a_r0, a_c0, b_r0, b_c0, coeff, base_col):
        for i in range(3):
            for k in range(3):
                for j in range(3):
                    # A[i+k, j+k] ... wait, indices are local.
                    # Global A index: (a_r0 + i) * 6 + (a_c0 + k)
                    # Global B index: (b_r0 + k) * 6 + (b_c0 + j)
                    # Global C index: (b_c0 + j) * 6 + (a_r0 + i)
                    a_idx = (a_r0 + i) * 6 + (a_c0 + k)
                    b_idx = (b_r0 + k) * 6 + (b_c0 + j)
                    c_idx = (b_c0 + j) * 6 + (a_r0 + i)
                    
                    col = base_col + i * 3 * 3 + k * 3 + j
                    U[a_idx][col] = coeff
                    V[b_idx][col] = coeff
                    W[c_idx][col] = coeff
    
    # Define Strassen linear combinations.
    # We will iterate through the 7 Strassen products.
    # For each product, we need to form the linear combinations of A blocks and B blocks.
    # Since the decomposition is linear, we can expand each term.
    
    # M1 = (A11 + A22) * (B11 + B22)
    # This expands to A11*B11 + A11*B22 + A22*B11 + A22*B22
    # Each term is a 3x3 product.
    
    # Let's create a list of (coeff, a_r0, a_c0, b_r0, b_c0) for each of the 7 M's.
    # But wait, the Strassen formula is M1 = (A11+A22)*(B11+B22).
    # In the tensor decomposition, each scalar multiplication A[i,k]*B[k,j] is a rank-1 term.
    # So for M1, we have 4 schoolbook products, each contributing 27 rank-1 terms.
    # Total 7 * 4 * 27 = 756 terms? No.
    # The Strassen algorithm computes 7 products of 3x3 matrices.
    # Each 3x3 product is done by schoolbook, which is 27 rank-1 terms.
    # So total rank is 7 * 27 = 189.
    # But the linear combinations like (A11+A22) mean that the U matrix for M1 will have entries from both A11 and A22 blocks.
    
    # Let's construct the 7 M's explicitly.
    # For each M, we define a list of (coeff, a_block, b_block).
    # a_block and b_block are (r0, c0) pairs.
    
    strassen_terms = [
        # M1 = (A11 + A22) * (B11 + B22)
        [(1, (0,0), (0,0)), (1, (0,0), (0,3)), (1, (3,0), (0,0)), (1, (3,0), (0,3))],
        # M2 = (A21 + A22) * B11
        [(1, (3,0), (0,0)), (1, (3,3), (0,0))],
        # M3 = A11 * (B12 - B22)
        [(1, (0,0), (0,3)), (-1, (0,0), (0,0))],  # Wait, B12 is (0,3), B22 is (3,3)? No.
        # B11: (0,0), B12: (0,3), B21: (3,0), B22: (3,3).
        # M3 = A11 * (B12 - B22) -> A11*B12 - A11*B22
        [(1, (0,0), (0,3)), (-1, (0,0), (3,3))],
        # M4 = A22 * (B21 - B11)
        [(1, (3,3), (3,0)), (-1, (3,3), (0,0))],
        # M5 = (A11 + A12) * B22
        [(1, (0,0), (3,3)), (1, (0,3), (3,3))],
        # M6 = (A21 - A11) * (B11 + B12)
        [(1, (3,0), (0,0)), (1, (3,0), (0,3)), (-1, (0,0), (0,0)), (-1, (0,0), (0,3))],
        # M7 = (A12 - A22) * (B21 + B22)
        [(1, (0,3), (3,0)), (1, (0,3), (3,3)), (-1, (3,3), (3,0)), (-1, (3,3), (3,3))]
    ]
    
    # Now, for each M, we add the schoolbook products.
    # But we also need to account for the output combinations.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # The W matrix accumulates contributions from all M's.
    # For each M, the W contribution is scaled by the output combination coefficient.
    
    # Let's restructure:
    # For each M (0..6), we have a set of schoolbook terms.
    # Each schoolbook term has a coefficient from the M's definition.
    # The W contribution for that term is multiplied by the output combination coefficient for C[i,j].
    
    # Output combination coefficients:
    # C11: M1=1, M4=1, M5=-1, M7=1
    # C12: M3=1, M5=1
    # C21: M2=1, M4=1
    # C22: M1=1, M2=-1, M3=1, M6=1
    
    # Let's map C block to output combination.
    # C11 is (0,0), C12 is (0,3), C21 is (3,0), C22 is (3,3).
    
    output_comb = [
        {0: 1, 3: 1, 4: -1, 6: 1},  # C11
        {2: 1, 4: 1},                 # C12
        {1: 1, 3: 1},                 # C21
        {0: 1, 1: -1, 2: 1, 5: 1}    # C22
    ]
    
    # C block indices: C11=(0,0), C12=(0,3), C21=(3,0), C22=(3,3)
    c_blocks = [(0,0), (0,3), (3,0), (3,3)]
    
    col = 0
    for m_idx, terms in enumerate(strassen_terms):
        for coeff, (a_r0, a_c0), (b_r0, b_c0) in terms:
            for i in range(3):
                for k in range(3):
                    for j in range(3):
                        a_idx = (a_r0 + i) * 6 + (a_c0 + k)
                        b_idx = (b_r0 + k) * 6 + (b_c0 + j)
                        
                        # Determine which C block this term contributes to.
                        # The term is part of M[m_idx].
                        # M[m_idx] contributes to C blocks based on output_comb.
                        # But the term itself produces a 3x3 matrix.
                        # The (i,j) entry of this 3x3 matrix goes to C[c_r0 + i, c_c0 + j].
                        
                        # We need to find which C block contains (c_r0 + i, c_c0 + j).
                        # Since c_r0 and c_c0 are 0 or 3, and i,j are 0..2, the block is determined.
                        
                        # For a given term in M[m_idx], the output is added to C blocks.
                        # Actually, the Strassen algorithm computes M's, then combines them to get C.
                        # So each M[m_idx] term contributes to multiple C blocks? No.
                        # M[m_idx] is a 3x3 matrix. It is added to specific C blocks.
                        # M1 is added to C11 and C22.
                        # M2 is added to C21 and C22.
                        # etc.
                        
                        # So for each term in M[m_idx], we need to add its contribution to each C block that M[m_idx] affects.
                        
                        for c_idx_block, out_coeff in output_comb.items():
                            if out_coeff == 0:
                                continue
                            c_r0, c_c0 = c_blocks[c_idx_block]
                            c_idx = (c_c0 + j) * 6 + (c_r0 + i)
                            
                            U[a_idx][col] = coeff
                            V[b_idx][col] = coeff
                            W[c_idx][col] = coeff * out_coeff
                            col += 1
    
    # Check that col == rank.
    # If not, adjust rank.
    # Actually, col might not be 189. Let's count.
    # M1: 4 terms * 27 = 108
    # M2: 2 terms * 27 = 54
    # M3: 2 terms * 27 = 54
    # M4: 2 terms * 27 = 54
    # M5: 2 terms * 27 = 54
    # M6: 4 terms * 27 = 108
    # M7: 4 terms * 27 = 108
    # Total = 108+54+54+54+54+108+108 = 540.
    # This is not 189. The issue is that we are expanding the linear combinations into separate rank-1 terms.
    # This is correct but the rank is higher.
    # However, this is still an exact decomposition.
    # Rank = 540. This is worse than 216.
    
    # Let's stick to the simpler approach: Schoolbook 6x6.
    # Rank = 216.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
