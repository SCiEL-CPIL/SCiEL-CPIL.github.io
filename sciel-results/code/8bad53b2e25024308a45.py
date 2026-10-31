"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-56 decomposition of the 4x4x4 matrix multiplication tensor
    using Strassen's algorithm on 2x2 blocks."""
    n, m, p = DIMENSIONS  # 4, 4, 4
    # The rank is 8 multiplications of 2x2 matrices, each done with rank 7
    rank = 56

    # Initialize U, V, W matrices
    # U: (n*m, rank) = (16, 56)
    # V: (m*p, rank) = (16, 56)
    # W: (p*n, rank) = (16, 56)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # We decompose the 4x4 matrices into 2x2 blocks of 2x2 submatrices.
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]

    # For each of the 8 block-multiplications in Strassen's algorithm,
    # we apply the rank-7 decomposition for 2x2 matrix multiplication.

    # Strassen's 7 intermediate products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)

    # For a 2x2 matrix multiplication C = AB where A, B are 2x2:
    # The rank-7 decomposition has the following structure:
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # Then:
    # M1 = (a + d)(e + h)
    # M2 = (c + d)e
    # M3 = a(f - h)
    # M4 = d(g - e)
    # M5 = (a + b)h
    # M6 = (c - a)(e + f)
    # M7 = (b - d)(g + h)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # We'll define a helper to add a rank-7 decomposition for multiplying
    # two 2x2 matrices, where the input matrices have entries that are
    # linear combinations of the original 4x4 matrix entries.

    def add_rank7(U, V, W, offset, a_coeffs, b_coeffs):
        """
        Add a rank-7 decomposition for multiplying two 2x2 matrices.
        
        a_coeffs: 4x4 array where a_coeffs[i][j] is the coefficient vector
                  (length 16) for the (i,j) entry of the left 2x2 matrix
        b_coeffs: 4x4 array where b_coeffs[i][j] is the coefficient vector
                  (length 16) for the (i,j) entry of the right 2x2 matrix
        
        offset: starting column in U, V, W
        """
        # The 7 Strassen multipliers for 2x2:
        # M1: (a+d)(e+h) -> U has (1,0,1,0) for a,d; V has (1,0,0,1) for e,h
        # M2: (c+d)e    -> U has (0,0,1,1) for c,d; V has (1,0,0,0) for e
        # M3: a(f-h)    -> U has (1,0,0,0) for a; V has (0,1,0,-1) for f,h
        # M4: d(g-e)    -> U has (0,0,0,1) for d; V has (0,0,1,-1) for g,e -> (-1,0,1,0)
        # M5: (a+b)h    -> U has (1,1,0,0) for a,b; V has (0,0,0,1) for h
        # M6: (c-a)(e+f) -> U has (-1,0,1,0) for c,a; V has (1,1,0,0) for e,f
        # M7: (b-d)(g+h) -> U has (0,1,0,-1) for b,d; V has (0,0,1,1) for g,h
        
        strassen_patterns = [
            # (U_coeffs for a,b,c,d), (V_coeffs for e,f,g,h)
            ((1, 0, 1, 0), (1, 0, 0, 1)),  # M1
            ((0, 0, 1, 1), (1, 0, 0, 0)),  # M2
            ((1, 0, 0, 0), (0, 1, 0, -1)), # M3
            ((0, 0, 0, 1), (-1, 0, 1, 0)), # M4
            ((1, 1, 0, 0), (0, 0, 0, 1)),  # M5
            ((-1, 0, 1, 0), (1, 1, 0, 0)), # M6
            ((0, 1, 0, -1), (0, 0, 1, 1)), # M7
        ]
        
        # For each of the 7 multipliers:
        for mult_idx in range(7):
            col = offset + mult_idx
            u_pattern, v_pattern = strassen_patterns[mult_idx]
            
            # u_pattern has 4 entries for the 2x2 matrix A positions: [a, b, c, d]
            # v_pattern has 4 entries for the 2x2 matrix B positions: [e, f, g, h]
            
            # For U: each position (i,j) in the 2x2 left matrix corresponds to
            # a specific entry in the 4x4 matrix. We need to map this to the
            # U vector.
            # The 2x2 matrix positions are:
            # (0,0) -> a, (0,1) -> b, (1,0) -> c, (1,1) -> d
            # In the 4x4 matrix, these correspond to specific indices.
            
            # For U (left matrix, n*m = 16 entries):
            # The left 2x2 matrix is a sub-block of the 4x4 matrix.
            # We need to know which 2x2 block of the 4x4 matrix this corresponds to.
            # a_coeffs tells us the linear combination for each position.
            
            # For U: for each position (pos_idx, coeff) in u_pattern:
            # pos_idx 0 -> a, 1 -> b, 2 -> c, 3 -> d
            # The coefficient vector for that position is in a_coeffs
            for pos_idx in range(4):
                coeff = u_pattern[pos_idx]
                if coeff != 0:
                    # a_coeffs[pos_idx // 2][pos_idx % 2] gives the coefficient vector
                    # for this position in the 2x2 matrix
                    coeff_vec = a_coeffs[pos_idx // 2][pos_idx % 2]
                    for entry_idx in range(16):
                        U[entry_idx][col] += coeff * coeff_vec[entry_idx]
            
            # For V (right matrix, m*p = 16 entries):
            # v_pattern has 4 entries: [e, f, g, h]
            # pos_idx 0 -> e, 1 -> f, 2 -> g, 3 -> h
            for pos_idx in range(4):
                coeff = v_pattern[pos_idx]
                if coeff != 0:
                    coeff_vec = b_coeffs[pos_idx // 2][pos_idx % 2]
                    for entry_idx in range(16):
                        V[entry_idx][col] += coeff * coeff_vec[entry_idx]
        
        # For W (output, p*n = 16 entries):
        # The output C00, C01, C10, C11 each get contributions from the 7 multipliers.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # The output 2x2 matrix positions are:
        # (0,0) -> C00, (0,1) -> C01, (1,0) -> C10, (1,1) -> C11
        # In the 4x4 output matrix, these correspond to specific indices.
        
        # We need to know which 2x2 block of the output this corresponds to.
        # The output coefficients determine where in the 4x4 matrix the results go.
        
        # For each output position (out_pos_idx) in {0,1,2,3} for [C00, C01, C10, C11]:
        # out_pos_idx 0 -> C00, 1 -> C01, 2 -> C10, 3 -> C11
        
        # The contribution of each multiplier to each output position:
        # M1 contributes to C00 (+1) and C11 (+1)
        # M2 contributes to C10 (+1) and C11 (-1)
        # M3 contributes to C01 (+1) and C11 (+1)
        # M4 contributes to C00 (+1) and C10 (+1)
        # M5 contributes to C00 (-1) and C01 (+1)
        # M6 contributes to C11 (+1)
        # M7 contributes to C00 (+1)
        
        w_contributions = [
            # For each multiplier (0-6), for each output position (0-3):
            # multiplier 0 (M1): C00(+1), C01(0), C10(0), C11(+1)
            [1, 0, 0, 1],
            # multiplier 1 (M2): C00(0), C01(0), C10(+1), C11(-1)
            [0, 0, 1, -1],
            # multiplier 2 (M3): C00(0), C01(+1), C10(0), C11(+1)
            [0, 1, 0, 1],
            # multiplier 3 (M4): C00(+1), C01(0), C10(+1), C11(0)
            [1, 0, 1, 0],
            # multiplier 4 (M5): C00(-1), C01(+1), C10(0), C11(0)
            [-1, 1, 0, 0],
            # multiplier 5 (M6): C00(0), C01(0), C10(0), C11(+1)
            [0, 0, 0, 1],
            # multiplier 6 (M7): C00(+1), C01(0), C10(0), C11(0)
            [1, 0, 0, 0],
        ]
        
        # Now we need to map output positions to the 4x4 output matrix.
        # The output 2x2 block corresponds to specific entries in the 4x4 matrix.
        # We need to define the output mapping.
        
        # Actually, I realize this approach is getting complex. Let me use a different
        # strategy: directly construct the full decomposition.
        
        pass  # This function is not used in the final implementation below

    # Let me use a cleaner approach: directly specify the 56 columns.
    # The 4x4 matrices are divided into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # A00 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]  (rows 0-1, cols 0-1)
    # A01 = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]  (rows 0-1, cols 2-3)
    # A10 = [[A[2,0], A[2,1]], [A[3,0], A[3,1]]]  (rows 2-3, cols 0-1)
    # A11 = [[A[2,2], A[2,3]], [A[3,2], A[3,3]]]  (rows 2-3, cols 2-3)
    
    # Similarly for B and C.
    
    # The 8 Strassen products for block multiplication:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B01 - B11)
    # P4 = A11(B10 - B00)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    
    # And:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Each Pi is a 2x2 matrix multiplication, done with rank 7.
    # So we have 8 * 7 = 56 total multiplications.
    
    # For each Pi, we need to specify:
    # - Which entries of A and B go into the two 2x2 matrices being multiplied
    # - Where the resulting 2x2 matrix goes in C
    
    # Let's define the 8 products more explicitly.
    
    # I'll use a different approach: directly build the U, V, W matrices
    # by specifying each of the 56 columns.
    
    # Helper function to add a rank-7 decomposition for multiplying two 2x2 matrices
    # where the left matrix has entries given by linear combinations of A's entries
    # and the right matrix has entries given by linear combinations of B's entries.
    
    def get_entry_index(matrix, row, col):
        """Get the flat index for entry (row, col) in a 4x4 matrix."""
        return row * 4 + col
    
    # For the left matrix (A side), we need to map 2x2 positions to 4x4 entries.
    # For a given 2x2 block (block_row, block_col) where block_row, block_col in {0, 1}:
    # The 2x2 entries are:
    # (0,0) -> A[block_row*2, block_col*2]
    # (0,1) -> A[block_row*2, block_col*2 + 1]
    # (1,0) -> A[block_row*2 + 1, block_col*2]
    # (1,1) -> A[block_row*2 + 1, block_col*2 + 1]
    
    # For each of the 8 products, we define:
    # - left_blocks: list of (block_row, block_col, sign) for the left 2x2 matrix
    # - right_blocks: list of (block_row, block_col, sign) for the right 2x2 matrix
    # - output_block: (block_row, block_col) for where the result goes
    
    # Product P1 = (A00 + A11)(B00 + B11)
    # Left: A00 (block 0,0) + A11 (block 1,1)
    # Right: B00 (block 0,0) + B11 (block 1,1)
    # Output: C00 (block 0,0)
    
    # Product P2 = (A10 + A11)B00
    # Left: A10 (block 1,0) + A11 (block 1,1)
    # Right: B00 (block 0,0)
    # Output: C10 (block 1,0)
    
    # Product P3 = A00(B01 - B11)
    # Left: A00 (block 0,0)
    # Right: B01 (block 0,1) - B11 (block 1,1)
    # Output: C01 (block 0,1)
    
    # Product P4 = A11(B10 - B00)
    # Left: A11 (block 1,1)
    # Right: B10 (block 1,0) - B00 (block 0,0)
    # Output: C00 (block 0,0) and C10 (block 1,0)
    
    # Product P5 = (A00 + A01)B11
    # Left: A00 (block 0,0) + A01 (block 0,1)
    # Right: B11 (block 1,1)
    # Output: C00 (block 0,0) and C01 (block 0,1)
    
    # Product P6 = (A10 - A00)(B00 + B01)
    # Left: A10 (block 1,0) - A00 (block 0,0)
    # Right: B00 (block 0,0) + B01 (block 0,1)
    # Output: C11 (block 1,1)
    
    # Product P7 = (A01 - A11)(B10 + B11)
    # Left: A01 (block 0,1) - A11 (block 1,1)
    # Right: B10 (block 1,0) + B11 (block 1,1)
    # Output: C00 (block 0,0)
    
    # For each product, we apply the rank-7 Strassen decomposition.
    
    # Strassen patterns for 2x2:
    strassen_patterns = [
        # (left_coeffs [a,b,c,d], right_coeffs [e,f,g,h])
        ((1, 0, 1, 0), (1, 0, 0, 1)),  # M1: (a+d)(e+h)
        ((0, 0, 1, 1), (1, 0, 0, 0)),  # M2: (c+d)e
        ((1, 0, 0, 0), (0, 1, 0, -1)), # M3: a(f-h)
        ((0, 0, 0, 1), (-1, 0, 1, 0)), # M4: d(g-e)
        ((1, 1, 0, 0), (0, 0, 0, 1)),  # M5: (a+b)h
        ((-1, 0, 1, 0), (1, 1, 0, 0)), # M6: (c-a)(e+f)
        ((0, 1, 0, -1), (0, 0, 1, 1)), # M7: (b-d)(g+h)
    ]
    
    # Output contributions for Strassen multipliers:
    # [C00, C01, C10, C11]
    strassen_output = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1],  # M3
        [1, 0, 1, 0],  # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1],  # M6
        [1, 0, 0, 0],  # M7
    ]
    
    # Define the 8 products with their left and right block specifications
    # Each block spec is a list of (block_row, block_col, sign)
    
    products = [
        # P1
        {
            'left': [(0, 0, 1), (1, 1, 1)],  # A00 + A11
            'right': [(0, 0, 1), (1, 1, 1)], # B00 + B11
            'output': (0, 0)  # C00
        },
        # P2
        {
            'left': [(1, 0, 1), (1, 1, 1)],  # A10 + A11
            'right': [(0, 0, 1)],             # B00
            'output': (1, 0)  # C10
        },
        # P3
        {
            'left': [(0, 0, 1)],              # A00
            'right': [(0, 1, 1), (1, 1, -1)], # B01 - B11
            'output': (0, 1)  # C01
        },
        # P4
        {
            'left': [(1, 1, 1)],              # A11
            'right': [(1, 0, 1), (0, 0, -1)], # B10 - B00
            'output': (0, 0)  # C00
        },
        # P5
        {
            'left': [(0, 0, 1), (0, 1, 1)],  # A00 + A01
            'right': [(1, 1, 1)],             # B11
            'output': (0, 0)  # C00
        },
        # P6
        {
            'left': [(1, 0, 1), (0, 0, -1)],  # A10 - A00
            'right': [(0, 0, 1), (0, 1, 1)],  # B00 + B01
            'output': (1, 1)  # C11
        },
        # P7
        {
            'left': [(0, 1, 1), (1, 1, -1)],  # A01 - A11
            'right': [(1, 0, 1), (1, 1, 1)],  # B10 + B11
            'output': (0, 0)  # C00
        },
        # We need an 8th product. Let me reconsider.
        # Actually, the standard Strassen for block matrices gives 7 products,
        # but we have 4 output blocks. Let me re-examine.
    ]
    
    # I realize I'm missing the 8th product. Let me use the standard 7-product
    # Strassen but I need to handle the fact that some outputs get multiple
    # contributions. Let me use a different approach.
    
    # Actually, for 2x2 block matrices, Strassen uses 7 products, not 8.
    # So the total rank should be 7 * 7 = 49, not 56.
    
    # Let me reconsider. The standard Strassen algorithm for 2x2 matrices
    # uses 7 multiplications. If we apply this to 2x2 blocks of 2x2 matrices,
    # we get 7 * 7 = 49 total scalar multiplications.
    
    # But wait, each "multiplication" in the block Strassen is itself a 2x2
    # matrix multiplication, which has rank 7. So total rank = 7 * 7 = 49.
    
    # Let me implement rank 49 instead.
    
    # I'll rebuild with rank 49.
    rank = 49
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # The 7 Strassen products for 2x2 block matrices:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B01 - B11)
    # P4 = A11(B10 - B00)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    
    # Output:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Each Pi is a 2x2 matrix multiplication with rank 7.
    
    # Define the 7 products
    products = [
        # P1
        {'left': [(0, 0, 1), (1, 1, 1)], 'right': [(0, 0, 1), (1, 1, 1)], 'output': (0, 0)},
        # P2
        {'left': [(1, 0, 1), (1, 1, 1)], 'right': [(0, 0, 1)], 'output': (1, 0)},
        # P3
        {'left': [(0, 0, 1)], 'right': [(0, 1, 1), (1, 1, -1)], 'output': (0, 1)},
        # P4
        {'left': [(1, 1, 1)], 'right': [(1, 0, 1), (0, 0, -1)], 'output': (0, 0)},
        # P5
        {'left': [(0, 0, 1), (0, 1, 1)], 'right': [(1, 1, 1)], 'output': (0, 0)},
        # P6
        {'left': [(1, 0, 1), (0, 0, -1)], 'right': [(0, 0, 1), (0, 1, 1)], 'output': (1, 1)},
        # P7
        {'left': [(0, 1, 1), (1, 1, -1)], 'right': [(1, 0, 1), (1, 1, 1)], 'output': (0, 0)},
    ]
    
    # For each product, apply the rank-7 Strassen decomposition.
    for prod_idx, prod in enumerate(products):
        base_col = prod_idx * 7
        
        # Build the left 2x2 matrix coefficients
        # left_coeffs[i][j] will be a list of 16 coefficients (one for each entry in A)
        left_coeffs = [[0.0] * 16 for _ in range(4)]
        for block_row, block_col, sign in prod['left']:
            for i in range(2):
                for j in range(2):
                    a_row = block_row * 2 + i
                    a_col = block_col * 2 + j
                    a_idx = a_row * 4 + a_col
                    # The position in the 2x2 matrix is (i, j)
                    # Map to Strassen positions: (0,0)->a, (0,1)->b, (1,0)->c, (1,1)->d
                    pos = i * 2 + j
                    left_coeffs[i][j][a_idx] += sign
        
        # Build the right 2x2 matrix coefficients
        right_coeffs = [[0.0] * 16 for _ in range(4)]
        for block_row, block_col, sign in prod['right']:
            for i in range(2):
                for j in range(2):
                    b_row = block_row * 2 + i
                    b_col = block_col * 2 + j
                    b_idx = b_row * 4 + b_col
                    pos = i * 2 + j
                    right_coeffs[i][j][b_idx] += sign
        
        # Apply the 7 Strassen multipliers
        for mult_idx in range(7):
            col = base_col + mult_idx
            u_pattern, v_pattern = strassen_patterns[mult_idx]
            
            # u_pattern: [coeff for a, b, c, d]
            # v_pattern: [coeff for e, f, g, h]
            
            # For U:
            for pos_idx in range(4):
                coeff = u_pattern[pos_idx]
                if coeff != 0:
                    # pos_idx 0->a(0,0), 1->b(0,1), 2->c(1,0), 3->d(1,1)
                    i = pos_idx // 2
                    j = pos_idx % 2
                    coeff_vec = left_coeffs[i][j]
                    for entry_idx in range(16):
                        U[entry_idx][col] += coeff * coeff_vec[entry_idx]
            
            # For V:
            for pos_idx in range(4):
                coeff = v_pattern[pos_idx]
                if coeff != 0:
                    i = pos_idx // 2
                    j = pos_idx % 2
                    coeff_vec = right_coeffs[i][j]
                    for entry_idx in range(16):
                        V[entry_idx][col] += coeff * coeff_vec[entry_idx]
        
        # For W: add contributions to the output block
        out_block_row, out_block_col = prod['output']
        for mult_idx in range(7):
            col = base_col + mult_idx
            # strassen_output[mult_idx] = [C00, C01, C10, C11]
            out_pos_idx = None
            if out_block_row == 0 and out_block_col == 0:
                out_pos_idx = 0  # C00
            elif out_block_row == 0 and out_block_col == 1:
                out_pos_idx = 1  # C01
            elif out_block_row == 1 and out_block_col == 0:
                out_pos_idx = 2  # C10
            elif out_block_row == 1 and out_block_col == 1:
                out_pos_idx = 3  # C11
            
            if out_pos_idx is not None:
                w_coeff = strassen_output[mult_idx][out_pos_idx]
                if w_coeff != 0:
                    for i in range(2):
                        for j in range(2):
                            c_row = out_block_row * 2 + i
                            c_col = out_block_col * 2 + j
                            c_idx = c_row * 4 + c_col
                            W[c_idx][col] += w_coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
