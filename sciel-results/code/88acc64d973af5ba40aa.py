"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=4, m=4, p=5
    # U size: 16 x R
    # V size: 20 x R
    # W size: 20 x R
    
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)

    # Part 1: 4x4x4 multiplication using Strassen (Rank 49)
    # We decompose A (4x4) into 4 blocks of 2x2: A00, A01, A10, A11
    # B (4x4) into 4 blocks of 2x2: B00, B01, B10, B11 (cols 0-3)
    # C (4x4) into 4 blocks of 2x2: C00, C01, C10, C11
    
    # Strassen 2x2 terms
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # Indices for A (4x4):
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    
    # Indices for B (4x5), cols 0-3:
    # B00: 0,1,5,6
    # B01: 2,3,7,8
    # B10: 10,11,15,16
    # B11: 12,13,17,18
    
    # Indices for C (4x5), cols 0-3:
    # C00: 0,1,4,5
    # C01: 2,3,6,7
    # C10: 10,11,14,15
    # C11: 12,13,16,17
    
    # Helper to get indices for a 2x2 block in a 4x4 matrix
    def get_a_indices(r, c):
        # r, c are block indices (0 or 1)
        return [r*4+c*2, r*4+c*2+1, r*4+2+c*2, r*4+2+c*2+1]
        
    def get_b_indices(r, c):
        # r, c are block indices (0 or 1) for cols 0-3
        return [r*5+c*2, r*5+c*2+1, r*5+2+c*2, r*5+2+c*2+1]
        
    def get_c_indices(r, c):
        # r, c are block indices (0 or 1) for cols 0-3
        # C index: k*n + i. k is col (0-3), i is row (0-3).
        # Block (r, c) corresponds to rows 2r..2r+1, cols 2c..2c+1.
        idx = []
        for i in range(2):
            for k in range(2):
                row = 2*r + i
                col = 2*c + k
                idx.append(col*n + row)
        return idx

    # Strassen 2x2 decomposition function
    # Adds 7 terms for multiplying two 2x2 matrices defined by their element indices
    def add_strassen_2x2(a_block_indices, b_block_indices, c_block_indices, a_coeffs, b_coeffs, c_coeffs):
        # a_block_indices: list of 4 indices for A block
        # b_block_indices: list of 4 indices for B block
        # c_block_indices: list of 4 indices for C block
        # a_coeffs: list of 4 coeffs for A block (from the linear form)
        # b_coeffs: list of 4 coeffs for B block
        # c_coeffs: list of 4 coeffs for C block (how M contributes to C)
        
        # This is complex because the "2x2" matrices are sums of blocks.
        # Instead, we expand the Strassen 2x2 into 7 scalar products directly on the 4x4 elements.
        pass

    # Simplified approach:
    # For each of the 7 Strassen terms, we have a 2x2 matrix multiplication.
    # We will implement the 7 scalar products of the 2x2 Strassen scheme.
    
    # Strassen 2x2 scalar products:
    # S0 = (a00+a11)(b00+b11)
    # S1 = (a10+a11)b00
    # S2 = a00(b01-b11)
    # S3 = a11(b10-b00)
    # S4 = (a00+a01)b11
    # S5 = (a10-a00)(b00+b01)
    # S6 = (a01-a11)(b10+b11)
    
    # For a given Strassen term k (0..6) in the 4x4 scheme:
    # The "a" elements are linear combinations of A blocks.
    # The "b" elements are linear combinations of B blocks.
    # We need to add the 7 Strassen scalar products for this pair of 2x2 matrices.
    
    # Let's define the 7 Strassen linear forms for the 4x4 case.
    # For each k in 0..6:
    #   A_k is a 2x2 matrix (sum of A blocks)
    #   B_k is a 2x2 matrix (sum of B blocks)
    #   We need to compute A_k * B_k and add to C blocks.
    
    # We will iterate k=0..6.
    for k in range(7):
        # Determine A_k and B_k block coefficients
        # A_k is a 2x2 matrix of block coefficients
        # B_k is a 2x2 matrix of block coefficients
        
        a_block_coeffs = [[0,0],[0,0]]
        b_block_coeffs = [[0,0],[0,0]]
        
        if k == 0:
            a_block_coeffs = [[1,0],[0,1]] # A00 + A11
            b_block_coeffs = [[1,0],[0,1]] # B00 + B11
        elif k == 1:
            a_block_coeffs = [[0,0],[1,1]] # A10 + A11
            b_block_coeffs = [[1,0],[0,0]] # B00
        elif k == 2:
            a_block_coeffs = [[1,0],[0,0]] # A00
            b_block_coeffs = [[0,1],[0,-1]] # B01 - B11
        elif k == 3:
            a_block_coeffs = [[0,0],[0,1]] # A11
            b_block_coeffs = [[0,-1],[1,0]] # B10 - B00
        elif k == 4:
            a_block_coeffs = [[1,1],[0,0]] # A00 + A01
            b_block_coeffs = [[0,0],[0,1]] # B11
        elif k == 5:
            a_block_coeffs = [[0,-1],[1,0]] # A10 - A00
            b_block_coeffs = [[1,1],[0,0]] # B00 + B01
        elif k == 6:
            a_block_coeffs = [[0,1],[0,-1]] # A01 - A11
            b_block_coeffs = [[0,0],[1,1]] # B10 + B11
            
        # Now we have two 2x2 matrices of block coefficients.
        # We need to compute the product and add to C.
        # The product is a 2x2 matrix of block products.
        # Each block product is a 2x2 matrix multiplication.
        # We use Strassen 2x2 for each block product?
        # No, the "elements" of A_k and B_k are 2x2 blocks of A and B.
        # So A_k[0,0] is a sum of A blocks.
        # We need to multiply these 2x2 matrices.
        # We can use the standard Strassen 2x2 scheme on these 2x2 matrices.
        # This will generate 7 scalar products, where each "scalar" is a 2x2 matrix product.
        # And each 2x2 matrix product is decomposed into 7 scalar products.
        # Total 49 terms.
        
        # To do this, we need to map the 4 "elements" of A_k to the 16 elements of A.
        # A_k[0,0] corresponds to A block 00.
        # A_k[0,1] corresponds to A block 01.
        # A_k[1,0] corresponds to A block 10.
        # A_k[1,1] corresponds to A block 11.
        
        # For each of the 7 Strassen terms of the 2x2 multiplication of A_k and B_k:
        # We need to form the linear forms for A and B.
        
        # Strassen 2x2 terms (S0..S6)
        for s in range(7):
            # Determine which "elements" (blocks) of A_k and B_k are involved
            # and their coefficients.
            a_elem_coeffs = [[0,0],[0,0]]
            b_elem_coeffs = [[0,0],[0,0]]
            
            if s == 0:
                a_elem_coeffs = [[1,0],[0,1]]
                b_elem_coeffs = [[1,0],[0,1]]
            elif s == 1:
                a_elem_coeffs = [[0,0],[1,1]]
                b_elem_coeffs = [[1,0],[0,0]]
            elif s == 2:
                a_elem_coeffs = [[1,0],[0,0]]
                b_elem_coeffs = [[0,1],[0,-1]]
            elif s == 3:
                a_elem_coeffs = [[0,0],[0,1]]
                b_elem_coeffs = [[0,-1],[1,0]]
            elif s == 4:
                a_elem_coeffs = [[1,1],[0,0]]
                b_elem_coeffs = [[0,0],[0,1]]
            elif s == 5:
                a_elem_coeffs = [[0,-1],[1,0]]
                b_elem_coeffs = [[1,1],[0,0]]
            elif s == 6:
                a_elem_coeffs = [[0,1],[0,-1]]
                b_elem_coeffs = [[0,0],[1,1]]
                
            # Now we have linear forms on the 4 blocks of A and B.
            # We need to expand these into linear forms on the 16 elements of A and 16 elements of B (cols 0-3).
            
            # u_vec: 16 elements of A
            u_vec = [0.0] * 16
            # v_vec: 20 elements of B (only 0-3 cols used)
            v_vec = [0.0] * 20
            
            # Map block coefficients to element coefficients
            # Block (br, bc) of A_k corresponds to A block (br, bc)
            # The coefficient is a_elem_coeffs[br][bc] * a_block_coeffs[br][bc]
            # Wait, a_block_coeffs is the coefficient of the A block in A_k.
            # a_elem_coeffs is the coefficient of the A_k element in the Strassen term.
            
            for br in range(2):
                for bc in range(2):
                    coeff = a_elem_coeffs[br][bc] * a_block_coeffs[br][bc]
                    if coeff != 0:
                        indices = get_a_indices(br, bc)
                        for idx in indices:
                            u_vec[idx] += coeff
                    
                    coeff_b = b_elem_coeffs[br][bc] * b_block_coeffs[br][bc]
                    if coeff_b != 0:
                        indices = get_b_indices(br, bc)
                        for idx in indices:
                            v_vec[idx] += coeff_b
            
            # Now we have u_vec and v_vec for this scalar product.
            # We need to add this term to W.
            # The result of this scalar product contributes to the C blocks.
            # The contribution depends on s (which Strassen term) and the block positions.
            # This part is tricky: We need to know how the scalar product S_s contributes to the C blocks.
            # In the 2x2 Strassen scheme, the result C_2x2 is a sum of S_s.
            # C_2x2[0,0] = S0 + S3 - S4 + S6
            # C_2x2[0,1] = S2 + S4
            # C_2x2[1,0] = S1 + S3
            # C_2x2[1,1] = S0 - S1 + S2 + S5
            
            # So for a given s, we know which C_2x2 blocks it contributes to and with what coefficient.
            c_block_contrib = [[0,0],[0,0]]
            if s == 0:
                c_block_contrib = [[1,0],[0,1]] # C00, C11
            elif s == 1:
                c_block_contrib = [[0,0],[1,-1]] # C10, -C11
            elif s == 2:
                c_block_contrib = [[0,1],[0,1]] # C01, C11
            elif s == 3:
                c_block_contrib = [[1,0],[1,0]] # C00, C10
            elif s == 4:
                c_block_contrib = [[-1,1],[0,0]] # -C00, C01
            elif s == 5:
                c_block_contrib = [[0,0],[0,1]] # C11
            elif s == 6:
                c_block_contrib = [[1,0],[0,0]] # C00
            
            # Now we build w_vec.
            w_vec = [0.0] * 20
            for cr in range(2):
                for cc in range(2):
                    coeff = c_block_contrib[cr][cc]
                    if coeff != 0:
                        indices = get_c_indices(cr, cc)
                        for idx in indices:
                            w_vec[idx] += coeff
            
            add_term(u_vec, v_vec, w_vec)

    # Part 2: 4x4x1 multiplication (Schoolbook Rank 16)
    # C[:, 4] = A @ B[:, 4]
    for i in range(4):
        for j in range(4):
            u_vec = [0.0] * 16
            v_vec = [0.0] * 20
            w_vec = [0.0] * 20
            
            u_vec[i*4+j] = 1.0
            v_vec[j*5+4] = 1.0
            w_vec[4*4+i] = 1.0 # k=4, n=4, i=i -> 16+i
            
            add_term(u_vec, v_vec, w_vec)

    # Convert lists to matrices if needed, but return format is list of lists
    # U is list of columns? No, "U": (n*m, rank).
    # My U is list of rows?
    # I appended u_vec (length 16) to U.
    # So U is a list of rank vectors, each of length 16.
    # The required format is (n*m, rank).
    # So I need to transpose U, V, W.
    
    U_mat = [[U[r][i] for r in range(len(U))] for i in range(16)]
    V_mat = [[V[r][i] for r in range(len(V))] for i in range(20)]
    W_mat = [[W[r][i] for r in range(len(W))] for i in range(20)]
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
