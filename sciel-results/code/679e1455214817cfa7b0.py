import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # We will construct the decomposition for M_{4,4,4} with rank 49
    # and M_{4,4,1} with rank 4.
    
    # Rank-7 decomposition for 2x2x2 (Strassen)
    # U, V, W are 4x7
    U_2x2 = np.array([
        [1, 0, 1, 0, 1, 1, 0], # A11
        [0, 0, 0, 0, 0, 0, -1], # A12? No, let's use the standard form
        # Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Indices for A: 0=A11, 1=A12, 2=A21, 3=A22
        # Indices for B: 0=B11, 1=B12, 2=B21, 3=B22
        # Indices for C: 0=C11, 1=C12, 2=C21, 3=C22
        
        # U (for A)
        [1, 1, 1, 1, 1, -1, 0], # A11
        [0, 0, 0, 0, 1, 0, 1],  # A12
        [0, 1, 0, 0, 0, 1, 0],  # A21
        [1, 1, 0, 1, 0, 0, -1]  # A22
    ])
    
    V_2x2 = np.array([
        [1, 1, 1, 1, 0, 1, 1], # B11
        [1, 0, 1, 0, 0, 1, 0], # B12
        [0, 0, 0, 1, 0, 0, 1], # B21
        [1, 0, -1, -1, 1, 0, 1] # B22
    ])
    
    W_2x2 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C11
        [0, 0, 1, 0, 1, 0, 0],  # C12
        [0, 1, 0, 1, 0, 0, 0],  # C21
        [1, -1, 1, 0, 0, 1, 0]  # C22
    ])
    
    # Verify Strassen
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Now build M_{4,4,4} decomposition
    # A is 4x4, split into 2x2 blocks A11, A12, A21, A22
    # B is 4x4, split into 2x2 blocks B11, B12, B21, B22
    # C is 4x4, split into 2x2 blocks C11, C12, C21, C22
    
    # We have 7 "multiplications" of 2x2 matrices.
    # For each multiplication i, we have 7 rank-1 terms.
    # Total 49 terms.
    
    U_4x4 = []
    V_4x4 = []
    W_4x4 = []
    
    # Map block indices to global indices
    # A: 0..15 (4x4)
    # Block A11: 0,1,4,5
    # Block A12: 2,3,6,7
    # Block A21: 8,9,12,13
    # Block A22: 10,11,14,15
    
    # B: 0..15
    # Block B11: 0,1,4,5
    # Block B12: 2,3,6,7
    # Block B21: 8,9,12,13
    # Block B22: 10,11,14,15
    
    # C: 0..15
    # Block C11: 0,1,4,5
    # Block C12: 2,3,6,7
    # Block C21: 8,9,12,13
    # Block C22: 10,11,14,15
    
    def block_indices(base, block_r, block_c):
        # base is 0 for A, B, C
        # block_r, block_c in 0..1
        # Returns list of 4 indices
        # Row in global: block_r * 2 + r_local
        # Col in global: block_c * 2 + c_local
        # Index = row * 4 + col
        idx = []
        for r in range(2):
            for c in range(2):
                idx.append((block_r * 2 + r) * 4 + (block_c * 2 + c))
        return idx
    
    # Strassen combinations for A blocks
    # M1: A11 + A22
    # M2: A21 + A22
    # M3: A11
    # M4: A22
    # M5: A11 + A12
    # M6: A21 - A11
    # M7: A12 - A22
    
    # Strassen combinations for B blocks
    # M1: B11 + B22
    # M2: B11
    # M3: B12 - B22
    # M4: B21 - B11
    # M5: B22
    # M6: B11 + B12
    # M7: B21 + B22
    
    # We need to expand each of these into the 4 elements of the 2x2 block.
    # Then for each of the 7 "products", we use the 2x2 Strassen decomposition.
    
    # Let's iterate over the 7 Strassen products
    for prod_idx in range(7):
        # Get the linear combination coefficients for A and B blocks
        # We need to map the block-level coefficients to element-level coefficients
        
        # A block coefficients for this product
        # prod_idx 0: M1 -> A11 + A22
        # prod_idx 1: M2 -> A21 + A22
        # prod_idx 2: M3 -> A11
        # prod_idx 3: M4 -> A22
        # prod_idx 4: M5 -> A11 + A12
        # prod_idx 5: M6 -> A21 - A11
        # prod_idx 6: M7 -> A12 - A22
        
        # B block coefficients
        # prod_idx 0: M1 -> B11 + B22
        # prod_idx 1: M2 -> B11
        # prod_idx 2: M3 -> B12 - B22
        # prod_idx 3: M4 -> B21 - B11
        # prod_idx 4: M5 -> B22
        # prod_idx 5: M6 -> B11 + B12
        # prod_idx 6: M7 -> B21 + B22
        
        # C block coefficients (which C block does this product contribute to?)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So for each product i, it contributes to specific C blocks with specific signs.
        
        # Let's define the contribution of product i to each C block
        # c_contrib[block_r, block_c] = coefficient
        
        c_contrib = np.zeros((2, 2))
        if prod_idx == 0: # M1
            c_contrib[0, 0] = 1  # C11
            c_contrib[1, 1] = 1  # C22
        elif prod_idx == 1: # M2
            c_contrib[1, 0] = 1  # C21
            c_contrib[1, 1] = -1 # C22
        elif prod_idx == 2: # M3
            c_contrib[0, 1] = 1  # C12
            c_contrib[1, 1] = 1  # C22
        elif prod_idx == 3: # M4
            c_contrib[0, 0] = 1  # C11
            c_contrib[1, 0] = 1  # C21
        elif prod_idx == 4: # M5
            c_contrib[0, 0] = -1 # C11
            c_contrib[0, 1] = 1  # C12
        elif prod_idx == 5: # M6
            c_contrib[1, 1] = 1  # C22
        elif prod_idx == 6: # M7
            c_contrib[0, 0] = 1  # C11
    
        # Now, for this product, we have a 2x2 matrix multiplication.
        # The "input" A-block is a linear combination of A11, A12, A21, A22.
        # The "input" B-block is a linear combination of B11, B12, B21, B22.
        # We need to determine the coefficients for each element of the 2x2 A-block and B-block.
        
        # A block coefficients (for the 4 elements of the 2x2 block)
        # The 2x2 block is formed by the linear combination of the 4 A sub-blocks.
        # E.g., for M1 (A11 + A22), the 2x2 block is A11 + A22.
        # Element (0,0) of the block is A11[0,0] + A22[0,0].
        
        # Let's define the coefficients for the 4 elements of the A-block
        # a_coeff[4] where 0..3 correspond to the 4 elements of the 2x2 block
        # But the elements come from different global blocks.
        
        # It is easier to iterate over the 7 rank-1 terms of the 2x2 Strassen decomposition.
        for r in range(7):
            # u_r is the coefficient for the 4 elements of the A-block (2x2)
            # v_r is the coefficient for the 4 elements of the B-block (2x2)
            # w_r is the coefficient for the 4 elements of the C-block (2x2)
            
            u_vec = U_2x2[:, r]
            v_vec = V_2x2[:, r]
            w_vec = W_2x2[:, r]
            
            # We need to expand u_vec, v_vec, w_vec to the global 4x4 indices.
            
            # u_vec[0] is coeff for A11 element (0,0) of the block?
            # No, u_vec[0] is coeff for A11 (the whole block)?
            # In the 2x2 Strassen, U_2x2 is 4x7.
            # Row 0 corresponds to A11 (the element (0,0) of the 2x2 matrix).
            # So u_vec[0] is the coefficient for the element (0,0) of the 2x2 A-block.
            
            # The element (0,0) of the 2x2 A-block is a linear combination of global A elements.
            # For M1 (A11 + A22), the 2x2 A-block is A11 + A22.
            # So element (0,0) is A11[0,0] + A22[0,0].
            
            # We need a function to get the global indices for each element of the 2x2 block.
            
            def get_global_indices_for_block_element(block_r, block_c, elem_r, elem_c, prod_idx):
                # This is getting complicated. Let's simplify.
                # We can just build the U, V, W matrices for the 4x4x4 tensor directly.
                
                pass

    # This approach is too complex to implement correctly in the time limit.
    # I will fall back to the schoolbook decomposition, which is guaranteed to be correct.
    # Rank 144.
    
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
