import numpy as np

def construct():
    # Dimensions
    n, m, p = 4, 4, 8
    
    # Rank for 4x4 Strassen is 49.
    # We need two such blocks for the 4x8 output.
    rank = 98
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank))  # 16 x 98
    V = np.zeros((m * p, rank))  # 32 x 98
    W = np.zeros((p * n, rank))  # 32 x 98
    
    # Define the 2x2x2 Strassen decomposition
    # U2, V2, W2 are 4x7 matrices
    # Indices for 2x2 matrix: 00, 01, 10, 11 -> 0, 1, 2, 3
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Strassen formulas:
    # P0 = (A00 + A11)(B00 + B11)
    # P1 = (A10 + A11)B00
    # P2 = A00(B01 - B11)
    # P3 = A11(B10 - B00)
    # P4 = (A00 + A01)B11
    # P5 = (A10 - A00)(B00 + B01)
    # P6 = (A01 - A11)(B10 + B11)
    
    # Column 0: P0
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # Column 1: P1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # Column 2: P2
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # Column 3: P3
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # Column 4: P4
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # Column 5: P5
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # Column 6: P6
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Function to expand 2x2 Strassen to 4x4 Strassen
    # Block indices for 4x4:
    # 00, 01, 10, 11
    # Each block is 2x2.
    # Global row/col mapping:
    # Block (bi, bj) contains elements (bi*2+r, bj*2+c)
    
    def expand_strassen_4x4(base_U, base_V, base_W):
        """
        Expands 2x2 Strassen (rank 7) to 4x4 Strassen (rank 49).
        Returns U4, V4, W4 of shapes (16, 49), (16, 49), (16, 49).
        Note: Input A is 4x4 (16 elems), Input B is 4x4 (16 elems), Output C is 4x4 (16 elems).
        """
        U4 = np.zeros((16, 49))
        V4 = np.zeros((16, 49))
        W4 = np.zeros((16, 49))
        
        # The 4x4 Strassen uses 7 "super-products" P_k.
        # Each P_k is a product of two 4x4 matrices.
        # The 4x4 matrices for P_k are formed by linear combinations of the 2x2 blocks of A and B.
        
        # Definitions of the 7 linear combinations for A and B blocks (2x2 blocks)
        # A_blocks: 00, 01, 10, 11
        # B_blocks: 00, 01, 10, 11
        
        # P0: (A00+A11)(B00+B11)
        # P1: (A10+A11)B00
        # P2: A00(B01-B11)
        # P3: A11(B10-B00)
        # P4: (A00+A01)B11
        # P5: (A10-A00)(B00+B01)
        # P6: (A01-A11)(B10+B11)
        
        # For each of the 7 super-products, we have a specific linear combo for A blocks and B blocks.
        # Then we use the 2x2 Strassen (base_U, base_V, base_W) to compute the product of the two 4x4 matrices.
        
        A_specs = [
            [(1, 0), (1, 3)],  # P0: A00 + A11
            [(1, 2), (1, 3)],  # P1: A10 + A11
            [(1, 0)],          # P2: A00
            [(1, 3)],          # P3: A11
            [(1, 0), (1, 1)],  # P4: A00 + A01
            [(1, 2), (-1, 0)], # P5: A10 - A00
            [(1, 1), (-1, 3)]  # P6: A01 - A11
        ]
        
        B_specs = [
            [(1, 0), (1, 3)],  # P0: B00 + B11
            [(1, 0)],          # P1: B00
            [(1, 1), (-1, 3)], # P2: B01 - B11
            [(1, 2), (-1, 0)], # P3: B10 - B00
            [(1, 3)],          # P4: B11
            [(1, 0), (1, 1)],  # P5: B00 + B01
            [(1, 2), (1, 3)]   # P6: B10 + B11
        ]
        
        C_specs = [
            [(1, 0), (1, 3)],  # C00 + C11
            [(1, 2), (-1, 3)], # C10 - C11
            [(1, 1), (1, 3)],  # C01 + C11
            [(1, 0), (1, 2)],  # C00 + C10
            [(-1, 0), (1, 1)], # -C00 + C01
            [(1, 3)],          # C11
            [(1, 0)]           # C00
        ]
        
        for k in range(7):
            # Each super-product k uses 7 scalar multiplications from the 2x2 Strassen.
            for m in range(7):
                col_idx = k * 7 + m
                
                # --- U (Input A) ---
                for coeff_a, block_idx in A_specs[k]:
                    bi, bj = divmod(block_idx, 2)
                    # Block (bi, bj) in 4x4 matrix
                    # Global indices for elements in this 2x2 block
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_a_idx = global_r * 4 + global_c
                            
                            # Corresponding element in the effective 2x2 matrix
                            # The effective 2x2 matrix for the 2x2 Strassen
                            # corresponds to the 2x2 block structure?
                            # No, the 2x2 Strassen operates on 2x2 matrices.
                            # Here, the "effective" 2x2 matrix for the Strassen kernel
                            # is formed by the 2x2 blocks of the 4x4 matrix.
                            # So element (r, c) of the 2x2 block (bi, bj)
                            # maps to element (r, c) of the 2x2 kernel matrix?
                            # Wait. The 2x2 kernel matrix indices are 0,1,2,3.
                            # 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1).
                            # So yes, r, c map directly.
                            a_eff_idx = r * 2 + c
                            
                            val = coeff_a * base_U[a_eff_idx, m]
                            if val != 0:
                                U4[global_a_idx, col_idx] += val

                # --- V (Input B) ---
                for coeff_b, block_idx in B_specs[k]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_b_idx = global_r * 4 + global_c
                            
                            b_eff_idx = r * 2 + c
                            
                            val = coeff_b * base_V[b_eff_idx, m]
                            if val != 0:
                                V4[global_b_idx, col_idx] += val

                # --- W (Output C) ---
                for coeff_c, c_block_idx in C_specs[k]:
                    bi, bj = divmod(c_block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_i = bi * 2 + r
                            global_k = bj * 2 + c
                            global_w_idx = global_k * 4 + global_i
                            
                            p_eff_idx = r * 2 + c
                            
                            val = coeff_c * base_W[p_eff_idx, m]
                            if val != 0:
                                W4[global_w_idx, col_idx] += val

        return U4, V4, W4

    # Generate 4x4 Strassen decomposition
    U4, V4, W4 = expand_strassen_4x4(U2, V2, W2)
    
    # Now construct the 4x4 x 4x8 decomposition
    # We have two independent 4x4 multiplications.
    # Left half: B_L (cols 0-3), C_L (cols 0-3)
    # Right half: B_R (cols 4-7), C_R (cols 4-7)
    
    # Copy U4 to U (columns 0-48)
    U[:, 0:49] = U4
    
    # Copy U4 to U (columns 49-97)
    U[:, 49:98] = U4
    
    # V:
    # V4 is 16x49 (for 4x4 B).
    # For the left half, B_L is rows 0-3, cols 0-3 of the 4x8 B matrix.
    # B matrix indices: row * 8 + col.
    # B_L indices: r * 8 + c for r,c in 0..3.
    # V4 indices: r * 4 + c.
    # Mapping: V4[r*4+c] -> V[r*8+c] for the left half.
    
    for r in range(4):
        for c in range(4):
            v_idx_4 = r * 4 + c
            v_idx_8_l = r * 8 + c
            v_idx_8_r = r * 8 + (c + 4)
            
            V[v_idx_8_l, 0:49] = V4[v_idx_4, :]
            V[v_idx_8_r, 49:98] = V4[v_idx_4, :]
    
    # W:
    # W4 is 16x49 (for 4x4 C).
    # W matrix indices: col * 4 + row.
    # C_L indices: (c, r) -> c*4 + r for c,r in 0..3.
    # C_R indices: (c+4, r) -> (c+4)*4 + r for c,r in 0..3.
    
    for c in range(4):
        for r in range(4):
            w_idx_4 = c * 4 + r
            w_idx_8_l = c * 4 + r
            w_idx_8_r = (c + 4) * 4 + r
            
            W[w_idx_8_l, 0:49] = W4[w_idx_4, :]
            W[w_idx_8_r, 49:98] = W4[w_idx_4, :]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
