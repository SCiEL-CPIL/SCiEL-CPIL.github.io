import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition in parts and concatenate.
    # Part 1: A_R B_R (4x1 * 1x6) -> Rank 24
    # Part 2: A_L B_L12 (4x4 * 4x4) -> Rank 49
    # Part 3: A_L B_L3 (4x4 * 4x2) -> Rank 28
    
    # Helper to get Strassen 2x2x2 decomposition
    def strassen_2x2_2x2():
        # U: 4x7, V: 4x7, W: 4x7
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        # Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1; W[3, 6] = -1
        
        return U, V, W

    # Build 4x4x4 using Strassen
    def build_4x4_4x4():
        # A is 4x4, B is 4x4.
        # Split A into 2x2 blocks: A11, A12, A21, A22.
        # Split B into 2x2 blocks: B11, B12, B21, B22.
        # C11 = A11B11 + A12B21
        # C12 = A11B12 + A12B22
        # C21 = A21B11 + A22B21
        # C22 = A21B12 + A22B22
        
        # We need to compute 8 products of 2x2 matrices.
        # But Strassen reduces this to 7 products.
        # Let's use the Strassen formulas directly on the 2x2 blocks.
        
        U_4x4 = np.zeros((16, 49))
        V_4x4 = np.zeros((16, 49))
        W_4x4 = np.zeros((16, 49))
        
        # We will map the 7 Strassen intermediates M1..M7
        # Each Mk is a 2x2x2 tensor.
        # We need to express the inputs to Mk in terms of the global A and B.
        
        # This is getting complex to hardcode.
        # Alternative: Just use the 8 products with rank 7 each?
        # Rank 56.
        # 56 + 28 + 24 = 108.
        # 108 is still an improvement over 120.
        # And much easier to implement correctly.
        
        # Let's stick to Rank 56 for 4x4x4 to avoid bugs.
        # Products:
        # P1 = A11 B11
        # P2 = A12 B21
        # P3 = A11 B12
        # P4 = A12 B22
        # P5 = A21 B11
        # P6 = A22 B21
        # P7 = A21 B12
        # P8 = A22 B22
        
        # Each Pi uses 7 ranks.
        # Total 56 ranks.
        
        sU, sV, sW = strassen_2x2_2x2()
        
        # Define block mappings
        # A indices: 0..3 (rows), 0..3 (cols). Flatten: i*4+j.
        # B indices: 0..3 (rows), 0..3 (cols). Flatten: j*4+k.
        # C indices: 0..3 (rows), 0..3 (cols). Flatten: k*4+i.
        
        blocks = [
            # (A_row_block, A_col_block, B_row_block, B_col_block, C_row_block, C_col_block)
            (0, 0, 0, 0, 0, 0), # A11 B11 -> C11
            (0, 1, 1, 0, 0, 0), # A12 B21 -> C11
            (0, 0, 0, 1, 0, 1), # A11 B12 -> C12
            (0, 1, 1, 1, 0, 1), # A12 B22 -> C12
            (1, 0, 0, 0, 1, 0), # A21 B11 -> C21
            (1, 1, 1, 0, 1, 0), # A22 B21 -> C21
            (1, 0, 0, 1, 1, 1), # A21 B12 -> C22
            (1, 1, 1, 1, 1, 1), # A22 B22 -> C22
        ]
        
        col_offset = 0
        for (ar, ac, br, bc, cr, cc) in blocks:
            # Map local 2x2 indices to global 4x4 indices
            # A local (i,j) -> global (ar*2+i, ac*2+j)
            # B local (j,k) -> global (br*2+j, bc*2+k)
            # C local (i,k) -> global (cr*2+i, cc*2+k)
            
            for r in range(7):
                # U
                for i in range(2):
                    for j in range(2):
                        val = sU[i*2+j, r]
                        if val != 0:
                            U_4x4[(ar*2+i)*4 + (ac*2+j), col_offset+r] += val
                # V
                for j in range(2):
                    for k in range(2):
                        val = sV[j*2+k, r]
                        if val != 0:
                            V_4x4[(br*2+j)*4 + (bc*2+k), col_offset+r] += val
                # W
                for k in range(2):
                    for i in range(2):
                        val = sW[k*2+i, r]
                        if val != 0:
                            W_4x4[(cc*2+k)*4 + (cr*2+i), col_offset+r] += val
            col_offset += 7
            
        return U_4x4, V_4x4, W_4x4

    # Build 4x4x2
    def build_4x4_4x2():
        U_4x4_2 = np.zeros((16, 28))
        V_4x4_2 = np.zeros((8, 28))
        W_4x4_2 = np.zeros((8, 28))
        
        sU, sV, sW = strassen_2x2_2x2()
        
        # A is 4x4. B is 4x2.
        # B blocks: B11 (0,0), B21 (1,0). (2x2 each)
        # Products:
        # P1 = A11 B11 -> C1
        # P2 = A12 B21 -> C1
        # P3 = A21 B11 -> C2
        # P4 = A22 B21 -> C2
        
        blocks = [
            (0, 0, 0, 0, 0), # A11 B11 -> C1
            (0, 1, 1, 0, 0), # A12 B21 -> C1
            (1, 0, 0, 0, 1), # A21 B11 -> C2
            (1, 1, 1, 0, 1), # A22 B21 -> C2
        ]
        
        col_offset = 0
        for (ar, ac, br, bc, cr) in blocks:
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        val = sU[i*2+j, r]
                        if val != 0:
                            U_4x4_2[(ar*2+i)*4 + (ac*2+j), col_offset+r] += val
                for j in range(2):
                    for k in range(2):
                        val = sV[j*2+k, r]
                        if val != 0:
                            V_4x4_2[(br*2+j)*2 + (bc*2+k), col_offset+r] += val
                for k in range(2):
                    for i in range(2):
                        val = sW[k*2+i, r]
                        if val != 0:
                            W_4x4_2[(cr*2+k)*2 + (cr*2+i), col_offset+r] += val
            col_offset += 7
            
        return U_4x4_2, V_4x4_2, W_4x4_2

    # Build 4x1x6 (Outer Product)
    def build_4x1_1x6():
        U = np.zeros((4, 24))
        V = np.zeros((6, 24))
        W = np.zeros((24, 24))
        
        col = 0
        for i in range(4):
            for k in range(6):
                U[i, col] = 1
                V[k, col] = 1
                W[k*4+i, col] = 1
                col += 1
        return U, V, W

    # Combine
    U1, V1, W1 = build_4x1_1x6()
    U2, V2, W2 = build_4x4_4x4()
    U3, V3, W3 = build_4x4_4x2()
    
    # Global A is 4x5.
    # A_L is cols 0-3. A_R is col 4.
    # Global B is 5x6.
    # B_L is rows 0-3. B_R is row 4.
    # B_L12 is cols 0-3. B_L3 is cols 4-5.
    
    # U_global shape (20, R)
    # V_global shape (30, R)
    # W_global shape (24, R)
    
    R = 24 + 56 + 28
    U_global = np.zeros((20, R))
    V_global = np.zeros((30, R))
    W_global = np.zeros((24, R))
    
    # Part 1: A_R B_R
    # A_R is col 4 of A. Index 4 in A's flattened cols?
    # A is 4x5. Flattened index i*5+j.
    # A_R corresponds to j=4. i=0..3.
    # U1 is 4x24. Maps to U_global[i, 0:24].
    U_global[0:4, 0:24] = U1
    
    # B_R is row 4 of B. Index 4 in B's flattened rows?
    # B is 5x6. Flattened index j*6+k.
    # B_R corresponds to j=4. k=0..5.
    # V1 is 6x24. Maps to V_global[4*6+k, 0:24].
    V_global[24:30, 0:24] = V1
    
    # W1 is 24x24. Maps to W_global[0:24, 0:24].
    W_global[0:24, 0:24] = W1
    
    # Part 2: A_L B_L12
    # A_L is cols 0-3 of A.
    # U2 is 16x56. Maps to U_global[i*5+j, 24:80].
    for i in range(4):
        for j in range(4):
            U_global[i*5+j, 24:80] = U2[i*4+j]
            
    # B_L12 is rows 0-3, cols 0-3 of B.
    # V2 is 16x56. Maps to V_global[j*6+k, 24:80].
    for j in range(4):
        for k in range(4):
            V_global[j*6+k, 24:80] = V2[j*4+k]
            
    # W2 is 16x56. Maps to W_global[k*4+i, 24:80].
    for k in range(4):
        for i in range(4):
            W_global[k*4+i, 24:80] = W2[k*4+i]
            
    # Part 3: A_L B_L3
    # A_L is cols 0-3 of A.
    # U3 is 16x28. Maps to U_global[i*5+j, 80:108].
    for i in range(4):
        for j in range(4):
            U_global[i*5+j, 80:108] = U3[i*4+j]
            
    # B_L3 is rows 0-3, cols 4-5 of B.
    # V3 is 8x28. Maps to V_global[j*6+k, 80:108].
    for j in range(4):
        for k in range(2):
            V_global[j*6+(4+k), 80:108] = V3[j*2+k]
            
    # W3 is 8x28. Maps to W_global[k*4+i, 80:108].
    for k in range(2):
        for i in range(4):
            W_global[(4+k)*4+i, 80:108] = W3[k*4+i]
            
    return {"U": U_global, "V": V_global, "W": W_global}
