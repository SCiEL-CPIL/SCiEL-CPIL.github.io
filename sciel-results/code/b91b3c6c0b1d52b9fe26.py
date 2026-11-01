def construct():
    n, m, p = 4, 4, 4
    
    # Strassen 2x2 coefficients
    # U_2: 7 x 4 (rows are the 7 products, columns are A elements [A00, A01, A10, A11])
    # V_2: 7 x 4 (rows are the 7 products, columns are B elements [B00, B01, B10, B11])
    # W_2: 7 x 4 (rows are the 7 products, columns are C elements [C00, C01, C10, C11])
    
    U_2 = [
        [1, 0, 0, 1],   # M1: (A00+A11)
        [0, 0, 1, 1],   # M2: (A10+A11)
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: (A00+A01)
        [-1, 0, 1, 0],  # M6: (A10-A00)
        [0, 1, 0, -1]   # M7: (A01-A11)
    ]
    V_2 = [
        [1, 0, 0, 1],   # M1: (B00+B11)
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: (B01-B11)
        [-1, 0, 1, 0],  # M4: (B10-B00)
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: (B00+B01)
        [0, 0, 1, 1]    # M7: (B10+B11)
    ]
    W_2 = [
        [1, 0, 0, 1],   # M1: C00 + C11
        [0, 1, 0, -1],  # M2: C10 - C11
        [0, 0, 1, 1],   # M3: C01 + C11
        [1, 1, 0, 0],   # M4: C00 + C10
        [-1, 0, 1, 0],  # M5: -C00 + C01
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 0]    # M7: C00
    ]
    
    # For 4x4, we have 4x4 = 16 elements
    # A elements indexed 0-15 (row-major: A[i][j] = i*4+j)
    # B elements indexed 0-15 (row-major: B[i][j] = i*4+j)
    # C elements indexed 0-15, but W[k*n+i] where n=4, so C[i][j] is at index j*4+i
    
    # Block decomposition: split 4x4 into four 2x2 blocks
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3), B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    # C blocks: C00 (rows 0-1, cols 0-1), C01 (rows 0-1, cols 2-3), C10 (rows 2-3, cols 0-1), C11 (rows 2-3, cols 2-3)
    
    # U: 16 x 49
    # V: 16 x 49
    # W: 16 x 49
    
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    # A block element indices (row-major within 4x4)
    # A00: [0,1,4,5] (A[0][0], A[0][1], A[1][0], A[1][1])
    # A01: [2,3,6,7] (A[0][2], A[0][3], A[1][2], A[1][3])
    # A10: [8,9,12,13] (A[2][0], A[2][1], A[3][0], A[3][1])
    # A11: [10,11,14,15] (A[2][2], A[2][3], A[3][2], A[3][3])
    A_block_idx = [
        [0, 1, 4, 5],   # A00
        [2, 3, 6, 7],   # A01
        [8, 9, 12, 13], # A10
        [10, 11, 14, 15] # A11
    ]
    
    # B block element indices
    B_block_idx = [
        [0, 1, 4, 5],   # B00
        [2, 3, 6, 7],   # B01
        [8, 9, 12, 13], # B10
        [10, 11, 14, 15] # B11
    ]
    
    # C block element indices (W[k*n+i] = W[j*4+i] for C[i][j])
    # C00: C[0][0]=0, C[0][1]=1, C[1][0]=4, C[1][1]=5
    # C01: C[0][2]=8, C[0][3]=9, C[1][2]=12, C[1][3]=13
    # C10: C[2][0]=2, C[2][1]=3, C[3][0]=6, C[3][1]=7
    # C11: C[2][2]=10, C[2][3]=11, C[3][2]=14, C[3][3]=15
    C_block_idx = [
        [0, 1, 4, 5],   # C00
        [8, 9, 12, 13], # C01
        [2, 3, 6, 7],   # C10
        [10, 11, 14, 15] # C11
    ]
    
    # Strassen block combinations
    # For each of the 7 products, we need to specify which A and B blocks are used and how they combine
    # M1: (A00+A11)(B00+B11) -> C00 + C11
    # M2: (A10+A11)B00 -> C10 - C11
    # M3: A00(B01-B11) -> C01 + C11
    # M4: A11(B10-B00) -> C00 + C10
    # M5: (A00+A01)B11 -> -C00 + C01
    # M6: (A10-A00)(B00+B01) -> C11
    # M7: (A01-A11)(B10+B11) -> C00
    
    # For each product k (0-6) and each Strassen term l (0-6):
    # We need to determine the contribution to U, V, W
    
    # The Strassen formula for 4x4:
    # We have 7 "virtual" 2x2 multiplications, each using 7 scalar multiplications
    # So total 49 scalar multiplications
    
    # For virtual product k (0-6), it uses A blocks and B blocks according to Strassen
    # Then each virtual product is decomposed into 7 scalar products
    
    # A block usage for each virtual product k:
    # k=0: A00+A11 -> blocks [0,3] with coeffs [1,1]
    # k=1: A10+A11 -> blocks [2,3] with coeffs [1,1]
    # k=2: A00 -> blocks [0] with coeffs [1]
    # k=3: A11 -> blocks [3] with coeffs [1]
    # k=4: A00+A01 -> blocks [0,1] with coeffs [1,1]
    # k=5: A10-A00 -> blocks [2,0] with coeffs [1,-1]
    # k=6: A01-A11 -> blocks [1,3] with coeffs [1,-1]
    
    A_block_usage = [
        [(0, 1), (3, 1)],  # k=0: A00*1 + A11*1
        [(2, 1), (3, 1)],  # k=1: A10*1 + A11*1
        [(0, 1)],          # k=2: A00*1
        [(3, 1)],          # k=3: A11*1
        [(0, 1), (1, 1)],  # k=4: A00*1 + A01*1
        [(2, 1), (0, -1)], # k=5: A10*1 + A00*(-1)
        [(1, 1), (3, -1)]  # k=6: A01*1 + A11*(-1)
    ]
    
    # B block usage for each virtual product k:
    # k=0: B00+B11 -> blocks [0,3] with coeffs [1,1]
    # k=1: B00 -> blocks [0] with coeffs [1]
    # k=2: B01-B11 -> blocks [1,3] with coeffs [1,-1]
    # k=3: B10-B00 -> blocks [2,0] with coeffs [1,-1]
    # k=4: B11 -> blocks [3] with coeffs [1]
    # k=5: B00+B01 -> blocks [0,1] with coeffs [1,1]
    # k=6: B10+B11 -> blocks [2,3] with coeffs [1,1]
    
    B_block_usage = [
        [(0, 1), (3, 1)],  # k=0
        [(0, 1)],          # k=1
        [(1, 1), (3, -1)], # k=2
        [(2, 1), (0, -1)], # k=3
        [(3, 1)],          # k=4
        [(0, 1), (1, 1)],  # k=5
        [(2, 1), (3, 1)]   # k=6
    ]
    
    # C block usage for each virtual product k:
    # k=0: C00 + C11 -> blocks [0,3] with coeffs [1,1]
    # k=1: C10 - C11 -> blocks [2,3] with coeffs [1,-1]
    # k=2: C01 + C11 -> blocks [1,3] with coeffs [1,1]
    # k=3: C00 + C10 -> blocks [0,2] with coeffs [1,1]
    # k=4: -C00 + C01 -> blocks [0,1] with coeffs [-1,1]
    # k=5: C11 -> blocks [3] with coeffs [1]
    # k=6: C00 -> blocks [0] with coeffs [1]
    
    C_block_usage = [
        [(0, 1), (3, 1)],  # k=0
        [(2, 1), (3, -1)], # k=1
        [(1, 1), (3, 1)],  # k=2
        [(0, 1), (2, 1)],  # k=3
        [(0, -1), (1, 1)], # k=4
        [(3, 1)],          # k=5
        [(0, 1)]           # k=6
    ]
    
    # Now construct U, V, W
    col = 0
    for k in range(7):  # virtual product index
        for l in range(7):  # Strassen term index
            # U contribution
            for (block_idx, block_coeff) in A_block_usage[k]:
                for local_i, local_coeff in enumerate(U_2[l]):
                    if local_coeff != 0:
                        global_idx = A_block_idx[block_idx][local_i]
                        U[global_idx][col] += block_coeff * local_coeff
            
            # V contribution
            for (block_idx, block_coeff) in B_block_usage[k]:
                for local_i, local_coeff in enumerate(V_2[l]):
                    if local_coeff != 0:
                        global_idx = B_block_idx[block_idx][local_i]
                        V[global_idx][col] += block_coeff * local_coeff
            
            # W contribution
            for (block_idx, block_coeff) in C_block_usage[k]:
                for local_i, local_coeff in enumerate(W_2[l]):
                    if local_coeff != 0:
                        global_idx = C_block_idx[block_idx][local_i]
                        W[global_idx][col] += block_coeff * local_coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
