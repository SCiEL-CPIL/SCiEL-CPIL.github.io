"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=10
    # Rank = 5 * 7 = 35
    rank = 35
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    # Strassen components for 2x2x2
    # U vectors (coeffs for A00, A01, A10, A11)
    U_strassen = [
        [1, 0, 0, 1],  # M1: A00 + A11
        [0, 0, 1, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [-1, 0, 1, 0], # M6: A10 - A00
        [0, 1, 0, -1]  # M7: A01 - A11
    ]
    
    # V vectors (coeffs for B00, B01, B10, B11)
    V_strassen = [
        [1, 0, 0, 1],  # M1: B00 + B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [0, 0, 1, -1], # M4: B10 - B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00 + B01
        [0, 0, 1, 1]   # M7: B10 + B11
    ]
    
    # W vectors (coeffs for C00, C01, C10, C11)
    # Note: C00, C01, C10, C11 correspond to local indices 0,1,2,3
    W_strassen = [
        [1, 0, 0, 1],   # M1: C00 + C11
        [0, 0, 1, -1],  # M2: C10 - C11
        [0, 1, 0, 1],   # M3: C01 + C11
        [1, 0, 1, 0],   # M4: C00 + C10
        [-1, 1, 0, 0],  # M5: -C00 + C01
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 0]    # M7: C00
    ]
    
    # Fill in the global matrices
    # There are 5 blocks of 2 columns in B (and C)
    num_blocks = p // 2  # 5
    
    for b in range(num_blocks):
        base_col = b * 7
        
        # Global indices for V (B matrix)
        # B is 2x10. Indices j*p + k.
        # Block b uses columns 2b and 2b+1.
        # Local B00 -> j=0, k=2b -> 0*10 + 2b = 2b
        # Local B01 -> j=0, k=2b+1 -> 0*10 + 2b+1 = 2b+1
        # Local B10 -> j=1, k=2b -> 1*10 + 2b = 10+2b
        # Local B11 -> j=1, k=2b+1 -> 1*10 + 2b+1 = 11+2b
        v_indices = [2*b, 2*b+1, 10+2*b, 11+2*b]
        
        # Global indices for W (C matrix)
        # C is 2x10. Indices k*n + i.
        # Block b uses columns 2b and 2b+1.
        # Local C00 -> k=2b, i=0 -> 2b*2 + 0 = 4b
        # Local C01 -> k=2b, i=1 -> 2b*2 + 1 = 4b+1
        # Local C10 -> k=2b+1, i=0 -> (2b+1)*2 + 0 = 4b+2
        # Local C11 -> k=2b+1, i=1 -> (2b+1)*2 + 1 = 4b+3
        w_indices = [4*b, 4*b+1, 4*b+2, 4*b+3]
        
        for t in range(7):
            col = base_col + t
            
            # Fill U
            for i in range(4):
                U[i][col] = U_strassen[t][i]
                
            # Fill V
            for j in range(4):
                V[v_indices[j]][col] = V_strassen[t][j]
                
            # Fill W
            for k in range(4):
                W[w_indices[k]][col] = W_strassen[t][k]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
