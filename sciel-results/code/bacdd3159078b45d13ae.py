# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    Theoretical Lower Bound:
    The tensor decomposes into three independent 2x2*2x2 blocks and one 2x2*2x1 block.
    Rank(2x2*2x2) = 7 (Strassen)
    Rank(2x2*2x1) = 4 (Schoolbook)
    Total Rank = 3 * 7 + 4 = 25.
    
    This program constructs this decomposition programmatically.
    """
    n, m, p = 2, 2, 7
    
    # Define the Strassen decomposition for 2x2 * 2x2
    # U: A coeffs (4x7), V: B coeffs (4x7), W: C coeffs (4x7)
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    strassen_U = [
        [1, 0, 0, 1],  # M1: A00 + A11
        [0, 0, 1, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [-1, 0, 1, 0], # M6: -A00 + A10
        [0, 1, 0, -1]  # M7: A01 - A11
    ]
    
    strassen_V = [
        [1, 0, 0, 1],  # M1: B00 + B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [0, 0, 1, -1], # M4: B10 - B00 (Note: standard Strassen often uses B10-B00 or similar, let's stick to the valid set)
        # Let's use the standard Strassen set that guarantees rank 7.
        # M1: (A00+A11)(B00+B11)
        # M2: (A10+A11)B00
        # M3: A00(B01-B11)
        # M4: A11(B10-B00)
        # M5: (A00+A01)B11
        # M6: (A10-A00)B01
        # M7: (A11-A01)B10
        
        # Re-evaluating V to match the U/W definitions above for consistency.
        # U M1: A00+A11. V M1: B00+B11. W M1: C00+C11.
        # U M2: A10+A11. V M2: B00. W M2: C10-C11.
        # U M3: A00. V M3: B01-B11. W M3: C01+C11.
        # U M4: A11. V M4: B10-B00. W M4: C00+C10.
        # U M5: A00+A01. V M5: B11. W M5: -C00+C01.
        # U M6: -A00+A10. V M6: B00+B01. W M6: C11.
        # U M7: A01-A11. V M7: B10+B11. W M7: C00.
        
        # Let's verify M6:
        # U6 = -A00 + A10. V6 = B00 + B01.
        # Product = -A00B00 - A00B01 + A10B00 + A10B01.
        # W6 = C11.
        # C11 = A00B01 + A01B00 + A10B01 + A11B11 ? No, C11 = A00B11 + A01B10 + A10B01 + A11B00?
        # Standard C = AB:
        # C00 = A00B00 + A01B10
        # C01 = A00B01 + A01B11
        # C10 = A10B00 + A11B10
        # C11 = A10B01 + A11B11
        
        # Let's check the provided Strassen in the previous code to ensure it was correct.
        # The previous code had:
        # V M6: B00 + B01. W M6: C11.
        # Let's check if the sum of products works.
        # This is a known valid Strassen decomposition. I will reuse the exact coefficients 
        # from the previous correct solution to ensure validity, but structured differently.
    ]
    
    # To be safe and strictly correct, I will define the matrices explicitly as in the known-good solution
    # but wrap them in a helper structure.
    
    # Correct Strassen Coefficients (verified in previous run)
    S_U = [
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1], 
        [0, 0, 1, -1], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]
    ]
    # W coefficients are defined as contributions to C indices.
    # C indices: 0:C00, 1:C01, 2:C10, 3:C11
    S_W = [
        {0: 1, 3: 1},     # M1
        {2: 1, 3: -1},    # M2
        {1: 1, 3: 1},     # M3
        {0: 1, 2: 1},     # M4
        {0: -1, 1: 1},    # M5
        {3: 1},           # M6
        {0: 1}            # M7
    ]
    
    # Schoolbook for 2x2 * 2x1 (last column)
    # A: 2x2, B: 2x1, C: 2x1
    # C0 = A00B0 + A01B1
    # C1 = A10B0 + A11B1
    # U (A): 4 rows (A00, A01, A10, A11)
    # V (B): 2 rows (B0, B1)
    # W (C): 2 rows (C0, C1)
    
    SB_U = [
        [1, 0, 0, 0], # A00
        [0, 1, 0, 0], # A01
        [0, 0, 1, 0], # A10
        [0, 0, 0, 1]  # A11
    ]
    SB_V = [
        [1, 0], # B0
        [0, 1], # B1
        [1, 0], # B0
        [0, 1]  # B1
    ]
    SB_W = [
        {0: 1}, # C0
        {0: 1}, # C0
        {1: 1}, # C1
        {1: 1}  # C1
    ]
    
    rank = 3 * 7 + 4 # 25
    
    # Initialize matrices
    # U shape: (n*m, rank) = (4, 25)
    # V shape: (m*p, rank) = (14, 25)
    # W shape: (p*n, rank) = (14, 25)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Place 3 Strassen blocks
    for b in range(3):
        for r in range(7):
            # Fill U
            for i in range(4):
                U[i][col] += S_U[r][i]
            
            # Fill V
            # B local indices: 0:B00, 1:B01, 2:B10, 3:B11
            # Global V index = j*p + k
            # Block b corresponds to k = 2b, 2b+1
            # Local 0 (B00) -> j=0, k=2b -> V[0*p + 2b]
            # Local 1 (B01) -> j=0, k=2b+1 -> V[0*p + 2b+1]
            # Local 2 (B10) -> j=1, k=2b -> V[1*p + 2b]
            # Local 3 (B11) -> j=1, k=2b+1 -> V[1*p + 2b+1]
            
            v_coeffs = S_V[r]
            for loc_idx in range(4):
                coeff = v_coeffs[loc_idx]
                if coeff != 0:
                    if loc_idx == 0:
                        v_idx = 0 * p + (2 * b)
                    elif loc_idx == 1:
                        v_idx = 0 * p + (2 * b + 1)
                    elif loc_idx == 2:
                        v_idx = 1 * p + (2 * b)
                    else: # loc_idx == 3
                        v_idx = 1 * p + (2 * b + 1)
                    V[v_idx][col] += coeff
            
            # Fill W
            # C local indices: 0:C00, 1:C01, 2:C10, 3:C11
            # Global W index = k*n + i
            # Block b corresponds to k = 2b, 2b+1
            # Local 0 (C00) -> k=2b, i=0 -> W[2b*n + 0]
            # Local 1 (C01) -> k=2b+1, i=0 -> W[(2b+1)*n + 0]
            # Local 2 (C10) -> k=2b, i=1 -> W[2b*n + 1]
            # Local 3 (C11) -> k=2b+1, i=1 -> W[(2b+1)*n + 1]
            
            for loc_idx, coeff in S_W[r].items():
                if loc_idx == 0:
                    w_idx = (2 * b) * n + 0
                elif loc_idx == 1:
                    w_idx = (2 * b + 1) * n + 0
                elif loc_idx == 2:
                    w_idx = (2 * b) * n + 1
                else: # loc_idx == 3
                    w_idx = (2 * b + 1) * n + 1
                W[w_idx][col] += coeff
            
            col += 1
            
    # Place 1 Schoolbook block (last column, k=6)
    for r in range(4):
        # Fill U
        for i in range(4):
            U[i][col] += SB_U[r][i]
        
        # Fill V
        # B local indices: 0:B0 (row 0, col 6), 1:B1 (row 1, col 6)
        # Global V index = j*p + k
        # Loc 0 -> j=0, k=6 -> V[0*p + 6]
        # Loc 1 -> j=1, k=6 -> V[1*p + 6]
        
        v_coeffs = SB_V[r]
        for loc_idx in range(2):
            coeff = v_coeffs[loc_idx]
            if coeff != 0:
                if loc_idx == 0:
                    v_idx = 0 * p + 6
                else:
                    v_idx = 1 * p + 6
                V[v_idx][col] += coeff
        
        # Fill W
        # C local indices: 0:C0 (row 0, col 6), 1:C1 (row 1, col 6)
        # Global W index = k*n + i
        # Loc 0 -> k=6, i=0 -> W[6*n + 0]
        # Loc 1 -> k=6, i=1 -> W[6*n + 1]
        
        for loc_idx, coeff in SB_W[r].items():
            if loc_idx == 0:
                w_idx = 6 * n + 0
            else:
                w_idx = 6 * n + 1
            W[w_idx][col] += coeff
        
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
