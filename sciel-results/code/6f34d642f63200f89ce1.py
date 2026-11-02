"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=7
    # We use 3 Strassen blocks (2x2x2) and 1 schoolbook block (2x2x1)
    # Rank = 3 * 7 + 4 = 25
    
    rank = 25
    
    U = [[0] * rank for _ in range(n * m)] # 4 x 25
    V = [[0] * rank for _ in range(m * p)] # 14 x 25
    W = [[0] * rank for _ in range(p * n)] # 14 x 25
    
    # Strassen 2x2x2 decomposition
    # U_strassen (4x7), V_strassen (4x7), W_strassen (4x7)
    # Indices for A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for B (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for C (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # Strassen Factors
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    U_s = [
        [1, 0, 1, 0, 1, 1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1], # A01
        [0, 1, 0, 0, 0, 1, 0], # A10
        [1, 1, 0, 1, 0, 0, -1] # A11
    ]
    
    V_s = [
        [1, 1, 1, -1, 0, 1, 1], # B00
        [0, 0, 0, 1, 1, 1, 0], # B01
        [1, 0, 1, 0, 0, 0, 1], # B10
        [1, 0, -1, 0, 1, 0, 1] # B11
    ]
    
    W_s = [
        [1, 0, 0, 0, 0, 0, 0], # C00
        [0, 0, 0, 0, 1, 0, 0], # C01
        [0, 1, 0, 0, 0, 1, 0], # C10
        [0, 0, 0, 1, 0, 0, 1]  # C11
    ]
    # Check W_s:
    # C00 = M1 - M2 - M4 + M7 ?
    # Standard Strassen reconstruction:
    # C00 = M1 - M2 - M4 + M7
    # C01 = M4 + M5
    # C10 = M2 + M3
    # C11 = M1 - M3 - M5 + M6
    
    # Let's re-verify W_s based on reconstruction:
    # C00: M1(1), M2(-1), M4(-1), M7(1) -> [1, -1, 0, -1, 0, 0, 1]
    # C01: M4(1), M5(1) -> [0, 0, 0, 1, 1, 0, 0]
    # C10: M2(1), M3(1) -> [0, 1, 1, 0, 0, 0, 0]
    # C11: M1(1), M3(-1), M5(-1), M6(1) -> [1, 0, -1, 0, -1, 1, 0]
    
    W_s = [
        [1, -1, 0, -1, 0, 0, 1], # C00
        [0, 0, 0, 1, 1, 0, 0], # C01
        [0, 1, 1, 0, 0, 0, 0], # C10
        [1, 0, -1, 0, -1, 1, 0] # C11
    ]
    
    # Schoolbook 2x2x1 decomposition
    # U_sb (4x4), V_sb (2x4), W_sb (2x4)
    # M1 = A00 B0
    # M2 = A01 B1
    # M3 = A10 B0
    # M4 = A11 B1
    
    U_sb = [
        [1, 0, 0, 0], # A00
        [0, 1, 0, 0], # A01
        [0, 0, 1, 0], # A10
        [0, 0, 0, 1]  # A11
    ]
    
    V_sb = [
        [1, 0, 1, 0], # B0
        [0, 1, 0, 1]  # B1
    ]
    
    W_sb = [
        [1, 1, 0, 0], # C0
        [0, 0, 1, 1]  # C1
    ]
    
    # Assemble U, V, W
    # Blocks: 3 Strassen, 1 Schoolbook
    # Column offset for each block
    
    col_offset = 0
    for block_idx in range(3):
        # Strassen block
        for i in range(4):
            for j in range(7):
                U[i][col_offset + j] = U_s[i][j]
        
        # V mapping: B is 2x7. Block 0 uses B cols 0,1. Block 1 uses 2,3. Block 2 uses 4,5.
        # V row indices: 0..13.
        # B00=0, B01=1, B10=2, B11=3
        # B20=4, B21=5, B30=6, B31=7
        # B40=8, B41=9, B50=10, B51=11
        # B60=12, B61=13 (Schoolbook)
        
        base_b_row = block_idx * 4
        for i in range(4):
            for j in range(7):
                V[base_b_row + i][col_offset + j] = V_s[i][j]
        
        # W mapping: C is 7x2. Block 0 uses C rows 0,1. Block 1 uses 2,3. Block 2 uses 4,5.
        # W row indices: 0..13.
        # C00=0, C01=1, C10=2, C11=3
        # C20=4, C21=5, C30=6, C31=7
        # C40=8, C41=9, C50=10, C51=11
        # C60=12, C61=13 (Schoolbook)
        
        base_c_row = block_idx * 4
        for i in range(4):
            for j in range(7):
                W[base_c_row + i][col_offset + j] = W_s[i][j]
        
        col_offset += 7
    
    # Schoolbook block (last column, index 6)
    # U
    for i in range(4):
        for j in range(4):
            U[i][col_offset + j] = U_sb[i][j]
    
    # V: B row 12 (B60), 13 (B61)
    for i in range(2):
        for j in range(4):
            V[12 + i][col_offset + j] = V_sb[i][j]
    
    # W: C row 12 (C60), 13 (C61)
    for i in range(2):
        for j in range(4):
            W[12 + i][col_offset + j] = W_sb[i][j]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
