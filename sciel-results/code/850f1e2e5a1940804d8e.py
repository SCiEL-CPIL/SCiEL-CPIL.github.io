import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Strassen decomposition for 2x2x2 (rank 7)
    # U, V, W are 4x7, 4x7, 4x7
    # Indices: A is 2x2 (0,1,2,3), B is 2x2 (0,1,2,3), C is 2x2 (0,1,2,3)
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    
    # Strassen M1..M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11) B00
    # M3 = A00 (B01-B11)
    # M4 = A11 (B10-B00)
    # M5 = (A00+A01) B11
    # M6 = (A10-A00) (B00+B01)
    # M7 = (A01-A11) (B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, -1], # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 1], # B00
        [0, 0, 1, 0, 1, 1, 0],  # B01
        [1, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # 2. 4x4x4 decomposition using 8 independent 2x2x2 Strassen blocks (rank 56)
    # A is 4x4, B is 4x4, C is 4x4
    # A blocks: A11, A12, A21, A22 (each 2x2)
    # B blocks: B11, B12, B21, B22 (each 2x2)
    # C blocks: C11 = A11 B11 + A12 B21
    #           C12 = A11 B12 + A12 B22
    #           C21 = A21 B11 + A22 B21
    #           C22 = A21 B12 + A22 B22
    
    # We compute 8 products:
    # P1 = A11 B11 (rank 7)
    # P2 = A12 B21 (rank 7)
    # P3 = A11 B12 (rank 7)
    # P4 = A12 B22 (rank 7)
    # P5 = A21 B11 (rank 7)
    # P6 = A22 B21 (rank 7)
    # P7 = A21 B12 (rank 7)
    # P8 = A22 B22 (rank 7)
    
    # Total rank 56.
    
    U_444 = []
    V_444 = []
    W_444 = []
    
    # Helper to add a Strassen block for a specific A block, B block, C block
    def add_strassen_block(U, V, W, a_block_idx, b_block_idx, c_block_idx, sign=1):
        # a_block_idx: 0=A11, 1=A12, 2=A21, 3=A22
        # b_block_idx: 0=B11, 1=B12, 2=B21, 3=B22
        # c_block_idx: 0=C11, 1=C12, 2=C21, 3=C22
        
        # Map block index to row/col offset in 4x4 matrix
        # A is row-major: A[i][j] = A[i*4+j]
        # A11 is rows 0-1, cols 0-1. Offset 0.
        # A12 is rows 0-1, cols 2-3. Offset 2.
        # A21 is rows 2-3, cols 0-1. Offset 4.
        # A22 is rows 2-3, cols 2-3. Offset 6.
        
        a_offsets = [0, 2, 4, 6]
        b_offsets = [0, 2, 4, 6]
        c_offsets = [0, 2, 4, 6]
        
        a_off = a_offsets[a_block_idx]
        b_off = b_offsets[b_block_idx]
        c_off = c_offsets[c_block_idx]
        
        for r in range(4):
            u_row = U[r * 4 + (a_off + r // 2) * 2 + (r % 2)] if False else None # This logic is messy
            
    # Let's just build the full U, V, W for 4x4x4 by iterating over the 8 products
    # U has 16 rows (4x4), V has 16 rows (4x4), W has 16 rows (4x4)
    # Each Strassen block contributes 7 columns.
    
    U_444 = [[0]*56 for _ in range(16)]
    V_444 = [[0]*56 for _ in range(16)]
    W_444 = [[0]*56 for _ in range(16)]
    
    products = [
        (0, 2, 0, 1), # A11 B21 -> C11
        (0, 0, 0, 1), # A11 B11 -> C11
        (1, 3, 1, 1), # A12 B22 -> C12
        (1, 1, 1, 1), # A12 B12 -> C12
        (2, 2, 2, 1), # A21 B21 -> C21
        (2, 0, 2, 1), # A21 B11 -> C21
        (3, 3, 3, 1), # A22 B22 -> C22
        (3, 1, 3, 1)  # A22 B12 -> C22
    ]
    
    # Wait, C11 = A11 B11 + A12 B21.
    # So we need A11 B11 and A12 B21.
    # My list above has (0,2,0) which is A11 B21? No, b_block_idx 2 is B21.
    # A11 is idx 0. B21 is idx 2. C11 is idx 0.
    # So (0, 2, 0) is correct for A11 B21.
    # And (0, 0, 0) is A11 B11.
    
    for i, (a_idx, b_idx, c_idx, sign) in enumerate(products):
        col_offset = i * 7
        # Add Strassen U, V, W to the appropriate rows
        for r in range(4):
            # Map Strassen row r to 4x4 row
            # Strassen A is 2x2.
            # If a_idx=0 (A11), rows 0,1 of 4x4.
            # If a_idx=1 (A12), rows 0,1 of 4x4.
            # If a_idx=2 (A21), rows 2,3 of 4x4.
            # If a_idx=3 (A22), rows 2,3 of 4x4.
            
            row_base = (a_idx // 2) * 2
            col_base = (a_idx % 2) * 2
            
            u_row = row_base + (r // 2)
            u_col = col_base + (r % 2)
            
            U_444[u_row * 4 + u_col][col_offset + r] = sign * U_strassen[r][0] if False else None # U_strassen is 4x7
            
        # Correctly populate U_444
        for r in range(4):
            row_base = (a_idx // 2) * 2
            col_base = (a_idx % 2) * 2
            u_row = row_base + (r // 2)
            u_col = col_base + (r % 2)
            for c in range(7):
                U_444[u_row * 4 + u_col][col_offset + c] += sign * U_strassen[r][c]
                
        for r in range(4):
            row_base = (b_idx // 2) * 2
            col_base = (b_idx % 2) * 2
            v_row = row_base + (r // 2)
            v_col = col_base + (r % 2)
            for c in range(7):
                V_444[v_row * 4 + v_col][col_offset + c] += sign * V_strassen[r][c]
                
        for r in range(4):
            row_base = (c_idx // 2) * 2
            col_base = (c_idx % 2) * 2
            w_row = row_base + (r // 2)
            w_col = col_base + (r % 2)
            for c in range(7):
                W_444[w_row * 4 + w_col][col_offset + c] += sign * W_strassen[r][c]

    # 3. 4x4x1 schoolbook (rank 16)
    # A is 4x4, B is 4x1, C is 4x1
    U_441 = [[0]*16 for _ in range(16)]
    V_441 = [[0]*16 for _ in range(4)]
    W_441 = [[0]*16 for _ in range(4)]
    
    for i in range(4):
        for j in range(4):
            col = i * 4 + j
            U_441[i * 4 + j][col] = 1
            V_441[j][col] = 1
            W_441[i][col] = 1
            
    # 4. Combine for 4x4x9
    # B is 4x9. Split into B1 (4x4), B2 (4x4), B3 (4x1).
    # C is 4x9. C1 (4x4), C2 (4x4), C3 (4x1).
    
    total_rank = 56 + 56 + 16
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(36)]
    W = [[0]*total_rank for _ in range(36)]
    
    # Part 1: A B1 -> C1
    # A is 4x4 (16 rows in U)
    # B1 is cols 0-3 of B (4x4). V rows 0-15.
    # C1 is cols 0-3 of C (4x4). W rows 0-15.
    
    for r in range(16):
        for c in range(56):
            U[r][c] = U_444[r][c]
            V[r][c] = V_444[r][c]
            W[r][c] = W_444[r][c]
            
    # Part 2: A B2 -> C2
    # B2 is cols 4-7 of B. V rows 16-31.
    # C2 is cols 4-7 of C. W rows 16-31.
    
    for r in range(16):
        for c in range(56):
            U[r][56 + c] = U_444[r][c]
            V[16 + r][56 + c] = V_444[r][c]
            W[16 + r][56 + c] = W_444[r][c]
            
    # Part 3: A B3 -> C3
    # B3 is cols 8 of B (4x1). V rows 32-35.
    # C3 is cols 8 of C (4x1). W rows 32-35.
    
    for r in range(16):
        for c in range(16):
            U[r][112 + c] = U_441[r][c]
            V[32 + (r % 4)][112 + c] = V_441[r % 4][c] # V_441 has 4 rows
            W[32 + (r // 4)][112 + c] = W_441[r // 4][c] # W_441 has 4 rows
            
    return {"U": U, "V": V, "W": W}
