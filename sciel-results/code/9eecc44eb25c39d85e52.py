import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Strassen 2x2x2 decomposition
    # U, V, W shapes (4, 7)
    U22 = np.zeros((4, 7))
    V22 = np.zeros((4, 7))
    W22 = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U22[0, 0] = 1; U22[3, 0] = 1
    V22[0, 0] = 1; V22[3, 0] = 1
    W22[0, 0] = 1; W22[1, 0] = 1; W22[2, 0] = 1; W22[3, 0] = 1 # C11, C12, C21, C22 contributions? 
    # Wait, W maps to C.
    # C11 gets M1. C12 gets 0. C21 gets 0. C22 gets -M1? No.
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M7
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's fill W22 based on these formulas.
    # Col 0 is M1.
    W22[0, 0] = 1   # C11
    W22[1, 0] = 0   # C12
    W22[2, 0] = 0   # C21
    W22[3, 0] = 1   # C22
    
    # Col 1 is M2 = (A21+A22)B11
    U22[2, 1] = 1; U22[3, 1] = 1
    V22[0, 1] = 1
    W22[0, 1] = 0
    W22[1, 1] = 0
    W22[2, 1] = 1   # C21
    W22[3, 1] = -1  # C22
    
    # Col 2 is M3 = A11 B22
    U22[0, 2] = 1
    V22[3, 2] = 1
    W22[0, 2] = 0
    W22[1, 2] = 1   # C12
    W22[2, 2] = 0
    W22[3, 2] = 1   # C22
    
    # Col 3 is M4 = A22 (B11-B22)
    U22[3, 3] = 1
    V22[0, 3] = 1; V22[3, 3] = -1
    W22[0, 3] = -1  # C11
    W22[1, 3] = 0
    W22[2, 3] = 1   # C21
    W22[3, 3] = 0
    
    # Col 4 is M5 = (A11+A12) B22
    U22[0, 4] = 1; U22[1, 4] = 1
    V22[3, 4] = 1
    W22[0, 4] = 1   # C11
    W22[1, 4] = 0
    W22[2, 4] = 0
    W22[3, 4] = 0
    
    # Col 5 is M6 = (A21-A11) (B11+B22)
    U22[2, 5] = 1; U22[0, 5] = -1
    V22[0, 5] = 1; V22[3, 5] = 1
    W22[0, 5] = 0
    W22[1, 5] = 0
    W22[2, 5] = 0
    W22[3, 5] = 1   # C22
    
    # Col 6 is M7 = (A12-A22) (B11+B22)
    U22[1, 6] = 1; U22[3, 6] = -1
    V22[0, 6] = 1; V22[3, 6] = 1
    W22[0, 6] = -1  # C11
    W22[1, 6] = 1   # C12
    W22[2, 6] = 0
    W22[3, 6] = 0

    # 2. Construct 4x4x4 Strassen from 2x2x2
    # U44, V44, W44 shapes (16, 49)
    U44 = np.zeros((16, 49))
    V44 = np.zeros((16, 49))
    W44 = np.zeros((16, 49))
    
    # 7 blocks. Each block uses 7 columns of the 49.
    # Block k corresponds to M_k in 4x4 Strassen.
    # We need to map the 2x2 indices to 4x4 indices.
    # A is 4x4. Blocks A11, A12, A21, A22.
    # A11 is rows 0,1 cols 0,1.
    # A12 is rows 0,1 cols 2,3.
    # A21 is rows 2,3 cols 0,1.
    # A22 is rows 2,3 cols 2,3.
    
    # V is for B. B is 4x4.
    # B11 is rows 0,1 cols 0,1.
    # B12 is rows 0,1 cols 2,3.
    # B21 is rows 2,3 cols 0,1.
    # B22 is rows 2,3 cols 2,3.
    
    # W is for C. C is 4x4.
    # C11, C12, C21, C22.
    
    # We define the 7 products M1..M7 for 4x4.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11 B22
    # M4 = A22 (B11-B22)
    # M5 = (A11+A12) B22
    # M6 = (A21-A11) (B11+B22)
    # M7 = (A12-A22) (B11+B22)
    
    # For each M_k, we instantiate the 2x2x2 decomposition.
    # The U part of 2x2x2 is 4x7. We need to place it in the 16x49 U44.
    # The rows of U22 correspond to A11, A12, A21, A22 (flattened 2x2).
    # Actually, U22 rows are 0,1,2,3 corresponding to A[0,0], A[0,1], A[1,0], A[1,1].
    # In 4x4, A11 is A[0:2, 0:2].
    # So U22 row 0 maps to U44 row 0.
    # U22 row 1 maps to U44 row 1.
    # U22 row 2 maps to U44 row 2.
    # U22 row 3 maps to U44 row 3.
    # But A11 is not the only block.
    # M1 uses A11 and A22.
    # So for M1, U22 row 0 (A11_00) maps to A[0,0] which is index 0.
    # U22 row 1 (A11_01) maps to A[0,1] which is index 1.
    # U22 row 2 (A11_10) maps to A[1,0] which is index 2.
    # U22 row 3 (A11_11) maps to A[1,1] which is index 3.
    # AND
    # U22 row 0 also has contribution from A22?
    # In 2x2 Strassen, U22[0,0]=1 means A11_00 is used.
    # In 4x4 Strassen M1, we use (A11+A22).
    # So A11_00 and A22_00 are both used.
    # A22_00 is A[2,2] which is index 8.
    # So U44[0, col] += 1 and U44[8, col] += 1.
    
    # We need a mapping function.
    # def map_2x2_to_4x4(block_type, local_idx):
    #    # block_type: 0=A11, 1=A12, 2=A21, 3=A22
    #    # local_idx: 0..3 (row-major 2x2)
    #    # returns global index in 16
    
    def idx_4x4(block, local):
        # block 0: rows 0,1 cols 0,1 -> base 0
        # block 1: rows 0,1 cols 2,3 -> base 2
        # block 2: rows 2,3 cols 0,1 -> base 8
        # block 3: rows 2,3 cols 2,3 -> base 10
        # local: 0,1,2,3
        # row = local // 2, col = local % 2
        r = local // 2
        c = local % 2
        if block == 0: return r * 4 + c
        if block == 1: return r * 4 + (c + 2)
        if block == 2: return (r + 2) * 4 + c
        if block == 3: return (r + 2) * 4 + (c + 2)

    # Define the 7 M_k terms for 4x4
    # Each term is a list of (U_block, U_local, coeff), (V_block, V_local, coeff), (W_block, W_local, coeff)
    # Actually, we can just copy the U22, V22, W22 structure and map indices.
    
    # M1: (A11+A22)(B11+B22)
    # U: A11 and A22.
    # V: B11 and B22.
    # W: C11, C22.
    
    terms = []
    
    # Helper to add a term
    def add_term(k, u_blocks, v_blocks, w_blocks):
        # u_blocks: list of (block, coeff)
        # v_blocks: list of (block, coeff)
        # w_blocks: list of (block, coeff)
        base_col = k * 7
        for i in range(7):
            col = base_col + i
            # U
            for (block, coeff) in u_blocks:
                for local in range(4):
                    u_val = U22[local, i]
                    if u_val != 0:
                        g_idx = idx_4x4(block, local)
                        U44[g_idx, col] += u_val * coeff
            # V
            for (block, coeff) in v_blocks:
                for local in range(4):
                    v_val = V22[local, i]
                    if v_val != 0:
                        g_idx = idx_4x4(block, local)
                        V44[g_idx, col] += v_val * coeff
            # W
            for (block, coeff) in w_blocks:
                for local in range(4):
                    w_val = W22[local, i]
                    if w_val != 0:
                        g_idx = idx_4x4(block, local)
                        W44[g_idx, col] += w_val * coeff

    # M1
    add_term(0, [(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])
    # M2
    add_term(1, [(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)])
    # M3
    add_term(2, [(0, 1)], [(3, 1)], [(1, 1), (3, 1)])
    # M4
    add_term(3, [(3, 1)], [(0, 1), (3, -1)], [(0, -1), (2, 1)])
    # M5
    add_term(4, [(0, 1), (1, 1)], [(3, 1)], [(0, 1)])
    # M6
    add_term(5, [(2, 1), (0, -1)], [(0, 1), (3, 1)], [(3, 1)])
    # M7
    add_term(6, [(1, 1), (3, -1)], [(0, 1), (3, 1)], [(0, -1), (1, 1)])

    # 3. Schoolbook 4x4x1
    U441 = np.zeros((16, 16))
    V441 = np.zeros((4, 16))
    W441 = np.zeros((4, 16))
    for i in range(4):
        for j in range(4):
            col = i * 4 + j
            U441[i * 4 + j, col] = 1
            V441[j, col] = 1
            W441[i, col] = 1

    # 4. Combine for 4x4x9
    # Total rank = 49 + 49 + 16 = 114
    rank = 114
    U_final = np.zeros((16, rank))
    V_final = np.zeros((36, rank))
    W_final = np.zeros((36, rank))
    
    # Part 1: A B_a (cols 0-3 of B)
    # U: same as U44
    U_final[:, 0:49] = U44
    # V: B_a is cols 0-3 of B. B is 4x9.
    # V44 is 16x49. Rows correspond to B indices.
    # B_a rows 0-3, cols 0-3.
    # In V_final, rows 0-15 correspond to B[0:4, 0:4].
    # So V_final[0:16, 0:49] = V44.
    V_final[0:16, 0:49] = V44
    # W: C_a is cols 0-3 of C.
    # W44 is 16x49. Rows correspond to C indices.
    # C_a rows 0-3, cols 0-3.
    # In W_final, rows 0-15 correspond to C[0:4, 0:4].
    # W_final row index = k * 4 + i (col k, row i).
    # C_a is cols 0-3. So rows 0-15 of W_final.
    W_final[0:16, 0:49] = W44
    
    # Part 2: A B_b1 (cols 4-7 of B)
    # U: same as U44
    U_final[:, 49:98] = U44
    # V: B_b1 is cols 4-7 of B.
    # B_b1 rows 0-3, cols 4-7.
    # In V_final, rows 16-31 correspond to B[0:4, 4:8].
    # V_final row index = j * 9 + k.
    # B_b1 is j in 0..3, k in 4..7.
    # So rows j*9 + 4 to j*9 + 7.
    for j in range(4):
        for local_k in range(4):
            g_row = j * 9 + (4 + local_k)
            V_final[g_row, 49:98] = V44[j * 4 + local_k, :]
            
    # W: C_b1 is cols 4-7 of C.
    # C_b1 rows 0-3, cols 4-7.
    # In W_final, rows 16-31 correspond to C[0:4, 4:8].
    # W_final row index = k * 4 + i.
    # C_b1 is k in 4..7, i in 0..3.
    # So rows k*4 + i.
    for k in range(4):
        for i in range(4):
            g_row = (4 + k) * 4 + i
            W_final[g_row, 49:98] = W44[k * 4 + i, :]

    # Part 3: A B_b2 (col 8 of B)
    # U: same as U441
    U_final[:, 98:114] = U441
    # V: B_b2 is col 8 of B.
    # B_b2 rows 0-3, col 8.
    # In V_final, rows 8, 17, 26, 35 correspond to B[0,8], B[1,8], B[2,8], B[3,8].
    # V441 is 4x16. Rows are j.
    for j in range(4):
        g_row = j * 9 + 8
        V_final[g_row, 98:114] = V441[j, :]
        
    # W: C_b2 is col 8 of C.
    # C_b2 rows 0-3, col 8.
    # In W_final, rows 32, 33, 34, 35 correspond to C[0,8], C[1,8], C[2,8], C[3,8].
    # W441 is 4x16. Rows are i.
    for i in range(4):
        g_row = 8 * 4 + i
        W_final[g_row, 98:114] = W441[i, :]

    return {"U": U_final, "V": V_final, "W": W_final}

