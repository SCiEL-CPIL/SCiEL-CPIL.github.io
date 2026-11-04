import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Define Strassen factors for 2x2x2
    # u, v, w are lists of 7 vectors of length 4
    # u[i] corresponds to A_ij where i = j*2 + k? No, index is i*2+j
    # u index: 0=00, 1=01, 2=10, 3=11
    
    u_2x2 = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    v_2x2 = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    w_2x2 = [
        [1, 0, 0, 1], # M1 -> C00, C11
        [0, 1, 0, -1], # M2 -> C10, C11
        [0, 0, 1, 1], # M3 -> C01, C11
        [1, 1, 0, 0], # M4 -> C00, C10
        [-1, 0, 1, 0], # M5 -> C00, C01
        [0, 0, 0, 1], # M6 -> C11
        [1, 0, 0, 0]  # M7 -> C00
    ]
    
    # 2. Construct 4x4x4 decomposition (Rank 49)
    # We need to map the 2x2 block factors to 4x4 factors.
    # A is 4x4. Blocks A00, A01, A10, A11 are 2x2.
    # U for 4x4 has size 16.
    # Index in U: i*4 + j.
    # Block A00 is rows 0,1 and cols 0,1.
    # Block A01 is rows 0,1 and cols 2,3.
    # Block A10 is rows 2,3 and cols 0,1.
    # Block A11 is rows 2,3 and cols 2,3.
    
    def expand_u_2x2_to_4x4(u_vec, block_idx):
        # u_vec is length 4 (2x2 block)
        # block_idx: 0=A00, 1=A01, 2=A10, 3=A11
        full_u = [0]*16
        r_block, c_block = divmod(block_idx, 2) # 0,0 -> 0; 0,1 -> 1; 1,0 -> 2; 1,1 -> 3
        # r_block is 0 or 1, c_block is 0 or 1
        for i in range(2):
            for j in range(2):
                local_idx = i*2 + j
                global_i = r_block*2 + i
                global_j = c_block*2 + j
                global_idx = global_i*4 + global_j
                full_u[global_idx] = u_vec[local_idx]
        return full_u
    
    def expand_v_2x2_to_4x4(v_vec, block_idx):
        # v_vec is length 4 (2x2 block)
        # B is 4x4. Blocks B00, B01, B10, B11.
        # V index: j*4 + k.
        # Block B00: rows 0,1 cols 0,1.
        full_v = [0]*16
        r_block, c_block = divmod(block_idx, 2)
        for i in range(2):
            for j in range(2):
                local_idx = i*2 + j
                global_j = r_block*2 + i
                global_k = c_block*2 + j
                global_idx = global_j*4 + global_k
                full_v[global_idx] = v_vec[local_idx]
        return full_v
    
    def expand_w_2x2_to_4x4(w_vec, block_idx):
        # w_vec is length 4 (2x2 block of C)
        # C is 4x4. Blocks C00, C01, C10, C11.
        # W index: k*4 + i.
        # Block C00: rows 0,1 cols 0,1.
        full_w = [0]*16
        r_block, c_block = divmod(block_idx, 2)
        # r_block corresponds to i (row of C), c_block to k (col of C)
        for i in range(2):
            for j in range(2):
                local_idx = i*2 + j # i_local, k_local
                global_i = r_block*2 + i
                global_k = c_block*2 + j
                global_idx = global_k*4 + global_i
                full_w[global_idx] = w_vec[local_idx]
        return full_w

    U_list = []
    V_list = []
    W_list = []
    
    # Part 1: 4x4x4
    # Strassen for 4x4x4 uses 7 products of 2x2 matrices.
    # The 7 products are linear combinations of A blocks and B blocks.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to generate the U, V, W vectors for each of these 7 products.
    # Each product is a 2x2 matrix multiplication, which we decompose into 7 rank-1 terms.
    # So for each of the 7 Strassen products, we add 7 terms to the global decomposition.
    
    # Coefficients for U (A blocks)
    # u_coeff[prod_idx][block_idx]
    u_coeff = [
        [1, 0, 0, 1], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00+A01
        [-1, 0, 1, 0], # M6: A10-A00
        [0, 1, 0, -1]  # M7: A01-A11
    ]
    
    # Coefficients for V (B blocks)
    v_coeff = [
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1], # M3: B01-B11
        [0, 0, 1, -1], # M4: B10-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    
    # Coefficients for W (C blocks)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    w_coeff = [
        [1, 0, 0, 1], # M1: C00, C11
        [0, 0, 1, -1], # M2: C10, C11
        [0, 1, 0, 1], # M3: C01, C11
        [1, 0, 1, 0], # M4: C00, C10
        [-1, 1, 0, 0], # M5: C00, C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    
    for prod_idx in range(7):
        for inner_idx in range(7):
            # Construct U vector
            u_vec = [0]*16
            for b_idx in range(4):
                coeff = u_coeff[prod_idx][b_idx]
                if coeff != 0:
                    # Expand the 2x2 u vector scaled by coeff
                    base_u = u_2x2[inner_idx]
                    expanded = expand_u_2x2_to_4x4(base_u, b_idx)
                    for k in range(16):
                        u_vec[k] += coeff * expanded[k]
            
            # Construct V vector
            v_vec = [0]*16
            for b_idx in range(4):
                coeff = v_coeff[prod_idx][b_idx]
                if coeff != 0:
                    base_v = v_2x2[inner_idx]
                    expanded = expand_v_2x2_to_4x4(base_v, b_idx)
                    for k in range(16):
                        v_vec[k] += coeff * expanded[k]
            
            # Construct W vector
            w_vec = [0]*16
            for b_idx in range(4):
                coeff = w_coeff[prod_idx][b_idx]
                if coeff != 0:
                    base_w = w_2x2[inner_idx]
                    expanded = expand_w_2x2_to_4x4(base_w, b_idx)
                    for k in range(16):
                        w_vec[k] += coeff * expanded[k]
            
            U_list.append(u_vec)
            V_list.append(v_vec)
            W_list.append(w_vec)
            
    # Part 2: 4x4x2
    # B is 4x2. Split into B_top (2x2) and B_bot (2x2).
    # C is 4x2. Split into C_top (2x2) and C_bot (2x2).
    # C_top = A00 B_top + A01 B_bot
    # C_bot = A10 B_top + A11 B_bot
    
    # We need to handle the V and W dimensions differently.
    # V is 4x6. Rows 0-3 are B_top, rows 4-7 are B_bot?
    # B is 4x6.
    # B_top is rows 0-1 of B? No, B is 4x6.
    # We split B into two 4x3 blocks? No, we split B into 4x2 blocks.
    # B = [B1, B2, B3] where each is 4x2.
    # We are computing C1 (4x2) and C2 (4x2).
    # C1 = A B1. C2 = A B2.
    # For C1: B1 is 4x2. Split into B1_top (2x2) and B1_bot (2x2).
    # C1_top = A00 B1_top + A01 B1_bot.
    # C1_bot = A10 B1_top + A11 B1_bot.
    
    # V indices: j*6 + k.
    # B1 is cols 0,1. B2 is cols 2,3. B3 is cols 4,5.
    # For B1_top (rows 0,1 cols 0,1): j in 0,1; k in 0,1.
    # Index: j*6 + k.
    
    def expand_v_2x2_to_4x6(v_vec, block_row, block_col):
        # block_row: 0 for top (rows 0,1), 1 for bot (rows 2,3)
        # block_col: 0 for B1 (cols 0,1), 1 for B2 (cols 2,3), 2 for B3 (cols 4,5)
        full_v = [0]*24
        for i in range(2):
            for j in range(2):
                local_idx = i*2 + j
                global_j = block_row*2 + i
                global_k = block_col*2 + j
                global_idx = global_j*6 + global_k
                full_v[global_idx] = v_vec[local_idx]
        return full_v
        
    def expand_w_2x2_to_4x6(w_vec, block_row, block_col):
        # w_vec is for C block (2x2)
        # block_row: 0 for C_top (rows 0,1), 1 for C_bot (rows 2,3)
        # block_col: 0 for C1 (cols 0,1), 1 for C2 (cols 2,3), 2 for C3 (cols 4,5)
        full_w = [0]*24
        for i in range(2):
            for j in range(2):
                local_idx = i*2 + j
                global_i = block_row*2 + i
                global_k = block_col*2 + j
                global_idx = global_k*4 + global_i
                full_w[global_idx] = w_vec[local_idx]
        return full_w

    # We need to compute C1 and C2.
    # For C1 (col block 0):
    # Terms: A00 B1_top, A01 B1_bot, A10 B1_top, A11 B1_bot.
    # For C2 (col block 1):
    # Terms: A00 B2_top, A01 B2_bot, A10 B2_top, A11 B2_bot.
    
    # We can just iterate over col_block in 0, 1.
    for col_block in range(2):
        # Top part: C_top = A00 B_top + A01 B_bot
        # A00 B_top:
        for inner_idx in range(7):
            u_vec = expand_u_2x2_to_4x4(u_2x2[inner_idx], 0) # A00
            v_vec = expand_v_2x2_to_4x6(v_2x2[inner_idx], 0, col_block) # B_top
            w_vec = expand_w_2x2_to_4x6(w_2x2[inner_idx], 0, col_block) # C_top
            U_list.append(u_vec)
            V_list.append(v_vec)
            W_list.append(w_vec)
            
            # A01 B_bot
            u_vec = expand_u_2x2_to_4x4(u_2x2[inner_idx], 1) # A01
            v_vec = expand_v_2x2_to_4x6(v_2x2[inner_idx], 1, col_block) # B_bot
            w_vec = expand_w_2x2_to_4x6(w_2x2[inner_idx], 0, col_block) # C_top
            U_list.append(u_vec)
            V_list.append(v_vec)
            W_list.append(w_vec)
            
        # Bottom part: C_bot = A10 B_top + A11 B_bot
        for inner_idx in range(7):
            # A10 B_top
            u_vec = expand_u_2x2_to_4x4(u_2x2[inner_idx], 2) # A10
            v_vec = expand_v_2x2_to_4x6(v_2x2[inner_idx], 0, col_block) # B_top
            w_vec = expand_w_2x2_to_4x6(w_2x2[inner_idx], 1, col_block) # C_bot
            U_list.append(u_vec)
            V_list.append(v_vec)
            W_list.append(w_vec)
            
            # A11 B_bot
            u_vec = expand_u_2x2_to_4x4(u_2x2[inner_idx], 3) # A11
            v_vec = expand_v_2x2_to_4x6(v_2x2[inner_idx], 1, col_block) # B_bot
            w_vec = expand_w_2x2_to_4x6(w_2x2[inner_idx], 1, col_block) # C_bot
            U_list.append(u_vec)
            V_list.append(v_vec)
            W_list.append(w_vec)

    # We have 49 + 28 = 77 factors.
    # Convert to lists of lists
    U = [u for u in U_list]
    V = [v for v in V_list]
    W = [w for w in W_list]
    
    return {"U": U, "V": V, "W": W}
