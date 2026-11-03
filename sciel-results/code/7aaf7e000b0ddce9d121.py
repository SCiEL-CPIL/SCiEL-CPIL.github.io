import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    U_rows = n * m  # 16
    V_rows = m * p  # 36
    W_rows = p * n  # 36
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_factors(u_vec, v_vec, w_vec):
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
        
    # Helper to create a zero vector
    def zero_vec(size):
        return [0] * size
        
    # Schoolbook for 2x2 * 2x2 -> 2x2
    # Returns list of (u_local, v_local, w_local) where indices are 0..3 for 2x2
    # u: (i, j) -> i*2+j
    # v: (j, k) -> j*2+k
    # w: (k, i) -> k*2+i
    def strassen_2x2_factors():
        factors = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u = [0]*4
                    v = [0]*4
                    w = [0]*4
                    u[i*2+j] = 1
                    v[j*2+k] = 1
                    w[k*2+i] = 1
                    factors.append((u, v, w))
        return factors

    # Strassen 4x4 * 4x4 -> 4x4 using schoolbook 2x2
    # A, B, C are 4x4.
    # Split into 2x2 blocks.
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # We need to add the factors for each Pi.
    # Each Pi is a 2x2 * 2x2 product.
    # The A-side of Pi is a linear combination of A's blocks.
    # The B-side of Pi is a linear combination of B's blocks.
    
    # Let's define the block indices.
    # A blocks: 00 (rows 0-1, cols 0-1), 01 (rows 0-1, cols 2-3), 10 (rows 2-3, cols 0-1), 11 (rows 2-3, cols 2-3)
    # B blocks: same relative to its own 4x4 grid.
    # C blocks: same relative to its own 4x4 grid.
    
    # For a given Strassen product Pi, the A-side is a 2x2 matrix formed by summing/differencing A blocks.
    # E.g., for P1, A_side = A11 + A22.
    # The rank-1 terms for A_side * B_side = C_side will have U vectors that are the sum of U vectors for A11 and A22.
    # Similarly for V.
    # The W vector is the W vector for the corresponding C block, with appropriate signs.
    
    # Let's precompute the local 2x2 schoolbook factors.
    local_2x2_factors = strassen_2x2_factors()
    
    # Function to add a 2x2 * 2x2 product to the global factors
    # a_blocks: list of (block_idx, sign) for A
    # b_blocks: list of (block_idx, sign) for B
    # c_block: (block_idx, sign) for C
    # b_offset: column offset for B in the 4x9 matrix
    # c_offset: column offset for C in the 4x9 matrix (same as b_offset for this structure)
    
    def add_2x2_product(a_blocks, b_blocks, c_block, b_col_offset):
        # a_blocks: e.g., [(0, 1), (3, 1)] for A11 + A22. Block 0 is top-left, 3 is bottom-right.
        # b_blocks: e.g., [(0, 1), (3, 1)]
        # c_block: (0, 1) for C11
        
        c_idx, c_sign = c_block
        
        for u_loc, v_loc, w_loc in local_2x2_factors:
            u_global = zero_vec(U_rows)
            v_global = zero_vec(V_rows)
            w_global = zero_vec(W_rows)
            
            # Fill U
            for a_idx, a_sign in a_blocks:
                # a_idx 0: rows 0-1, cols 0-1
                # a_idx 1: rows 0-1, cols 2-3
                # a_idx 2: rows 2-3, cols 0-1
                # a_idx 3: rows 2-3, cols 2-3
                
                # Map local 2x2 index (0..3) to global 4x4 index
                # Local (li, lj) where li, lj in 0..1
                # Global row: (a_idx // 2) * 2 + li
                # Global col: (a_idx % 2) * 2 + lj
                
                for li in range(2):
                    for lj in range(2):
                        local_idx = li * 2 + lj
                        val = u_loc[local_idx] * a_sign
                        if val != 0:
                            g_row = (a_idx // 2) * 2 + li
                            g_col = (a_idx % 2) * 2 + lj
                            u_global[g_row * m + g_col] += val
                            
            # Fill V
            for b_idx, b_sign in b_blocks:
                for lj in range(2):
                    for lk in range(2):
                        local_idx = lj * 2 + lk
                        val = v_loc[local_idx] * b_sign
                        if val != 0:
                            g_row = (b_idx // 2) * 2 + lj
                            # B col offset
                            g_col = b_col_offset + (b_idx % 2) * 2 + lk
                            v_global[g_row * p + g_col] += val
                            
            # Fill W
            for lk in range(2):
                for li in range(2):
                    local_idx = lk * 2 + li
                    val = w_loc[local_idx] * c_sign
                    if val != 0:
                        g_row = (c_idx % 2) * 2 + lk  # C col
                        g_col = (c_idx // 2) * 2 + li  # C row
                        # W index is k*n + i, so g_row is k, g_col is i
                        w_global[g_row * n + g_col] += val
                        
            add_factors(u_global, v_global, w_global)
            
    # Define Strassen products for 4x4
    # A blocks: 0(TL), 1(TR), 2(BL), 3(BR)
    # B blocks: 0(TL), 1(TR), 2(BL), 3(BR)
    
    # P1 = (A11+A22)(B11+B22) -> C11, C22
    add_2x2_product([(0,1), (3,1)], [(0,1), (3,1)], (0, 1), 0) # C11 part
    add_2x2_product([(0,1), (3,1)], [(0,1), (3,1)], (3, 1), 0) # C22 part? No, C22 gets P1 - P2 + P3 + P6
    
    # Wait, the C blocks are determined by the position.
    # P1 contributes to C11 and C22.
    # C11 = P1 + P4 - P5 + P7
    # C22 = P1 - P2 + P3 + P6
    
    # Let's re-structure. For each Pi, we add its contribution to the specific C blocks.
    
    # P1: A=(0,1)+(3,1), B=(0,1)+(3,1)
    # Contributes to C11 (block 0) with sign +1
    # Contributes to C22 (block 3) with sign +1
    add_2x2_product([(0,1), (3,1)], [(0,1), (3,1)], (0, 1), 0)
    add_2x2_product([(0,1), (3,1)], [(0,1), (3,1)], (3, 1), 0)
    
    # P2: A=(2,1)+(3,1), B=(0,1)
    # Contributes to C21 (block 2) with sign +1
    # Contributes to C22 (block 3) with sign -1
    add_2x2_product([(2,1), (3,1)], [(0,1)], (2, 1), 0)
    add_2x2_product([(2,1), (3,1)], [(0,1)], (3, -1), 0)
    
    # P3: A=(0,1), B=(1,1)+(3,-1)
    # Contributes to C12 (block 1) with sign +1
    # Contributes to C22 (block 3) with sign +1
    add_2x2_product([(0,1)], [(1,1), (3,-1)], (1, 1), 0)
    add_2x2_product([(0,1)], [(1,1), (3,-1)], (3, 1), 0)
    
    # P4: A=(3,1), B=(2,1)+(0,-1)
    # Contributes to C11 (block 0) with sign +1
    # Contributes to C21 (block 2) with sign +1
    add_2x2_product([(3,1)], [(2,1), (0,-1)], (0, 1), 0)
    add_2x2_product([(3,1)], [(2,1), (0,-1)], (2, 1), 0)
    
    # P5: A=(0,1)+(1,1), B=(3,1)
    # Contributes to C11 (block 0) with sign -1
    # Contributes to C12 (block 1) with sign +1
    add_2x2_product([(0,1), (1,1)], [(3,1)], (0, -1), 0)
    add_2x2_product([(0,1), (1,1)], [(3,1)], (1, 1), 0)
    
    # P6: A=(2,1)+(0,-1), B=(0,1)+(1,1)
    # Contributes to C22 (block 3) with sign +1
    add_2x2_product([(2,1), (0,-1)], [(0,1), (1,1)], (3, 1), 0)
    
    # P7: A=(1,1)+(3,-1), B=(2,1)+(3,1)
    # Contributes to C11 (block 0) with sign +1
    add_2x2_product([(1,1), (3,-1)], [(2,1), (3,1)], (0, 1), 0)
    
    # This handles the first 4x4 * 4x4 block (B cols 0-3, C cols 0-3).
    # Rank so far: 7 * 8 = 56.
    
    # Now for the second 4x4 * 4x4 block (B cols 4-7, C cols 4-7).
    # We just repeat the same logic with b_col_offset = 4.
    
    # P1
    add_2x2_product([(0,1), (3,1)], [(0,1), (3,1)], (0, 1), 4)
    add_2x2_product([(0,1), (3,1)], [(0,1), (3,1)], (3, 1), 4)
    # P2
    add_2x2_product([(2,1), (3,1)], [(0,1)], (2, 1), 4)
    add_2x2_product([(2,1), (3,1)], [(0,1)], (3, -1), 4)
    # P3
    add_2x2_product([(0,1)], [(1,1), (3,-1)], (1, 1), 4)
    add_2x2_product([(0,1)], [(1,1), (3,-1)], (3, 1), 4)
    # P4
    add_2x2_product([(3,1)], [(2,1), (0,-1)], (0, 1), 4)
    add_2x2_product([(3,1)], [(2,1), (0,-1)], (2, 1), 4)
    # P5
    add_2x2_product([(0,1), (1,1)], [(3,1)], (0, -1), 4)
    add_2x2_product([(0,1), (1,1)], [(3,1)], (1, 1), 4)
    # P6
    add_2x2_product([(2,1), (0,-1)], [(0,1), (1,1)], (3, 1), 4)
    # P7
    add_2x2_product([(1,1), (3,-1)], [(2,1), (3,1)], (0, 1), 4)
    
    # Rank so far: 56 + 56 = 112.
    
    # Now for the last 4x4 * 4x1 block (B col 8, C col 8).
    # This is just schoolbook 4x4 * 4x1.
    # 16 terms.
    for i in range(4):
        for j in range(4):
            u = zero_vec(U_rows)
            v = zero_vec(V_rows)
            w = zero_vec(W_rows)
            
            u[i*m + j] = 1
            v[j*p + 8] = 1
            w[8*n + i] = 1
            
            add_factors(u, v, w)
            
    # Total rank: 112 + 16 = 128.
    
    rank = len(U_list)
    U = [u for u in U_list]
    V = [v for v in V_list]
    W = [w for w in W_list]
    
    return {"U": U, "V": V, "W": W}
