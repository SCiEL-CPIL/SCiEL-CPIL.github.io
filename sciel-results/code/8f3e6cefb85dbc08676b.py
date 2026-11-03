import time
import sys

# EVOLVE-BLOCK-START

def construct():
    n, m, p = 4, 4, 7
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # We'll use a hybrid approach:
    # Part 1: 4x4x4 using Strassen (rank 49)
    # Part 2: 4x4x3 using schoolbook (rank 48)
    # Total rank: 97
    
    total_rank = 49 + 48
    
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x rank
    V = [[0] * total_rank for _ in range(m * p)]  # 28 x rank
    W = [[0] * total_rank for _ in range(p * n)]  # 28 x rank
    
    # --- Part 1: 4x4x4 Strassen (columns 0-3 of B and C) ---
    # Decompose A (4x4), B_4 (4x4), C_4 (4x4) into 2x2 blocks
    # Use Strassen at block level, then Strassen within each block product
    
    # Base 2x2x2 Strassen UVW
    def get_strassen_2x2_2x2():
        """Returns U, V, W for 2x2x2 Strassen, each 4x7"""
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        
        # M3 = A00(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1
        
        # M4 = A11(B10-B00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[1][4] = 1; W[3][4] = -1
        
        # M6 = (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = -1
        
        return U, V, W
    
    base_U, base_V, base_W = get_strassen_2x2_2x2()
    
    # For 4x4x4, we have 7 block-level Strassen steps
    # Each step involves products of 2x2 submatrices
    # We need to map these to global indices
    
    # Block-level Strassen for 4x4:
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # M2 = (A10+A11)B00 -> contributes to C10
    # M3 = A00(B01-B11) -> contributes to C01
    # M4 = A11(B10-B00) -> contributes to C00, C10
    # M5 = (A00+A01)B11 -> contributes to C01, C11
    # M6 = (A10-A00)(B00+B01) -> contributes to C11
    # M7 = (A01-A11)(B10+B11) -> contributes to C00, C11
    
    # Each block product is a 2x2x2 multiplication, which we decompose with base Strassen
    
    # Define the 7 block-level Strassen steps
    # Each step: list of (a_block, a_coeff), (b_block, b_coeff), and which c_blocks it contributes to with coefficients
    # a_block: (br, bc) where br, bc in {0,1}
    # b_block: (br, bc) where br, bc in {0,1}
    # c_blocks: list of ((cbr, cbc), coeff)
    
    block_steps = [
        # Step 0: M1 = (A00+A11)(B00+B11)
        {'a': [((0,0), 1), ((1,1), 1)], 'b': [((0,0), 1), ((1,1), 1)], 'c': [((0,0), 1), ((1,1), 1)]},
        # Step 1: M2 = (A10+A11)B00
        {'a': [((1,0), 1), ((1,1), 1)], 'b': [((0,0), 1)], 'c': [((1,0), 1)]},
        # Step 2: M3 = A00(B01-B11)
        {'a': [((0,0), 1)], 'b': [((0,1), 1), ((1,1), -1)], 'c': [((0,1), 1)]},
        # Step 3: M4 = A11(B10-B00)
        {'a': [((1,1), 1)], 'b': [((1,0), 1), ((0,0), -1)], 'c': [((0,0), 1), ((1,0), 1)]},
        # Step 4: M5 = (A00+A01)B11
        {'a': [((0,0), 1), ((0,1), 1)], 'b': [((1,1), 1)], 'c': [((0,1), 1), ((1,1), -1)]},
        # Step 5: M6 = (A10-A00)(B00+B01)
        {'a': [((1,0), 1), ((0,0), -1)], 'b': [((0,0), 1), ((0,1), 1)], 'c': [((1,1), 1)]},
        # Step 6: M7 = (A01-A11)(B10+B11)
        {'a': [((0,1), 1), ((1,1), -1)], 'b': [((1,0), 1), ((1,1), 1)], 'c': [((0,0), 1), ((1,1), -1)]},
    ]
    
    # Helper functions for global index computation
    # A index: i*m + j where i is row, j is col
    # B index: j*p + k where j is row, k is col (p=7, but for 4x4x4 part, k in 0..3)
    # W index: k*n + i where k is col, i is row
    
    def a_global_idx(br, bc, lr, lc):
        """Global A index from block (br,bc) and local (lr,lc)"""
        row = br * 2 + lr
        col = bc * 2 + lc
        return row * m + col
    
    def b_global_idx(br, bc, lr, lc):
        """Global B index from block (br,bc) and local (lr,lc) for the 4x4x4 part"""
        row = br * 2 + lr
        col = bc * 2 + lc
        return row * p + col  # col is 0..3 for the 4x4x4 part
    
    def w_global_idx(cbr, cbc, lr, lc):
        """Global W index from C block (cbr,cbc) and local (lr,lc)"""
        col = cbc * 2 + lc
        row = cbr * 2 + lr
        return col * n + row
    
    # For each block-level Strassen step, we have a 2x2x2 subproblem
    # The subproblem involves:
    #   X = sum of a_blocks (each 2x2)
    #   Y = sum of b_blocks (each 2x2)
    #   Z = X * Y (2x2 result)
    #   Then Z contributes to the specified c_blocks
    
    # We decompose X*Y using base Strassen
    # For each of the 7 base Strassen terms:
    #   X_local = linear combination of elements of X
    #   Y_local = linear combination of elements of Y
    #   Z_local = linear combination of elements of Z
    
    # X(lr, lc) = sum over a_blocks: coeff_a * A_block(br, bc)[lr, lc]
    # So for a specific global A element A_block(br, bc)[lr, lc],
    # its coefficient in X(lr, lc) is coeff_a
    
    # Similarly for Y and Z
    
    for step_idx, step in enumerate(block_steps):
        for base_term in range(7):
            rank_idx = step_idx * 7 + base_term
            
            # Process U (A part)
            for lr in range(2):
                for lc in range(2):
                    local_a_idx = lr * 2 + lc
                    coeff_u = base_U[local_a_idx][base_term]
                    if coeff_u == 0:
                        continue
                    # This local element of X comes from summing a_blocks
                    for (br, bc), coeff_a in step['a']:
                        if coeff_a == 0:
                            continue
                        global_a = a_global_idx(br, bc, lr, lc)
                        U[global_a][rank_idx] += coeff_u * coeff_a
            
            # Process V (B part)
            for lr in range(2):
                for lc in range(2):
                    local_b_idx = lr * 2 + lc
                    coeff_v = base_V[local_b_idx][base_term]
                    if coeff_v == 0:
                        continue
                    for (br, bc), coeff_b in step['b']:
                        if coeff_b == 0:
                            continue
                        global_b = b_global_idx(br, bc, lr, lc)
                        V[global_b][rank_idx] += coeff_v * coeff_b
            
            # Process W (C part)
            for lr in range(2):
                for lc in range(2):
                    local_c_idx = lr * 2 + lc
                    coeff_w = base_W[local_c_idx][base_term]
                    if coeff_w == 0:
                        continue
                    for (cbr, cbc), coeff_c in step['c']:
                        if coeff_c == 0:
                            continue
                        global_w = w_global_idx(cbr, cbc, lr, lc)
                        W[global_w][rank_idx] += coeff_w * coeff_c
    
    # --- Part 2: 4x4x3 schoolbook (columns 4-6 of B) ---
    # For each (i, j, k) where i in 0..3, j in 0..3, k in 4..6
    # This is 4*4*3 = 48 terms
    
    schoolbook_start = 49
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                rank_idx = schoolbook_start + t
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][rank_idx] = 1
                V[v_idx][rank_idx] = 1
                W[w_idx][rank_idx] = 1
                t += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
