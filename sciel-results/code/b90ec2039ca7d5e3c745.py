"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 part (Strassen)
    # Rank 32 for 4x4x2 part (Schoolbook)
    rank_444 = 49
    rank_442 = 32
    total_rank = rank_444 + rank_442
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 using Strassen (Rank 49) ---
    
    # Helper: Generate Strassen 2x2 coefficients
    # Returns list of 7 dicts: { 'A': { (i,j): coeff }, 'B': { (k,l): coeff }, 'C': { (i,k): coeff } }
    # Note: C coeff is the weight of this M_r in the final C entry.
    # We need to expand this into global indices.
    
    # Define Strassen 2x2 formulas
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Coefficients for A and B in each M
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    
    strassen_A = [
        {(0,0): 1, (1,1): 1},      # M1
        {(1,0): 1, (1,1): 1},      # M2
        {(0,0): 1},                # M3
        {(1,1): 1},                # M4
        {(0,0): 1, (0,1): 1},      # M5
        {(1,0): 1, (0,0): -1},     # M6
        {(0,1): 1, (1,1): -1}      # M7
    ]
    
    strassen_B = [
        {(0,0): 1, (1,1): 1},      # M1
        {(0,0): 1},                # M2
        {(0,1): 1, (1,1): -1},     # M3
        {(1,0): 1, (0,0): -1},     # M4
        {(1,1): 1},                # M5
        {(0,0): 1, (0,1): 1},      # M6
        {(1,0): 1, (1,1): 1}       # M7
    ]
    
    # Contribution to C (2x2) for each M
    # C indices: (0,0), (0,1), (1,0), (1,1)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_C = [
        {(0,0): 1, (1,1): 1},      # M1
        {(1,0): 1, (1,1): -1},     # M2
        {(0,1): 1, (1,1): 1},      # M3
        {(0,0): 1, (1,0): 1},      # M4
        {(0,0): -1, (0,1): 1},     # M5
        {(1,1): 1},                # M6
        {(0,0): 1}                 # M7
    ]
    
    # Block Formulas for 4x4
    # X_r and Y_r are 2x2 blocks of A and B
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    # Define block coefficients
    # X_r = sum alpha[u,v] A_block[u,v]
    # Y_r = sum beta[u,v] B_block[u,v]
    
    block_X = [
        {(0,0): 1, (1,1): 1},      # X1 = A00 + A11
        {(1,0): 1, (1,1): 1},      # X2 = A10 + A11
        {(0,0): 1},                # X3 = A00
        {(1,1): 1},                # X4 = A11
        {(0,0): 1, (0,1): 1},      # X5 = A00 + A01
        {(1,0): 1, (0,0): -1},     # X6 = A10 - A00
        {(0,1): 1, (1,1): -1}      # X7 = A01 - A11
    ]
    
    block_Y = [
        {(0,0): 1, (1,1): 1},      # Y1 = B00 + B11
        {(0,0): 1},                # Y2 = B00
        {(0,1): 1, (1,1): -1},     # Y3 = B01 - B11
        {(1,0): 1, (0,0): -1},     # Y4 = B10 - B00
        {(1,1): 1},                # Y5 = B11
        {(0,0): 1, (0,1): 1},      # Y6 = B00 + B01
        {(1,0): 1, (1,1): 1}       # Y7 = B10 + B11
    ]
    
    # Contribution of block P_r to C blocks
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    block_C = [
        {(0,0): 1, (1,1): 1},      # P1
        {(1,0): 1, (1,1): -1},     # P2
        {(0,1): 1, (1,1): 1},      # P3
        {(0,0): 1, (1,0): 1},      # P4
        {(0,0): -1, (0,1): 1},     # P5
        {(1,1): 1},                # P6
        {(0,0): 1}                 # P7
    ]
    
    col = 0
    for r in range(7):
        # X_r and Y_r are linear combinations of blocks
        # We need to expand the scalar Strassen formulas for X_r and Y_r
        # into global A and B indices.
        
        # For each scalar Strassen formula s in 0..6
        for s in range(7):
            # Coefficients for A in this scalar product
            # A_scalar = sum_{i,j} strassen_A[s][(i,j)] * X_r[i,j]
            # X_r[i,j] = sum_{u,v} block_X[r][(u,v)] * A_block[u,v][i,j]
            
            # A_block[u,v][i,j] corresponds to global A[2u+i, 2v+j]
            
            a_coeffs = {}
            for (i,j), coeff_a in strassen_A[s].items():
                for (u,v), coeff_x in block_X[r].items():
                    global_i = 2*u + i
                    global_j = 2*v + j
                    key = (global_i, global_j)
                    a_coeffs[key] = a_coeffs.get(key, 0) + coeff_a * coeff_x
            
            # Coefficients for B in this scalar product
            # B_scalar = sum_{k,l} strassen_B[s][(k,l)] * Y_r[k,l]
            # Y_r[k,l] = sum_{w,z} block_Y[r][(w,z)] * B_block[w,z][k,l]
            
            # B_block[w,z][k,l] corresponds to global B[2w+k, 2z+l]
            
            b_coeffs = {}
            for (k,l), coeff_b in strassen_B[s].items():
                for (w,z), coeff_y in block_Y[r].items():
                    global_k = 2*w + k
                    global_l = 2*z + l
                    key = (global_k, global_l)
                    b_coeffs[key] = b_coeffs.get(key, 0) + coeff_b * coeff_y
            
            # Contribution to C
            # C_block[cu,cv] += block_C[r][(cu,cv)] * P_r[cu,cv]
            # P_r[cu,cv] = sum_{i,k} strassen_C[s][(i,k)] * M_s
            # So M_s contributes to C_block[cu,cv] with weight block_C[r][(cu,cv)] * strassen_C[s][(i,k)]
            # where (i,k) is the position in the 2x2 block P_r.
            
            # Global C index: 2*cu + i, 2*cv + k
            
            for (i,k), coeff_c_s in strassen_C[s].items():
                for (cu,cv), coeff_c_block in block_C[r].items():
                    global_ci = 2*cu + i
                    global_ck = 2*cv + k
                    weight = coeff_c_s * coeff_c_block
                    
                    # Update U, V, W
                    # U index: global_i * m + global_j
                    # V index: global_k * p + global_l  (Note: p=4 for this part)
                    # W index: global_ck * n + global_ci (Note: n=4)
                    
                    # Wait, V index for 4x4x4 part:
                    # B is 4x4. j is col of A (0..3), k is col of B (0..3).
                    # V row index: j * 4 + k.
                    
                    # W index: k * n + i.
                    # k is col of B (0..3), i is row of C (0..3).
                    # W row index: k * 4 + i.
                    
                    for (gi, gj), coeff_u in a_coeffs.items():
                        u_idx = gi * 4 + gj
                        U[u_idx][col] += coeff_u * weight # Wait, U is just coeff_u?
                        # No, U[u_idx][col] is the coefficient of A[gi,gj] in the r-th factor.
                        # The factor is (sum U A) (sum V B) W.
                        # So U[u_idx][col] should be coeff_u.
                        # And V[v_idx][col] should be coeff_b.
                        # And W[w_idx][col] should be weight.
                        
                        # But we are iterating over C contributions.
                        # The term is M_s = (sum A) (sum B).
                        # M_s contributes to C with weight 'weight'.
                        # So we need to add to U, V, W such that
                        # U[u] V[v] W[w] = coeff_u * coeff_b * weight.
                        
                        # We can set:
                        # U[u][col] = coeff_u
                        # V[v][col] = coeff_b
                        # W[w][col] = weight
                        # But U and V are shared across all C contributions for this M_s?
                        # No, M_s is a single scalar product.
                        # It contributes to multiple C entries.
                        # So we need to add to W for each C entry.
                        # But U and V are fixed for this M_s.
                        
                        # So we should:
                        # 1. Set U and V for this column based on a_coeffs and b_coeffs.
                        # 2. Set W for each C entry based on weight.
                        
                        pass # Logic needs restructuring
                    
                    # Restructure:
                    # For each M_s (defined by r, s):
                    # 1. Determine a_coeffs and b_coeffs.
                    # 2. Set U and V for this column.
                    # 3. Determine all C entries it contributes to, and set W.
                    
            # Let's redo the loop logic inside the code block.
            
            # We will handle the assignment outside the inner loops for clarity in the final code.
            
    # The above logic is a bit tangled. I will write the final code carefully.
    
    # --- Re-implementation of Part 1 ---
    
    col = 0
    for r in range(7):
        for s in range(7):
            # Compute a_coeffs and b_coeffs as above
            a_coeffs = {}
            for (i,j), coeff_a in strassen_A[s].items():
                for (u,v), coeff_x in block_X[r].items():
                    global_i = 2*u + i
                    global_j = 2*v + j
                    key = (global_i, global_j)
                    a_coeffs[key] = a_coeffs.get(key, 0) + coeff_a * coeff_x
            
            b_coeffs = {}
            for (k,l), coeff_b in strassen_B[s].items():
                for (w,z), coeff_y in block_Y[r].items():
                    global_k = 2*w + k
                    global_l = 2*z + l
                    key = (global_k, global_l)
                    b_coeffs[key] = b_coeffs.get(key, 0) + coeff_b * coeff_y
            
            # Set U and V
            for (gi, gj), coeff_u in a_coeffs.items():
                u_idx = gi * 4 + gj
                U[u_idx][col] = coeff_u
            
            for (gk, gl), coeff_v in b_coeffs.items():
                v_idx = gk * 4 + gl
                V[v_idx][col] = coeff_v
            
            # Set W
            for (i,k), coeff_c_s in strassen_C[s].items():
                for (cu,cv), coeff_c_block in block_C[r].items():
                    global_ci = 2*cu + i
                    global_ck = 2*cv + k
                    weight = coeff_c_s * coeff_c_block
                    w_idx = global_ck * 4 + global_ci
                    W[w_idx][col] += weight
            
            col += 1
            
    # --- Part 2: 4x4x2 using Schoolbook (Rank 32) ---
    
    # B is 4x2.
    # A is 4x4.
    # C is 4x2.
    
    for i in range(4):
        for j in range(4):
            for k in range(2):
                u_idx = i * 4 + j
                v_idx = j * 2 + k
                w_idx = k * 4 + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
