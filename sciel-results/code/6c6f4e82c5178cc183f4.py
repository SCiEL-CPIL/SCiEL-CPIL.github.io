def construct():
    n, m, p = 4, 4, 5
    
    # Base 2x2x2 Strassen terms
    # U, V, W vectors of length 4 each
    base_terms = [
        # M1 = (A11+A22)(B11+B22), contributes to C11, C22
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # M2 = (A21+A22)B11, contributes to C21
        ([0,0,1,1], [1,0,0,0], [0,1,0,0]),
        # M3 = A11(B12-B22), contributes to C12
        ([1,0,0,0], [0,1,0,-1], [0,0,1,0]),
        # M4 = A22(B21-B11), contributes to C21
        ([0,0,0,1], [0,0,1,-1], [0,1,0,0]),
        # M5 = (A11+A12)B22, contributes -C11+C12
        ([1,1,0,0], [0,0,0,1], [-1,0,1,0]),
        # M6 = (A21-A11)(B11+B12), contributes to C22
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        # M7 = (A12-A22)(B21+B22), contributes to C11
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    
    # Top-level 4x4x4 Strassen coefficients
    # Each entry: (A_block_coefs, B_block_coefs, C_block_coefs)
    # Block order: [block00, block01, block10, block11]
    top_terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),  # M1
        ([0,0,1,1], [1,0,0,0], [0,1,0,0]),  # M2
        ([1,0,0,0], [0,1,0,-1], [0,0,1,0]),  # M3
        ([0,0,0,1], [0,0,1,-1], [0,1,0,0]),  # M4
        ([1,1,0,0], [0,0,0,1], [-1,0,1,0]),  # M5
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),  # M6
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),  # M7
    ]
    
    # Build U, V, W for 4x4x4 part (rank 49)
    U_list = []
    V_list = []
    W_list = []
    
    # Block to global index mappings
    # A is 4x4, flat index i*4+j
    # Block (br, bc) where br,bc in {0,1}: rows 2*br to 2*br+1, cols 2*bc to 2*bc+1
    # Local (i_l, j_l) in block → global i = 2*br+i_l, j = 2*bc+j_l → flat = i*4+j
    
    # B is 4x5, flat index j*5+k
    # For the 4x4x4 part, we use cols 0-3 of B
    # Block (br, bc): rows 2*br to 2*br+1, cols 2*bc to 2*bc+1
    # Local (i_l, j_l) → global row = 2*br+i_l, col = 2*bc+j_l → flat = row*5+col
    
    # C is 4x5, W flat index k*4+i
    # For the 4x4x4 part, we use cols 0-3 of C
    # Block (br, bc): rows 2*br to 2*br+1, cols 2*bc to 2*bc+1
    # Local (i_l, j_l) → global row i = 2*br+i_l, col k = 2*bc+j_l → W flat = k*4+i
    
    for top_idx in range(7):
        A_coefs, B_coefs, C_coefs = top_terms[top_idx]
        for base_idx in range(7):
            U_base, V_base, W_base = base_terms[base_idx]
            
            U_vec = [0] * 16  # 4x4
            V_vec = [0] * 20  # 4x5
            W_vec = [0] * 20  # 5x4 (p*n = 5*4)
            
            # Expand U
            for block_idx in range(4):
                if A_coefs[block_idx] == 0:
                    continue
                br = block_idx // 2  # block row
                bc = block_idx % 2   # block col
                for i_l in range(2):
                    for j_l in range(2):
                        local_idx = i_l * 2 + j_l
                        global_i = 2 * br + i_l
                        global_j = 2 * bc + j_l
                        global_flat = global_i * 4 + global_j
                        U_vec[global_flat] = A_coefs[block_idx] * U_base[local_idx]
            
            # Expand V
            for block_idx in range(4):
                if B_coefs[block_idx] == 0:
                    continue
                br = block_idx // 2
                bc = block_idx % 2
                for i_l in range(2):
                    for j_l in range(2):
                        local_idx = i_l * 2 + j_l
                        global_row = 2 * br + i_l
                        global_col = 2 * bc + j_l
                        global_flat = global_row * 5 + global_col
                        V_vec[global_flat] = B_coefs[block_idx] * V_base[local_idx]
            
            # Expand W
            for block_idx in range(4):
                if C_coefs[block_idx] == 0:
                    continue
                br = block_idx // 2
                bc = block_idx % 2
                for i_l in range(2):
                    for j_l in range(2):
                        local_idx = i_l * 2 + j_l
                        global_i = 2 * br + i_l
                        global_k = 2 * bc + j_l
                        global_flat = global_k * 4 + global_i
                        W_vec[global_flat] = C_coefs[block_idx] * W_base[local_idx]
            
            U_list.append(U_vec)
            V_list.append(V_vec)
            W_list.append(W_vec)
    
    # Add schoolbook terms for the 5th column of B (and C)
    # C[i,4] = sum_j A[i,j] * B[j,4]
    for i in range(4):
        for j in range(4):
            U_vec = [0] * 16
            V_vec = [0] * 20
            W_vec = [0] * 20
            U_vec[i * 4 + j] = 1
            V_vec[j * 5 + 4] = 1
            W_vec[4 * 4 + i] = 1  # k=4, i → flat = 4*4+i = 16+i
            U_list.append(U_vec)
            V_list.append(V_vec)
            W_list.append(W_vec)
    
    rank = len(U_list)
    return {"U": U_list, "V": V_list, "W": W_list}
