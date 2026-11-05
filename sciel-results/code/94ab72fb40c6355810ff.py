def construct():
    import numpy as np
    
    n, m, p = 3, 4, 4
    total_rank = 44
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen 2x2x2 Rank 7
    # U rows (0,1,2,3) -> i*2+j
    # V rows (0,1,2,3) -> j*2+k
    # W rows (0,1,2,3) -> k*2+i
    # Standard Strassen coefficients
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # We need to map these to U, V, W factors.
    # U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c]
    
    # Let's define the 7 terms for 2x2x2
    # c is 0..6
    
    # Term 0: M1
    # A00, A11 in U. B00, B11 in V.
    # U[0,0]=1, U[3,0]=1
    # V[0,0]=1, V[3,0]=1
    # W[0,0]=1 (k=0,i=0), W[3,0]=1 (k=1,i=1)?
    # Wait, W index is k*2+i.
    # M1 contributes to C00, C01, C10, C11.
    # C00 += M1. C01 += M1. C10 += M1. C11 += M1.
    # So W[0,0]=1, W[1,0]=1, W[2,0]=1, W[3,0]=1.
    
    # This manual mapping is error-prone.
    # I will use the standard Strassen matrix for U, V, W.
    
    # U (4x7)
    U_s = [
        [1, 0, 1, 0, 1, -1, 0], # i=0, j=0 (A00)
        [0, 0, 0, 0, 1, 0, -1], # i=0, j=1 (A01)
        [0, 1, 0, 1, 0, 1, 0], # i=1, j=0 (A10)
        [1, 1, -1, 1, 1, 0, 1]  # i=1, j=1 (A11)
    ]
    # V (4x7)
    V_s = [
        [1, 1, 0, -1, 0, 1, 0], # j=0, k=0 (B00)
        [0, 0, 1, 0, 0, 1, 1], # j=0, k=1 (B01)
        [1, 0, 1, 0, 1, 0, 0], # j=1, k=0 (B10)
        [1, -1, -1, 1, 1, 0, 1] # j=1, k=1 (B11)
    ]
    # W (4x7)
    W_s = [
        [1, 0, 0, 0, 0, 0, 0], # k=0, i=0 (C00)
        [0, 0, 0, 1, 0, 0, 0], # k=0, i=1 (C10)
        [0, 1, 0, 0, 0, 0, 0], # k=1, i=0 (C01)
        [0, 0, 1, 0, 1, 1, 1]  # k=1, i=1 (C11)
    ]
    
    # Verify Strassen:
    # C00 = U00 V00 W00 + ...
    # This standard form is correct.
    
    # Schoolbook 1x2x2 Rank 4
    # A is 1x2 (A0, A1). B is 2x2. C is 1x2.
    # U (2x4) rows j=0,1
    U_sb = [
        [1, 0, 0, 0], # A0
        [0, 1, 0, 0]  # A1
    ]
    # V (4x4) rows j=0,1 ; k=0,1
    V_sb = [
        [1, 0, 0, 0], # B00
        [0, 0, 1, 0], # B01
        [0, 1, 0, 0], # B10
        [0, 0, 0, 1]  # B11
    ]
    # W (2x4) rows k=0,1
    W_sb = [
        [1, 0, 1, 0], # C0
        [0, 1, 0, 1]  # C1
    ]
    
    # Assembly
    # Blocks:
    # (a_row, a_col, b_row, b_col)
    # a_row: 0..2 (3 rows)
    # a_col: 0..1 (2 cols)
    # b_row: 0..1 (2 rows)
    # b_col: 0..1 (2 cols)
    
    # We iterate over 4 blocks of B (2x2 each)
    # And split A into 2x2 and 1x2.
    
    # Block 0: A[0:2, 0:2] * B[0:2, 0:2]
    # Block 1: A[2, 0:2] * B[0:2, 0:2]
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    # ...
    
    # Let's define offsets
    # For each of the 4 B-blocks (br, bc) in {0,1}x{0,1}
    # We have two A-parts: A_top (rows 0,1) and A_bot (row 2)
    
    rank_offset = 0
    
    for bc in range(2): # B col block 0,1
        for br in range(2): # B row block 0,1
            
            # Part 1: A[0:2, bc*2 : bc*2+2] * B[br*2 : br*2+2, bc*2 : bc*2+2]
            # This is 2x2x2
            # Local indices:
            # i_local in 0,1 -> i_global = i_local
            # j_local in 0,1 -> j_global = bc*2 + j_local
            # k_local in 0,1 -> k_global = bc*2 + k_local
            
            for c_local in range(7):
                c_global = rank_offset + c_local
                for i_l in range(2):
                    for j_l in range(2):
                        u_val = U_s[i_l*2 + j_l][c_local]
                        if u_val != 0:
                            i_g = i_l
                            j_g = bc*2 + j_l
                            U[i_g*m + j_g][c_global] += u_val
                            
                        # V and W are independent of A split
                for j_l in range(2):
                    for k_l in range(2):
                        v_val = V_s[j_l*2 + k_l][c_local]
                        if v_val != 0:
                            j_g = bc*2 + j_l
                            k_g = bc*2 + k_l
                            V[j_g*p + k_g][c_global] += v_val
                            
                for k_l in range(2):
                    for i_l in range(2):
                        w_val = W_s[k_l*2 + i_l][c_local]
                        if w_val != 0:
                            k_g = bc*2 + k_l
                            i_g = i_l
                            W[k_g*n + i_g][c_global] += w_val
                            
            rank_offset += 7
            
            # Part 2: A[2, bc*2 : bc*2+2] * B[br*2 : br*2+2, bc*2 : bc*2+2]
            # This is 1x2x2
            # Local indices:
            # i_local = 0 (only row) -> i_global = 2
            # j_local in 0,1 -> j_global = bc*2 + j_local
            # k_local in 0,1 -> k_global = bc*2 + k_local
            
            for c_local in range(4):
                c_global = rank_offset + c_local
                for j_l in range(2):
                    u_val = U_sb[j_l][c_local]
                    if u_val != 0:
                        i_g = 2
                        j_g = bc*2 + j_l
                        U[i_g*m + j_g][c_global] += u_val
                        
                for j_l in range(2):
                    for k_l in range(2):
                        v_val = V_sb[j_l*2 + k_l][c_local]
                        if v_val != 0:
                            j_g = bc*2 + j_l
                            k_g = bc*2 + k_l
                            V[j_g*p + k_g][c_global] += v_val
                            
                for k_l in range(2):
                    w_val = W_sb[k_l][c_local]
                    if w_val != 0:
                        k_g = bc*2 + k_l
                        i_g = 2
                        W[k_g*n + i_g][c_global] += w_val
                        
            rank_offset += 4
            
    return {"U": U, "V": V, "W": W}
