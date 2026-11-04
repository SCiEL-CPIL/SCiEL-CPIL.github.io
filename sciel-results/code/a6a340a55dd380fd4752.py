# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    DEADLINE = 145
    
    n, m, p = 4, 4, 5
    
    # Target Rank: 49 (for 4x4x4) + 16 (for 4x4x1) = 65
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)] # 16 x 65
    V = [[0] * rank for _ in range(m * p)] # 20 x 65
    W = [[0] * rank for _ in range(p * n)] # 20 x 65
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs, r_idx):
        for idx in range(n * m):
            U[idx][r_idx] += u_coeffs[idx]
        for idx in range(m * p):
            V[idx][r_idx] += v_coeffs[idx]
        for idx in range(p * n):
            W[idx][r_idx] += w_coeffs[idx]

    # Strassen 2x2 coefficients
    # L coeffs for P1..P7 (indices 0..6)
    # P1: L00+L11, P2: L10+L11, P3: L00, P4: L11, P5: L00+L01, P6: L10-L00, P7: L01-L11
    L_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    # R coeffs for P1..P7
    # P1: R00+R11, P2: R00, P3: R01-R11, P4: R10-R00, P5: R11, P6: R00+R01, P7: R10+R11
    R_strassen = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    # C block contributions from P1..P7
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    # Indices: 0:C00, 1:C01, 2:C10, 3:C11
    C_strassen = [
        [1, 0, 0, 1],   # P1
        [0, 0, 1, -1],  # P2
        [0, 1, 0, 1],   # P3
        [1, 0, 1, 0],   # P4
        [-1, 1, 0, 0],  # P5
        [0, 0, 0, 1],   # P6
        [1, 0, 0, 0]    # P7
    ]
    
    r_idx = 0
    
    # --- Part 1: 4x4x4 (Cols 0-3 of B, Rows 0-3 of C) ---
    # Outer Strassen on 2x2 blocks of A and B (first 4 cols)
    # Outer L coeffs (for A00, A01, A10, A11)
    # Same structure as L_strassen
    A_outer = L_strassen
    # Outer R coeffs (for B00, B01, B10, B11)
    B_outer = R_strassen
    # Outer C contributions (for C00, C01, C10, C11)
    C_outer = C_strassen
    
    # Iterate over 7 outer products
    for m_idx in range(7):
        # Iterate over 7 inner products for each outer product
        for s_idx in range(7):
            u_coeffs = [0] * (n * m)
            v_coeffs = [0] * (m * p)
            w_coeffs = [0] * (p * n)
            
            # --- Calculate U (A) coefficients ---
            # A is 4x4. Blocks A00, A01, A10, A11 are 2x2.
            # Outer step: Form L_m = sum_{b} A_outer[m_idx][b] * A_block_b
            # Inner step: Form L_inner = sum_{u} L_strassen[s_idx][u] * L_m_block_u
            # So A_block_b contributes to L_m_block_b with coeff A_outer[m_idx][b]
            # Then L_m_block_u contributes to L_inner_block_u with coeff L_strassen[s_idx][u]
            # Note: b and u are indices 0..3 corresponding to the 4 2x2 blocks.
            
            for b in range(4):
                coeff_outer = A_outer[m_idx][b]
                if coeff_outer == 0: continue
                # L_strassen[s_idx][b] is the coeff of L_m_block_b in L_inner
                coeff_inner = L_strassen[s_idx][b]
                if coeff_inner == 0: continue
                
                total_coeff = coeff_outer * coeff_inner
                if total_coeff == 0: continue
                
                # Add total_coeff to all elements in A_block_b
                # Block b: row_block = b//2, col_block = b%2
                r_block = b // 2
                c_block = b % 2
                for r in range(2):
                    for c in range(2):
                        i = r_block * 2 + r
                        j = c_block * 2 + c
                        u_coeffs[i * m + j] += total_coeff
            
            # --- Calculate V (B) coefficients ---
            # B is 4x5. We only use cols 0-3 for this part.
            # Blocks B00, B01, B10, B11 are 2x2 (rows 0-3, cols 0-3).
            # Outer step: Form R_m = sum_{b} B_outer[m_idx][b] * B_block_b
            # Inner step: Form R_inner = sum_{u} R_strassen[s_idx][u] * R_m_block_u
            
            for b in range(4):
                coeff_outer = B_outer[m_idx][b]
                if coeff_outer == 0: continue
                coeff_inner = R_strassen[s_idx][b]
                if coeff_inner == 0: continue
                
                total_coeff = coeff_outer * coeff_inner
                if total_coeff == 0: continue
                
                # Add total_coeff to all elements in B_block_b
                # Block b: row_block = b//2, col_block = b%2
                # Note: B is 4x5. Cols 0-3 are valid.
                r_block = b // 2
                c_block = b % 2
                for r in range(2):
                    for c in range(2):
                        j = r_block * 2 + r
                        k = c_block * 2 + c
                        v_coeffs[j * p + k] += total_coeff
            
            # --- Calculate W (C) coefficients ---
            # C is 4x5. We only use rows 0-3, cols 0-3 for this part.
            # Blocks C00, C01, C10, C11 are 2x2.
            # Outer step: M_m contributes to C blocks via C_outer[m_idx]
            # Inner step: P_s contributes to M_m blocks via C_strassen[s_idx]
            
            # M_m is a 2x2 matrix of 2x2 blocks.
            # M_m_block_b (b=0..3) is formed by inner products.
            # P_s contributes to M_m_block_b with coeff C_strassen[s_idx][b]
            # M_m_block_b contributes to C_block_b with coeff C_outer[m_idx][b]
            
            for b in range(4):
                coeff_inner = C_strassen[s_idx][b]
                if coeff_inner == 0: continue
                coeff_outer = C_outer[m_idx][b]
                if coeff_outer == 0: continue
                
                total_coeff = coeff_inner * coeff_outer
                if total_coeff == 0: continue
                
                # Add total_coeff to all elements in C_block_b
                # Block b: row_block = b//2, col_block = b%2
                r_block = b // 2
                c_block = b % 2
                for r in range(2):
                    for c in range(2):
                        i = r_block * 2 + r
                        k = c_block * 2 + c
                        # W index: k * n + i
                        w_coeffs[k * n + i] += total_coeff
            
            add_term(u_coeffs, v_coeffs, w_coeffs, r_idx)
            r_idx += 1
            
            if time.time() - start_time > DEADLINE - 5:
                # Fallback to schoolbook if too slow (should not happen for fixed 65)
                return construct_schoolbook()

    # --- Part 2: 4x4x1 (Col 4 of B, Rows 0-3 of C) ---
    # Schoolbook: C[i,4] = sum_j A[i,j] B[j,4]
    for i in range(4):
        for j in range(4):
            u_coeffs = [0] * (n * m)
            v_coeffs = [0] * (m * p)
            w_coeffs = [0] * (p * n)
            
            u_coeffs[i * m + j] = 1
            v_coeffs[j * p + 4] = 1
            w_coeffs[4 * n + i] = 1
            
            add_term(u_coeffs, v_coeffs, w_coeffs, r_idx)
            r_idx += 1

    return {"U": U, "V": V, "W": W}

def construct_schoolbook():
    n, m, p = 4, 4, 5
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r_idx] = 1
                V[j * p + k][r_idx] = 1
                W[k * n + i][r_idx] = 1
                r_idx += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
