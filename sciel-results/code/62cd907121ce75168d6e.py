import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition based on 8 Strassen steps of rank 7
    rank = 56
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen's 7 multiplications for 2x2 matrices
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # Coefficients for A: [a00, a01, a10, a11]
    # Coefficients for B: [b00, b01, b10, b11]
    # Coefficients for C: [c00, c01, c10, c11]
    
    strassen_steps = [
        # (A_coeffs, B_coeffs, C_coeffs)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0]), # M1 -> c00
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # M2 -> c10
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]), # M3 -> c01
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 0, 1]), # M4 -> c11
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]), # M5 -> c01 (Wait, M5 is (a00+a01)b11. c01 gets M5-M1? No.)
        
        # Let's re-verify Strassen formulas for C
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        
        # So we need to accumulate contributions.
        # It is easier to define the 7 M's and then sum them for C.
        # But the rank decomposition is a sum of rank-1 tensors.
        # Each M_c is a product of a linear form in A and a linear form in B.
        # So each M_c corresponds to a rank-1 tensor in the A-B space.
        # The C part is a linear combination of these M_c's.
        # So for each c, we have U_c, V_c, and W_c (which is a vector of 4 coeffs for the 2x2 C block).
        
        # Let's list the 7 M's with their A and B coeffs.
        # M1: A=[1,0,0,1], B=[1,0,0,1]
        # M2: A=[0,0,1,1], B=[1,0,0,0]
        # M3: A=[1,0,0,0], B=[0,1,0,-1]
        # M4: A=[0,0,0,1], B=[0,0,1,-1]
        # M5: A=[1,1,0,0], B=[0,0,0,1]
        # M6: A=[0,-1,1,0], B=[1,1,0,0]
        # M7: A=[0,1,0,-1], B=[0,0,1,1]
        
        # And the C contributions:
        # c00: M1, M4, -M5, M7
        # c01: M3, M5
        # c10: M2, M4
        # c11: M1, -M2, M3, M6
    ]
    
    # Define the 7 steps clearly
    steps = [
        ([1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1]), # M3
        ([0, 0, 0, 1], [0, 0, 1, -1]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1]), # M5
        ([0, -1, 1, 0], [1, 1, 0, 0]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1]), # M7
    ]
    
    # C contributions: list of (step_index, coeff) for each c entry
    # c00: M1(1), M4(1), M5(-1), M7(1)
    # c01: M3(1), M5(1)
    # c10: M2(1), M4(1)
    # c11: M1(1), M2(-1), M3(1), M6(1)
    
    c_contribs = [
        [(0, 1), (3, 1), (4, -1), (6, 1)], # c00
        [(2, 1), (4, 1)],                  # c01
        [(1, 1), (3, 1)],                  # c10
        [(0, 1), (1, -1), (2, 1), (5, 1)] # c11
    ]
    
    # We have 4 blocks of A: A1 (cols 0,1), A2 (cols 2,3)
    # We have 4 blocks of B: B11 (rows 0,1 cols 0,1), B12 (rows 0,1 cols 2,3) ...
    # Wait, B is 4x8.
    # B11 is rows 0,1 and cols 0,1. (2x2)
    # B12 is rows 0,1 and cols 2,3. (2x2)
    # B13 is rows 0,1 and cols 4,5. (2x2)
    # B14 is rows 0,1 and cols 6,7. (2x2)
    # B21 is rows 2,3 and cols 0,1. (2x2)
    # ...
    
    # C is 2x8.
    # C00 (cols 0,1) = A1 B11 + A2 B21
    # C01 (cols 2,3) = A1 B12 + A2 B22
    # C02 (cols 4,5) = A1 B13 + A2 B23
    # C03 (cols 6,7) = A1 B14 + A2 B24
    
    # Total 8 Strassen steps (each for a 2x2x2 multiplication).
    # Step 1: A1 * B11 -> contributes to C00
    # Step 2: A2 * B21 -> contributes to C00
    # Step 3: A1 * B12 -> contributes to C01
    # Step 4: A2 * B22 -> contributes to C01
    # Step 5: A1 * B13 -> contributes to C02
    # Step 6: A2 * B23 -> contributes to C02
    # Step 7: A1 * B14 -> contributes to C03
    # Step 8: A2 * B24 -> contributes to C03
    
    # For each step, we have 7 sub-steps (M1..M7).
    # Total 56 sub-steps.
    
    col = 0
    for step_idx in range(8):
        # Determine A block and B block
        if step_idx % 2 == 0:
            a_block = 0 # A1 (cols 0,1)
        else:
            a_block = 1 # A2 (cols 2,3)
            
        b_block_idx = step_idx // 2 # 0,1,2,3 corresponding to B1x, B2x?
        # step 0: B11 (rows 0,1, cols 0,1)
        # step 1: B21 (rows 2,3, cols 0,1)
        # step 2: B12 (rows 0,1, cols 2,3)
        # step 3: B22 (rows 2,3, cols 2,3)
        
        # Map step_idx to B row/col offset
        # step 0: r=0, c=0
        # step 1: r=2, c=0
        # step 2: r=0, c=2
        # step 3: r=2, c=2
        # step 4: r=0, c=4
        # step 5: r=2, c=4
        # step 6: r=0, c=6
        # step 7: r=2, c=6
        
        b_row_offset = (step_idx % 2) * 2
        b_col_offset = (step_idx // 2) * 2
        
        # Determine C block offset
        # step 0,1 -> C cols 0,1
        # step 2,3 -> C cols 2,3
        # step 4,5 -> C cols 4,5
        # step 6,7 -> C cols 6,7
        c_col_offset = (step_idx // 2) * 2
        
        for m_idx in range(7):
            a_coeffs, b_coeffs = steps[m_idx]
            
            # Fill U
            # A indices: i in 0,1; j in 0,1 (local)
            # Global j = a_block*2 + j_local
            for i_local in range(2):
                for j_local in range(2):
                    coeff = a_coeffs[i_local * 2 + j_local]
                    if coeff != 0:
                        i_global = i_local
                        j_global = a_block * 2 + j_local
                        U[i_global * m + j_global][col] = coeff
                        
            # Fill V
            # B indices: j in 0,1 (local); k in 0,1 (local)
            # Global j = b_row_offset + j_local
            # Global k = b_col_offset + k_local
            for j_local in range(2):
                for k_local in range(2):
                    coeff = b_coeffs[j_local * 2 + k_local]
                    if coeff != 0:
                        j_global = b_row_offset + j_local
                        k_global = b_col_offset + k_local
                        V[j_global * p + k_global][col] = coeff
                        
            # Fill W
            # C indices: i in 0,1; k in 0,1 (local)
            # Global i = i_local
            # Global k = c_col_offset + k_local
            # We need to find which C entries are affected by this M_m_idx
            # We sum over c_contribs
            for c_local_idx in range(4):
                # Find contribution of m_idx to this c_local entry
                coeff_c = 0
                for (m_ref, c_ref) in c_contribs[c_local_idx]:
                    if m_ref == m_idx:
                        coeff_c += c_ref
                        break
                        
                if coeff_c != 0:
                    i_local = c_local_idx // 2
                    k_local = c_local_idx % 2
                    i_global = i_local
                    k_global = c_col_offset + k_local
                    W[k_global * n + i_global][col] = coeff_c
                    
            col += 1
            
    return {"U": U, "V": V, "W": W}
