def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Strassen 2x2x2 decomposition
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # M0 = (a00+a11)(b00+b11)
    # M1 = (a10+a11)b00
    # M2 = a00(b01-b11)
    # M3 = a11(b10-b00)
    # M4 = (a00+a01)b11
    # M5 = (a10-a00)(b00+b01)
    # M6 = (a01-a11)(b10+b11)
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5

    # U coeffs for A side (indices 0,1,2,3 correspond to a00, a01, a10, a11)
    # V coeffs for B side (indices 0,1,2,3 correspond to b00, b01, b10, b11)
    # W coeffs for C side (indices 0,1,2,3 correspond to C00, C01, C10, C11)
    
    # Note: In the global mapping, we will map local indices to global indices.
    
    a_coeffs = [
        [1, 0, 0, 1], # M0: a00 + a11
        [0, 0, 1, 1], # M1: a10 + a11
        [1, 0, 0, 0], # M2: a00
        [0, 0, 0, 1], # M3: a11
        [1, 1, 0, 0], # M4: a00 + a01
        [0, 0, 1, -1],# M5: a10 - a00
        [0, 1, 0, -1] # M6: a01 - a11
    ]
    
    b_coeffs = [
        [1, 0, 0, 1], # M0: b00 + b11
        [1, 0, 0, 0], # M1: b00
        [0, 1, 0, -1],# M2: b01 - b11
        [0, 0, 1, -1],# M3: b10 - b00
        [0, 0, 0, 1], # M4: b11
        [1, 1, 0, 0], # M5: b00 + b01
        [0, 0, 1, 1]  # M6: b10 + b11
    ]
    
    w_coeffs = [
        [1, 0, 0, 0], # M0: C00
        [0, 0, 1, 0], # M1: C10
        [0, 1, 0, 0], # M2: C01
        [1, 0, 1, 0], # M3: C00, C10
        [-1, 1, 0, 0],# M4: C00, C01
        [0, 0, 0, 1], # M5: C11
        [1, 0, 0, 0]  # M6: C00
    ]
    
    # Check W coeffs sum:
    # C00: M0(1) + M3(1) - M4(1) + M6(1) = 1+1-1+1 = 2? 
    # Wait, Strassen formulas:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # Let's re-verify M3 in C00 and C10.
    # M3 = a11(b10-b00).
    # C00 = M0 + M3 - M4 + M6.
    # C10 = M1 + M3.
    # So M3 contributes to both C00 and C10.
    # My w_coeffs for M3 (index 3) is [1, 0, 1, 0]. Correct.
    
    # M4 = (a00+a01)b11.
    # C00 = ... - M4 ...
    # C01 = ... + M4 ...
    # So M4 contributes -1 to C00, +1 to C01.
    # My w_coeffs for M4 (index 4) is [-1, 1, 0, 0]. Correct.
    
    # M6 = (a01-a11)(b10+b11).
    # C00 = ... + M6 ...
    # So M6 contributes +1 to C00.
    # My w_coeffs for M6 (index 6) is [1, 0, 0, 0]. Correct.
    
    # M0 = (a00+a11)(b00+b11).
    # C00 = M0 ...
    # C11 = M0 ...
    # So M0 contributes +1 to C00, +1 to C11.
    # My w_coeffs for M0 (index 0) is [1, 0, 0, 0]. 
    # WAIT. C11 also needs M0.
    # C11 = M0 - M1 + M2 + M5.
    # So M0 should be [1, 0, 0, 1].
    # Let's fix w_coeffs.
    
    w_coeffs = [
        [1, 0, 0, 1], # M0: C00, C11
        [0, 0, 1, -1],# M1: C10, C11
        [0, 1, 0, 1], # M2: C01, C11
        [1, 0, 1, 0], # M3: C00, C10
        [-1, 1, 0, 0],# M4: C00, C01
        [0, 0, 0, 1], # M5: C11
        [1, 0, 0, 0]  # M6: C00
    ]
    
    # Verify:
    # C00: M0(1)+M3(1)-M4(1)+M6(1) = 2? No.
    # C00 = M0 + M3 - M4 + M6.
    # Coeffs: M0:1, M3:1, M4:-1, M6:1.
    # Sum = 1+1-1+1 = 2? 
    # Let's check Strassen again.
    # a00b00 + a01b10 = C00.
    # M0 = (a00+a11)(b00+b11) = a00b00 + a00b11 + a11b00 + a11b11
    # M3 = a11(b10-b00) = a11b10 - a11b00
    # M4 = (a00+a01)b11 = a00b11 + a01b11
    # M6 = (a01-a11)(b10+b11) = a01b10 + a01b11 - a11b10 - a11b11
    # M0 + M3 - M4 + M6 = 
    # (a00b00 + a00b11 + a11b00 + a11b11) 
    # + (a11b10 - a11b00) 
    # - (a00b11 + a01b11) 
    # + (a01b10 + a01b11 - a11b10 - a11b11)
    # = a00b00 + a00b11 + a11b00 + a11b11 + a11b10 - a11b00 - a00b11 - a01b11 + a01b10 + a01b11 - a11b10 - a11b11
    # Cancel a00b11, a11b00, a11b11, a11b10, a01b11.
    # Remaining: a00b00 + a01b10.
    # Correct.
    
    # So w_coeffs for M0 is [1, 0, 0, 1] is WRONG for C00?
    # No, M0 contributes to C00 and C11.
    # In C00, coeff is +1.
    # In C11, coeff is +1.
    # So [1, 0, 0, 1] is correct for M0.
    
    # M1 = (a10+a11)b00 = a10b00 + a11b00
    # C10 = M1 + M3.
    # C11 = M0 - M1 + M2 + M5.
    # M1 contributes +1 to C10, -1 to C11.
    # w_coeffs[1] = [0, 0, 1, -1]. Correct.
    
    # M2 = a00(b01-b11) = a00b01 - a00b11
    # C01 = M2 + M4.
    # C11 = ... + M2 ...
    # M2 contributes +1 to C01, +1 to C11.
    # w_coeffs[2] = [0, 1, 0, 1]. Correct.
    
    # M3 = a11(b10-b00) = a11b10 - a11b00
    # C00 = ... + M3 ...
    # C10 = M1 + M3.
    # M3 contributes +1 to C00, +1 to C10.
    # w_coeffs[3] = [1, 0, 1, 0]. Correct.
    
    # M4 = (a00+a01)b11 = a00b11 + a01b11
    # C00 = ... - M4 ...
    # C01 = M2 + M4.
    # M4 contributes -1 to C00, +1 to C01.
    # w_coeffs[4] = [-1, 1, 0, 0]. Correct.
    
    # M5 = (a10-a00)(b00+b01) = a10b00 + a10b01 - a00b00 - a00b01
    # C11 = ... + M5 ...
    # M5 contributes +1 to C11.
    # w_coeffs[5] = [0, 0, 0, 1]. Correct.
    
    # M6 = (a01-a11)(b10+b11) = a01b10 + a01b11 - a11b10 - a11b11
    # C00 = ... + M6 ...
    # M6 contributes +1 to C00.
    # w_coeffs[6] = [1, 0, 0, 0]. Correct.
    
    # Now build the blocks.
    
    # Block 0: A_L (cols 0,1) * B_11 (rows 0,1, cols 0,1) -> C_L (cols 0,1)
    # A local (i,j) -> Global U[i*4 + j]
    # B local (j,k) -> Global V[j*4 + k]
    # C local (i,k) -> Global W[k*2 + i]
    
    # Block 1: A_R (cols 2,3) * B_21 (rows 2,3, cols 0,1) -> C_L (cols 0,1)
    # A local (i,j) -> Global U[i*4 + 2 + j]
    # B local (j,k) -> Global V[(2+j)*4 + k]
    # C local (i,k) -> Global W[k*2 + i]
    
    # Block 2: A_L (cols 0,1) * B_12 (rows 0,1, cols 2,3) -> C_R (cols 2,3)
    # A local (i,j) -> Global U[i*4 + j]
    # B local (j,k) -> Global V[j*4 + 2 + k]
    # C local (i,k) -> Global W[(2+k)*2 + i]
    
    # Block 3: A_R (cols 2,3) * B_22 (rows 2,3, cols 2,3) -> C_R (cols 2,3)
    # A local (i,j) -> Global U[i*4 + 2 + j]
    # B local (j,k) -> Global V[(2+j)*4 + 2 + k]
    # C local (i,k) -> Global W[(2+k)*2 + i]
    
    col = 0
    for block in range(4):
        # Determine offsets
        if block == 0:
            a_off_i = 0; a_off_j = 0
            b_off_i = 0; b_off_j = 0
            c_off_i = 0; c_off_j = 0
        elif block == 1:
            a_off_i = 0; a_off_j = 2
            b_off_i = 2; b_off_j = 0
            c_off_i = 0; c_off_j = 0
        elif block == 2:
            a_off_i = 0; a_off_j = 0
            b_off_i = 0; b_off_j = 2
            c_off_i = 0; c_off_j = 2
        else: # block 3
            a_off_i = 0; a_off_j = 2
            b_off_i = 2; b_off_j = 2
            c_off_i = 0; c_off_j = 2
            
        for s in range(7):
            # U
            for la in range(4):
                if a_coeffs[s][la] != 0:
                    li, lj = divmod(la, 2)
                    u_idx = (li + a_off_i) * 4 + (lj + a_off_j)
                    U[u_idx][col] += a_coeffs[s][la]
            
            # V
            for lb in range(4):
                if b_coeffs[s][lb] != 0:
                    li, lj = divmod(lb, 2)
                    v_idx = (li + b_off_i) * 4 + (lj + b_off_j)
                    V[v_idx][col] += b_coeffs[s][lb]
                    
            # W
            for lc in range(4):
                if w_coeffs[s][lc] != 0:
                    li, lj = divmod(lc, 2)
                    # C local (li, lj) -> Global C[li, lj + c_off_j]
                    # W index: (col_idx) * n + row_idx = (lj + c_off_j) * 2 + li
                    w_idx = (lj + c_off_j) * 2 + (li + c_off_i)
                    W[w_idx][col] += w_coeffs[s][lc]
                    
            col += 1
            
    return {"U": U, "V": V, "W": W}
