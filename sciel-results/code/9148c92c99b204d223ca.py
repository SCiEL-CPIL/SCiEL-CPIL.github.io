def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definitions
    # Each term is (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: dict {index: coeff}
    # B_coeffs: dict {index: coeff} (local to block, i.e., 00, 01, 10, 11)
    # C_coeffs: dict {index: coeff} (local to block)
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B00-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B11-B01)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # A indices: 00:0, 01:1, 10:2, 11:3
    # B local indices: 00:0, 01:1, 10:2, 11:3
    # C local indices: 00:0, 01:1, 10:2, 11:3
    
    strassen_terms = [
        # (A_dict, B_dict, C_dict)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # M1 -> C00(+1), C11(+1)
        ({2: 1, 3: 1}, {0: 1}, {2: 1}),              # M2 -> C10(+1)
        ({0: 1}, {2: 1, 0: -1}, {1: 1}),             # M3 -> C01(+1)
        ({3: 1}, {0: 1, 2: -1}, {0: 1, 2: 1}),       # M4 -> C00(+1), C10(+1)
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),       # M5 -> C00(-1), C01(+1)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),       # M6 -> C11(+1)
        ({1: 1, 3: -1}, {3: 1, 1: -1}, {0: 1, 3: 1}) # M7 -> C00(+1), C11(+1)
    ]
    
    # Wait, I need to map C local indices to global W indices.
    # C local 00 -> C[0, 2l] -> W[2l*2 + 0]
    # C local 01 -> C[0, 2l+1] -> W[(2l+1)*2 + 0]
    # C local 10 -> C[1, 2l] -> W[2l*2 + 1]
    # C local 11 -> C[1, 2l+1] -> W[(2l+1)*2 + 1]
    
    for l in range(5):
        for s in range(7):
            r = 7 * l + s
            A_dict, B_dict, C_dict = strassen_terms[s]
            
            # Fill U
            for a_idx, coeff in A_dict.items():
                U[a_idx][r] = coeff
                
            # Fill V
            # B local idx 0 -> B[0, 2l] -> V[0*10 + 2l]
            # B local idx 1 -> B[0, 2l+1] -> V[0*10 + 2l+1]
            # B local idx 2 -> B[1, 2l] -> V[1*10 + 2l]
            # B local idx 3 -> B[1, 2l+1] -> V[1*10 + 2l+1]
            for b_idx, coeff in B_dict.items():
                if b_idx == 0:
                    v_idx = 0 * 10 + 2 * l
                elif b_idx == 1:
                    v_idx = 0 * 10 + 2 * l + 1
                elif b_idx == 2:
                    v_idx = 1 * 10 + 2 * l
                elif b_idx == 3:
                    v_idx = 1 * 10 + 2 * l + 1
                V[v_idx][r] = coeff
                
            # Fill W
            # C local idx 0 -> C[0, 2l] -> W[2l*2 + 0]
            # C local idx 1 -> C[0, 2l+1] -> W[(2l+1)*2 + 0]
            # C local idx 2 -> C[1, 2l] -> W[2l*2 + 1]
            # C local idx 3 -> C[1, 2l+1] -> W[(2l+1)*2 + 1]
            for c_idx, coeff in C_dict.items():
                if c_idx == 0:
                    w_idx = 2 * l * 2 + 0
                elif c_idx == 1:
                    w_idx = (2 * l + 1) * 2 + 0
                elif c_idx == 2:
                    w_idx = 2 * l * 2 + 1
                elif c_idx == 3:
                    w_idx = (2 * l + 1) * 2 + 1
                W[w_idx][r] = coeff
                
    return {"U": U, "V": V, "W": W}
