"""Matrix multiplication tensor decomposition: 2x4 * 4x8 -> 2x8"""

def construct():
    n, m, p = 2, 4, 8
    
    # Strassen's rank-7 decomposition for 2x2 * 2x2 -> 2x2
    # The 8 independent 2x2 sub-problems give rank = 8 * 7 = 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # (8, 56)
    V = [[0] * rank for _ in range(m * p)]  # (32, 56)
    W = [[0] * rank for _ in range(p * n)]  # (16, 56)
    
    # Strassen coefficients:
    # S_1 = (A00+A11)(B00+B11)
    # S_2 = (A10+A11)B00
    # S_3 = A00(B01-B11)
    # S_4 = A11(B10-B00)
    # S_5 = (A00+A01)B11
    # S_6 = (A10-A00)(B00+B01)
    # S_7 = (A01-A11)(B10+B11)
    #
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    
    # U coefficients: for each Strassen term t, which (i,k_local) pairs have non-zero entries
    # i in {0,1}, k_local in {0,1}
    U_coeffs = {
        0: {0: {0: 1, 1: 0}, 1: {0: 1, 1: 1}},  # S1: A00 + A11
        1: {0: {0: 0, 1: 0}, 1: {0: 1, 1: 1}},  # S2: A10 + A11
        2: {0: {0: 1, 1: 0}, 1: {0: 0, 1: 0}},  # S3: A00
        3: {0: {0: 0, 1: 0}, 1: {0: 0, 1: 1}},  # S4: A11
        4: {0: {0: 1, 1: 1}, 1: {0: 0, 1: 0}},  # S5: A00 + A01
        5: {0: {0: -1, 1: 0}, 1: {0: 1, 1: 0}}, # S6: A10 - A00
        6: {0: {0: 0, 1: 1}, 1: {0: 0, 1: -1}}, # S7: A01 - A11
    }
    
    # V coefficients: for each Strassen term t, which (k_local, j_local) pairs have non-zero entries
    # k_local in {0,1}, j_local in {0,1}
    V_coeffs = {
        0: {0: {0: 1, 1: 1}, 1: {0: 0, 1: 1}},  # S1: B00 + B11
        1: {0: {0: 1, 1: 0}, 1: {0: 0, 1: 0}},  # S2: B00
        2: {0: {0: 0, 1: 1}, 1: {0: 0, 1: -1}}, # S3: B01 - B11
        3: {0: {0: -1, 1: 0}, 1: {0: 1, 1: 0}}, # S4: B10 - B00
        4: {0: {0: 0, 1: 0}, 1: {0: 0, 1: 1}},  # S5: B11
        5: {0: {0: 1, 1: 1}, 1: {0: 0, 1: 0}},  # S6: B00 + B01
        6: {0: {0: 0, 1: 0}, 1: {0: 1, 1: 1}},  # S7: B10 + B11
    }
    
    # W coefficients: for each (i, j_local) and each Strassen term t
    # W[j_global * n + i][col]
    # i in {0,1}, j_local in {0,1}
    W_coeffs = {
        (0, 0): [1, 0, 0, 1, -1, 0, 1],   # C00 = S1 + S4 - S5 + S7
        (0, 1): [0, 0, 1, 0, 1, 0, 0],    # C01 = S3 + S5
        (1, 0): [0, 1, 0, 1, 0, 0, 0],    # C10 = S2 + S4
        (1, 1): [1, -1, 1, 0, 0, 1, 0],   # C11 = S1 - S2 + S3 + S6
    }
    
    # The 2x4 * 4x8 multiplication decomposes into:
    # - 2 row groups of A (i in {0,1})
    # - 4 column groups of A / row groups of B (k in {0,1,2,3})
    # - 8 column groups of B / C (j in {0,...,7})
    #
    # We can group k into 2 pairs: {0,1} and {2,3}
    # And group j into 4 pairs: {0,1}, {2,3}, {4,5}, {6,7}
    #
    # So we have 2 (i) * 2 (k_pair) * 4 (j_pair) = 16 sub-problems
    # But wait, each sub-problem is 2x2 * 2x2 -> 2x2
    # Actually, i is already 2 rows, so each (k_pair, j_pair) gives one 2x2*2x2 sub-problem
    # That's 2 * 4 = 8 sub-problems, each with rank 7, total rank 56.
    
    col = 0
    for k_pair in range(2):  # k_pair: 0 or 1, corresponding to k in {2*k_pair, 2*k_pair+1}
        for j_pair in range(4):  # j_pair: 0,1,2,3, corresponding to j in {2*j_pair, 2*j_pair+1}
            for t in range(7):  # Strassen term 0..6
                # U: for i in {0,1}, k_local in {0,1}
                for i in range(2):
                    for k_local in range(2):
                        u_val = U_coeffs[t][i][k_local]
                        if u_val != 0:
                            U[i * m + (2 * k_pair + k_local)][col] += u_val
                
                # V: for k_local in {0,1}, j_local in {0,1}
                for k_local in range(2):
                    for j_local in range(2):
                        v_val = V_coeffs[t][k_local][j_local]
                        if v_val != 0:
                            V[(2 * k_pair + k_local) * p + (2 * j_pair + j_local)][col] += v_val
                
                # W: for i in {0,1}, j_local in {0,1}
                for i in range(2):
                    for j_local in range(2):
                        w_val = W_coeffs[(i, j_local)][t]
                        if w_val != 0:
                            W[(2 * j_pair + j_local) * n + i][col] += w_val
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
