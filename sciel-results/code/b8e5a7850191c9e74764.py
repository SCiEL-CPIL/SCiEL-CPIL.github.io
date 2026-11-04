def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2
    # P1: (a+d)(e+h)
    # P2: (c+d)e
    # P3: a(f-h)
    # P4: d(g-e)
    # P5: (a+b)h
    # P6: (c-a)(e+f)
    # P7: (b-d)(g+h)
    
    # Local indices: a=00, b=01, c=10, d=11
    # e=00, f=01, g=10, h=11
    
    # We define the linear forms for A and B for each P
    # A forms: list of (local_idx, coeff)
    # B forms: list of (local_idx, coeff)
    
    # P1
    A_forms = [
        [(0, 1), (3, 1)], # a+d
        [(2, 1), (3, 1)], # c+d
        [(0, 1)],         # a
        [(3, 1)],         # d
        [(0, 1), (1, 1)], # a+b
        [(2, 1), (0, -1)],# c-a
        [(1, 1), (3, -1)] # b-d
    ]
    B_forms = [
        [(0, 1), (3, 1)], # e+h
        [(0, 1)],         # e
        [(1, 1), (3, -1)],# f-h
        [(2, 1), (0, -1)],# g-e
        [(3, 1)],         # h
        [(0, 1), (1, 1)], # e+f
        [(2, 1), (3, 1)]  # g+h
    ]
    
    # W contributions for each P to local C11, C12, C21, C22
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    # Local C indices: 00, 01, 10, 11
    # We map local C to global C indices based on block
    
    # Block definitions:
    # Block 1: A_L (j=0,1), B_LL (j=0,1, k=0,1), C cols 0,1
    # Block 2: A_R (j=2,3), B_RL (j=2,3, k=0,1), C cols 0,1
    # Block 3: A_L (j=0,1), B_LR (j=0,1, k=2,3), C cols 2,3
    # Block 4: A_R (j=2,3), B_RR (j=2,3, k=2,3), C cols 2,3
    
    blocks = [
        {'a_j': [0, 1], 'b_j': [0, 1], 'b_k': [0, 1], 'c_col_offset': 0},
        {'a_j': [2, 3], 'b_j': [2, 3], 'b_k': [0, 1], 'c_col_offset': 0},
        {'a_j': [0, 1], 'b_j': [0, 1], 'b_k': [2, 3], 'c_col_offset': 2},
        {'a_j': [2, 3], 'b_j': [2, 3], 'b_k': [2, 3], 'c_col_offset': 2}
    ]
    
    col_idx = 0
    for block in blocks:
        a_j0, a_j1 = block['a_j']
        b_j0, b_j1 = block['b_j']
        b_k0, b_k1 = block['b_k']
        c_off = block['c_col_offset']
        
        # Local A indices mapping:
        # 00 -> i=0, j=a_j0
        # 01 -> i=0, j=a_j1
        # 10 -> i=1, j=a_j0
        # 11 -> i=1, j=a_j1
        a_map = {
            0: (0, a_j0),
            1: (0, a_j1),
            2: (1, a_j0),
            3: (1, a_j1)
        }
        
        # Local B indices mapping:
        # 00 -> j=b_j0, k=b_k0
        # 01 -> j=b_j0, k=b_k1
        # 10 -> j=b_j1, k=b_k0
        # 11 -> j=b_j1, k=b_k1
        b_map = {
            0: (b_j0, b_k0),
            1: (b_j0, b_k1),
            2: (b_j1, b_k0),
            3: (b_j1, b_k1)
        }
        
        # Local C indices mapping to global C indices
        # Local 00 -> row 0, col c_off
        # Local 01 -> row 0, col c_off+1
        # Local 10 -> row 1, col c_off
        # Local 11 -> row 1, col c_off+1
        # Global index = k*n + i = col*2 + row
        c_map = {
            0: (c_off, 0),       # 00
            1: (c_off+1, 0),     # 01
            2: (c_off, 1),       # 10
            3: (c_off+1, 1)      # 11
        }
        
        # W coefficients for each P (0..6) to each local C (0..3)
        # P0 (P1): C00=1, C11=1
        # P1 (P2): C10=1, C11=-1
        # P2 (P3): C01=1, C11=1
        # P3 (P4): C00=1, C10=1
        # P4 (P5): C00=-1, C01=1
        # P5 (P6): C11=1
        # P6 (P7): C00=1
        w_coeffs = [
            {0: 1, 3: 1},       # P1
            {2: 1, 3: -1},      # P2
            {1: 1, 3: 1},       # P3
            {0: 1, 2: 1},       # P4
            {0: -1, 1: 1},      # P5
            {3: 1},             # P6
            {0: 1}              # P7
        ]
        
        for s in range(7):
            # Fill U
            for local_idx, coeff in A_forms[s]:
                i, j = a_map[local_idx]
                U[i * m + j][col_idx] += coeff
            
            # Fill V
            for local_idx, coeff in B_forms[s]:
                j, k = b_map[local_idx]
                V[j * p + k][col_idx] += coeff
            
            # Fill W
            for local_c, coeff in w_coeffs[s].items():
                k, i = c_map[local_c]
                W[k * n + i][col_idx] += coeff
            
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}
