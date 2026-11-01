import time

DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2x2
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B10-B11)
    # M4: A11(B01-B00)
    # M5: (A00+A10)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # U factors (A indices: 00, 01, 10, 11)
    U_factors = [
        [1, 0, 1, 0, 1, -1, 1],  # A00
        [0, 0, 0, 0, 0, 0, -1],  # A01
        [0, 1, 0, 0, 1, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ]
    
    # V factors (B indices: 00, 01, 10, 11)
    # M1: B00, B11
    # M2: B00
    # M3: B10, -B11
    # M4: B01, -B00
    # M5: B11
    # M6: B00, B01
    # M7: B10, B11
    V_factors = [
        {0: 1, 3: 1},  # M1
        {0: 1},         # M2
        {2: 1, 3: -1},  # M3
        {1: 1, 0: -1},  # M4
        {3: 1},         # M5
        {0: 1, 1: 1},   # M6
        {2: 1, 3: 1}    # M7
    ]
    
    # W factors (C indices: 00, 01, 10, 11)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M2 + M3 - M6 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    W_factors = [
        {0: 1, 3: 1},       # M1: C00, C11
        {1: 1, 2: 1, 3: -1},# M2: C01, C10, -C11
        {1: 1, 3: 1},       # M3: C01, C11
        {0: -1, 2: 1},      # M4: -C00, C10
        {0: 1},             # M5: C00
        {1: -1},            # M6: -C01
        {0: -1, 1: 1, 3: 1} # M7: -C00, C01, C11
    ]

    # Map local indices to global
    # U: i in {0,1}, j in {0,1} -> idx = i*m + j = i*2 + j
    # V: j in {0,1}, k in {0,1} local -> idx = j*p + k_global
    # W: k in {0,1} local, i in {0,1} -> idx = k_global*n + i
    
    for l in range(6):
        col_start = l * 7
        for r in range(7):
            col = col_start + r
            # U
            for i in range(2):
                for j in range(2):
                    u_idx = i * 2 + j
                    U[u_idx][col] = U_factors[i*2+j][r]
            
            # V
            for local_idx, val in V_factors[r].items():
                j, k_loc = local_idx // 2, local_idx % 2
                global_k = 2 * l + k_loc
                v_idx = j * 12 + global_k
                V[v_idx][col] = val
            
            # W
            for local_idx, val in W_factors[r].items():
                k_loc, i = local_idx // 2, local_idx % 2
                global_k = 2 * l + k_loc
                w_idx = global_k * 2 + i
                W[w_idx][col] = val

    return {"U": U, "V": "W": W}
