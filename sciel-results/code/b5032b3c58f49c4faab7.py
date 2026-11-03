def construct():
    import time
    start = time.time()
    deadline = start + 145.0

    n, m, p = 3, 4, 8
    rank = 88
    
    U = [[0]*rank for _ in range(n*m)]  # 12 x 88
    V = [[0]*rank for _ in range(m*p)]  # 32 x 88
    W = [[0]*rank for _ in range(p*n)]  # 24 x 88
    
    col = 0
    
    # Strassen 2x2x2 factors
    # U_local: 4x7, V_local: 4x7, W_local: 4x7
    # A indices: [A00, A01, A10, A11] → rows 0,1,2,3
    # B indices: [B00, B01, B10, B11] → rows 0,1,2,3
    # C indices: [C00, C01, C10, C11] → rows 0,1,2,3
    
    strassen_U = [
        [1, 0, 0, 1],  # M1: A00+A11
        [0, 0, 1, 1],  # M2: A10+A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00+A01
        [-1, 0, 1, 0], # M6: A10-A00
        [0, 1, 0, -1], # M7: A01-A11
    ]
    strassen_V = [
        [1, 0, 0, 1],  # M1: B00+B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01-B11
        [0, 0, 1, -1], # M4: B10-B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00+B01
        [0, 0, 1, 1],  # M7: B10+B11
    ]
    strassen_W = [
        [1, 0, 0, 1],  # M1: C00+C11
        [0, 0, 1, -1], # M2: C10-C11
        [0, 1, 0, 1],  # M3: C01+C11
        [1, 0, 1, 0],  # M4: C00+C10
        [-1, 1, 0, 0], # M5: -C00+C01
        [0, 0, 0, 1],  # M6: C11
        [1, 0, 0, 0],  # M7: C00
    ]
    
    # Process 8 groups of 2x2x2 (i in {0,1})
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            i0 = 0
            for s in range(7):
                for li in range(4):
                    i_local = li // 2  # 0 or 1
                    j_local = li % 2   # 0 or 1
                    i_global = i0 + i_local
                    j_global = j0 + j_local
                    U[i_global * m + j_global][col] = strassen_U[li][s]
                for lj in range(4):
                    j_local = lj // 2
                    k_local = lj % 2
                    j_global = j0 + j_local
                    k_global = k0 + k_local
                    V[j_global * p + k_global][col] = strassen_V[lj][s]
                for lc in range(4):
                    i_local = lc // 2
                    k_local = lc % 2
                    i_global = i0 + i_local
                    k_global = k0 + k_local
                    W[k_global * n + i_global][col] = strassen_W[lc][s]
                col += 1
    
    # Process 8 groups of 1x2x2 (i=2)
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            i = 2
            for j_local in range(2):
                for k_local in range(2):
                    j_global = j0 + j_local
                    k_global = k0 + k_local
                    U[i * m + j_global][col] = 1
                    V[j_global * p + k_global][col] = 1
                    W[k_global * n + i][col] = 1
                    col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
