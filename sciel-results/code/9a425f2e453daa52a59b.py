"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank calculation:
    # Block 1 (2x2x6): 3 * 7 = 21
    # Block 2 (2x3x6): 3 * (4 + 7) = 33
    # Block 3 (1x5x6): 30
    # Total: 84
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add a decomposition block to U, V, W
    def add_block(U_local, V_local, W_local, n_map, m_map, p_map, r):
        # U_local is (n_local, r)
        # V_local is (m_local, r)
        # W_local is (p_local, r)
        # n_map: list of global n indices corresponding to local n 0..n_local-1
        # m_map: list of global m indices
        # p_map: list of global p indices
        
        for i_local in range(len(n_map)):
            for c in range(r):
                if U_local[i_local][c] != 0:
                    U[n_map[i_local] * m + m_map[0]][col_offset + c] += U_local[i_local][c] # Wait, U index is i*m + j?
                    # U is indexed by [i*m + j] in seed?
                    # Seed: U[i * m + j][column]
                    # So U index is i*m + j.
                    # My n_map gives i. m_map gives j.
                    pass
        
        # Correct indexing logic:
        # U is size (n*m) x rank. Index is i*m + j.
        # V is size (m*p) x rank. Index is j*p + k.
        # W is size (p*n) x rank. Index is k*n + i.
        
        for i_local in range(len(n_map)):
            for j_local in range(len(m_map)):
                for c in range(r):
                    val = U_local[i_local][j_local * 1 + 0] if len(U_local[0]) == 1 else 0 # U_local shape?
                    # U_local should be flattened?
                    # Let's represent local U as (n_local, m_local, r) or similar?
                    # Better: U_local[i_local, j_local, c]
                    pass
                    
    # Let's re-implement more directly.
    
    def add_strassen_2x2x2(n_idx, m_idx, p_idx, start_col):
        # n_idx: list of 2 global n indices
        # m_idx: list of 2 global m indices
        # p_idx: list of 2 global p indices
        # Returns number of columns used (7)
        
        # Strassen matrices for 2x2x2
        # U is 4x7, V is 4x7, W is 4x7
        # Local indices:
        # n: 0, 1
        # m: 0, 1
        # p: 0, 1
        
        # U[i, j, c] -> Global U[i_global * m + j_global]
        # V[j, k, c] -> Global V[j_global * p + k_global]
        # W[k, i, c] -> Global W[k_global * n + i_global]
        
        # Strassen coefficients
        # M1 = (a10+a11)(b00+b01)
        # M2 = (a10+a11)b10
        # M3 = a00(b10-b11)
        # M4 = a11(b00-b10)
        # M5 = (a00+a01)b11
        # M6 = (a01-a10)(b00+b10)
        # M7 = (a11-a01)(b01+b11)
        
        # U vectors (a part)
        u1 = [0, 0, 1, 1] # a10, a11 (flattened a: a00, a01, a10, a11)
        u2 = [0, 0, 1, 1]
        u3 = [1, 0, 0, 0] # a00
        u4 = [0, 0, 0, 1] # a11
        u5 = [1, 1, 0, 0] # a00, a01
        u6 = [0, 1, -1, 0] # a01, a10
        u7 = [0, -1, 1, 0] # a11, a01
        
        # V vectors (b part)
        # b flattened: b00, b01, b10, b11
        v1 = [1, 1, 0, 0] # b00, b01
        v2 = [0, 0, 1, 0] # b10
        v3 = [0, -1, 1, 0] # b10, b11
        v4 = [1, 0, -1, 0] # b00, b10
        v5 = [0, 0, 0, 1] # b11
        v6 = [1, 0, 1, 0] # b00, b10
        v7 = [0, 1, 0, 1] # b01, b11
        
        # W vectors (c part)
        # c flattened: c00, c01, c10, c11
        w1 = [1, 0, 0, 1] # c00, c11
        w2 = [0, 0, 1, -1] # c10, c11
        w3 = [0, 1, 0, 1] # c01, c11
        w4 = [1, 0, 1, 0] # c00, c10
        w5 = [-1, 1, 0, -1] # c00, c01, c11
        w6 = [0, 0, 0, 1] # c11
        w7 = [1, 0, 0, 0] # c00
        
        # Wait, let's verify W coefficients for Strassen.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # W[c, m] where c is col, m is output index (0..3)
        # Output 0 (c00): M1:1, M4:1, M5:-1, M7:1
        # Output 1 (c01): M3:1, M5:1
        # Output 2 (c10): M2:1, M4:1
        # Output 3 (c11): M1:1, M2:-1, M3:1, M6:1
        
        w_mat = [
            [1, 0, 0, 1], # M1, M2, M3, M4... no, W is (p*n, rank).
            # W[k, i, c].
            # Let's build W explicitly for the 4 outputs.
        ]
        
        # W for c00 (k=0, i=0): [1, 0, 0, 1, -1, 0, 1] (M1, M2, M3, M4, M5, M6, M7)
        # W for c01 (k=1, i=0): [0, 0, 1, 0, 1, 0, 0]
        # W for c10 (k=0, i=1): [0, 1, 0, 1, 0, 0, 0]
        # W for c11 (k=1, i=1): [1, -1, 1, 0, 0, 1, 0]
        
        w_vals = [
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ]
        
        # Apply to global matrices
        for c in range(7):
            # U part
            for i_local in range(2):
                for j_local in range(2):
                    # U_local[i_local, j_local] value
                    # u vectors are defined for flattened a (a00, a01, a10, a11)
                    # Index in u vector: i_local * 2 + j_local
                    idx_u = i_local * 2 + j_local
                    val_u = 0
                    if c == 0: val_u = u1[idx_u]
                    elif c == 1: val_u = u2[idx_u]
                    elif c == 2: val_u = u3[idx_u]
                    elif c == 3: val_u = u4[idx_u]
                    elif c == 4: val_u = u5[idx_u]
                    elif c == 5: val_u = u6[idx_u]
                    elif c == 6: val_u = u7[idx_u]
                    
                    if val_u != 0:
                        U[n_idx[i_local] * m + m_idx[j_local]][start_col + c] += val_u
            
            # V part
            for j_local in range(2):
                for k_local in range(2):
                    idx_v = j_local * 2 + k_local
                    val_v = 0
                    if c == 0: val_v = v1[idx_v]
                    elif c == 1: val_v = v2[idx_v]
                    elif c == 2: val_v = v3[idx_v]
                    elif c == 3: val_v = v4[idx_v]
                    elif c == 4: val_v = v5[idx_v]
                    elif c == 5: val_v = v6[idx_v]
                    elif c == 6: val_v = v7[idx_v]
                    
                    if val_v != 0:
                        V[m_idx[j_local] * p + p_idx[k_local]][start_col + c] += val_v
            
            # W part
            for k_local in range(2):
                for i_local in range(2):
                    idx_w = k_local * 2 + i_local
                    val_w = w_vals[idx_w][c]
                    
                    if val_w != 0:
                        W[p_idx[k_local] * n + n_idx[i_local]][start_col + c] += val_w
                        
        return 7

    def add_schoolbook_2x1x2(n_idx, m_idx, p_idx, start_col):
        # 2x1x2 schoolbook. Rank 4.
        # n: 2, m: 1, p: 2
        # U: 2x1 -> 2 rows, 1 col.
        # V: 1x2 -> 1 row, 2 cols.
        # W: 2x2 -> 2 rows, 2 cols.
        
        # Multiplications:
        # c00 = a00 * b00
        # c01 = a00 * b01
        # c10 = a10 * b00
        # c11 = a10 * b01
        
        # U vectors (a):
        # c0: a00 (idx 0) -> [1, 0]
        # c1: a00 (idx 0) -> [1, 0]
        # c2: a10 (idx 1) -> [0, 1]
        # c3: a10 (idx 1) -> [0, 1]
        
        # V vectors (b):
        # c0: b00 (idx 0) -> [1]
        # c1: b01 (idx 1) -> [0]
        # c2: b00 (idx 0) -> [1]
        # c3: b01 (idx 1) -> [0]
        
        # W vectors (c):
        # c0: c00 (idx 0) -> 1
        # c1: c01 (idx 1) -> 1
        # c2: c10 (idx 2) -> 1
        # c3: c11 (idx 3) -> 1
        
        for c in range(4):
            # U
            if c == 0 or c == 1:
                U[n_idx[0] * m + m_idx[0]][start_col + c] += 1
            else:
                U[n_idx[1] * m + m_idx[0]][start_col + c] += 1
            
            # V
            if c == 0 or c == 2:
                V[m_idx[0] * p + p_idx[0]][start_col + c] += 1
            else:
                V[m_idx[0] * p + p_idx[1]][start_col + c] += 1
            
            # W
            if c == 0:
                W[p_idx[0] * n + n_idx[0]][start_col + c] += 1
            elif c == 1:
                W[p_idx[1] * n + n_idx[0]][start_col + c] += 1
            elif c == 2:
                W[p_idx[0] * n + n_idx[1]][start_col + c] += 1
            else:
                W[p_idx[1] * n + n_idx[1]][start_col + c] += 1
        
        return 4

    def add_schoolbook_1x5x6(n_idx, m_idx, p_idx, start_col):
        # 1x5x6 schoolbook. Rank 30.
        # n: 1, m: 5, p: 6.
        for j in range(5):
            for k in range(6):
                U[n_idx[0] * m + m_idx[j]][start_col + j * 6 + k] = 1
                V[m_idx[j] * p + p_idx[k]][start_col + j * 6 + k] = 1
                W[p_idx[k] * n + n_idx[0]][start_col + j * 6 + k] = 1
        return 30

    # 1. Block 1: n=2, m=2, p=6 (Strassen x3)
    n_idx_1 = [0, 1]
    m_idx_1 = [0, 1]
    for t in range(3):
        p_idx_1 = [2 * t, 2 * t + 1]
        cols = add_strassen_2x2x2(n_idx_1, m_idx_1, p_idx_1, col_offset)
        col_offset += cols

    # 2. Block 2: n=2, m=3, p=6
    n_idx_2 = [0, 1]
    # m=3 split into m=1 (global idx 2) and m=2 (global idx 3,4)
    m_idx_2a = [2]
    m_idx_2b = [3, 4]
    
    for t in range(3):
        p_idx_2 = [2 * t, 2 * t + 1]
        # 2a: 2x1x2
        cols_a = add_schoolbook_2x1x2(n_idx_2, m_idx_2a, p_idx_2, col_offset)
        col_offset += cols_a
        # 2b: 2x2x2
        cols_b = add_strassen_2x2x2(n_idx_2, m_idx_2b, p_idx_2, col_offset)
        col_offset += cols_b

    # 3. Block 3: n=1, m=5, p=6
    n_idx_3 = [2]
    m_idx_3 = [0, 1, 2, 3, 4]
    p_idx_3 = [0, 1, 2, 3, 4, 5]
    cols_3 = add_schoolbook_1x5x6(n_idx_3, m_idx_3, p_idx_3, col_offset)
    col_offset += cols_3

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
