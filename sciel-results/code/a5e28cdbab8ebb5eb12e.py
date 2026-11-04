def construct():
    import numpy as np

    n, m, p = 3, 4, 8
    
    # Strassen terms for 2x2x2
    # u, v, w are lists of 4 floats
    # Indexing: u[i*2+j], v[j*2+k], w[k*2+i]
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]), # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 1]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1])  # M7
    ]

    U = []
    V = []
    W = []

    def add_term(u_vals, v_vals, w_vals):
        U.append(u_vals)
        V.append(v_vals)
        W.append(w_vals)

    # Helper to add a Strassen block
    # i_offset: global i offset (0 for top, 2 for bottom? No, Strassen is for 2x2)
    # j_offset: global j offset
    # k_offset: global k offset
    def add_strassen_block(i_off, j_off, k_off):
        for u, v, w in strassen_terms:
            # Create global vectors
            u_glob = [0] * (n * m) # 12
            v_glob = [0] * (m * p) # 32
            w_glob = [0] * (p * n) # 24
            
            for li in range(2):
                for lj in range(2):
                    idx_u = li * 2 + lj
                    val = u[idx_u]
                    if val != 0:
                        gi = i_off + li
                        gj = j_off + lj
                        u_glob[gi * m + gj] += val
                        
            for lj in range(2):
                for lk in range(2):
                    idx_v = lj * 2 + lk
                    val = v[idx_v]
                    if val != 0:
                        gj = j_off + lj
                        gk = k_off + lk
                        v_glob[gj * p + gk] += val
                        
            for lk in range(2):
                for li in range(2):
                    idx_w = lk * 2 + li
                    val = w[idx_w]
                    if val != 0:
                        gk = k_off + lk
                        gi = i_off + li
                        w_glob[gk * n + gi] += val
                        
            add_term(u_glob, v_glob, w_glob)

    # Block 1: k in 0..3
    # Top 2 rows (i in 0..1)
    # P1: A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen_block(0, 0, 0)
    # P2: A[0:2, 2:4] * B[2:4, 0:2]
    add_strassen_block(0, 2, 0)
    # P3: A[0:2, 0:2] * B[0:2, 2:4]
    add_strassen_block(0, 0, 2)
    # P4: A[0:2, 2:4] * B[2:4, 2:4]
    add_strassen_block(0, 2, 2)
    
    # Bottom row (i=2)
    for j in range(4):
        u_glob = [0] * (n * m)
        v_glob = [0] * (m * p)
        w_glob = [0] * (p * n)
        
        u_glob[2 * m + j] = 1
        
        for k in range(4):
            v_glob[j * p + k] = 1
            w_glob[k * n + 2] = 1
            
        add_term(u_glob, v_glob, w_glob)

    # Block 2: k in 4..7
    # Top 2 rows (i in 0..1)
    # P1: A[0:2, 0:2] * B[0:2, 4:6]
    add_strassen_block(0, 0, 4)
    # P2: A[0:2, 2:4] * B[2:4, 4:6]
    add_strassen_block(0, 2, 4)
    # P3: A[0:2, 0:2] * B[0:2, 6:8]
    add_strassen_block(0, 0, 6)
    # P4: A[0:2, 2:4] * B[2:4, 6:8]
    add_strassen_block(0, 2, 6)
    
    # Bottom row (i=2)
    for j in range(4):
        u_glob = [0] * (n * m)
        v_glob = [0] * (m * p)
        w_glob = [0] * (p * n)
        
        u_glob[2 * m + j] = 1
        
        for k in range(4, 8):
            v_glob[j * p + k] = 1
            w_glob[k * n + 2] = 1
            
        add_term(u_glob, v_glob, w_glob)

    return {"U": U, "V": V, "W": W}
