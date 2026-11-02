"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 4 blocks of 2x2 (Strassen, rank 7) -> 28
    # 1 block of 2x2 (Strassen, rank 7) -> 7
    # 1 block of 2x1 (Schoolbook, rank 4) -> 4
    # Total rank = 39
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        for i in range(len(u_vec)):
            if u_vec[i] != 0:
                U[i][col] += u_vec[i]
        for i in range(len(v_vec)):
            if v_vec[i] != 0:
                V[i][col] += v_vec[i]
        for i in range(len(w_vec)):
            if w_vec[i] != 0:
                W[i][col] += w_vec[i]
        col += 1

    # Strassen terms for a 2x2 block
    # u, v are 4-dim (flattened 2x2)
    # w is 4-dim (flattened 2x2, local index i*2+k)
    strassen_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    strassen_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_w = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    # Process 4 blocks of size 2
    for block_idx in range(4):
        k_offset = block_idx * 2
        for t in range(7):
            u_local = strassen_u[t]
            v_local = strassen_v[t]
            w_local = strassen_w[t]
            
            u_global = [0] * (n*m)
            v_global = [0] * (m*p)
            w_global = [0] * (p*n)
            
            # Map u
            for li in range(4):
                if u_local[li] != 0:
                    i = li // 2
                    j = li % 2
                    u_global[i*m + j] = u_local[li]
            
            # Map v
            for li in range(4):
                if v_local[li] != 0:
                    j = li // 2
                    k = li % 2
                    v_global[j*p + (k_offset + k)] = v_local[li]
            
            # Map w
            for li in range(4):
                if w_local[li] != 0:
                    i = li // 2
                    k = li % 2
                    w_global[(k_offset + k)*n + i] = w_local[li]
            
            add_term(u_global, v_global, w_global)

    # Process 1 block of size 2 (cols 8, 9)
    k_offset = 8
    for t in range(7):
        u_local = strassen_u[t]
        v_local = strassen_v[t]
        w_local = strassen_w[t]
        
        u_global = [0] * (n*m)
        v_global = [0] * (m*p)
        w_global = [0] * (p*n)
        
        for li in range(4):
            if u_local[li] != 0:
                i = li // 2
                j = li % 2
                u_global[i*m + j] = u_local[li]
        
        for li in range(4):
            if v_local[li] != 0:
                j = li // 2
                k = li % 2
                v_global[j*p + (k_offset + k)] = v_local[li]
        
        for li in range(4):
            if w_local[li] != 0:
                i = li // 2
                k = li % 2
                w_global[(k_offset + k)*n + i] = w_local[li]
        
        add_term(u_global, v_global, w_global)

    # Process 1 block of size 1 (col 10)
    k_offset = 10
    # Schoolbook for 2x2 * 2x1
    # C_i = A_i0 B_0 + A_i1 B_1
    terms_2x1 = [
        # u, v, w
        ([1, 0, 0, 0], [1, 0], [1, 0]), # A00 B0 -> C0
        ([0, 1, 0, 0], [0, 1], [1, 0]), # A01 B1 -> C0
        ([0, 0, 1, 0], [1, 0], [0, 1]), # A10 B0 -> C1
        ([0, 0, 0, 1], [0, 1], [0, 1])  # A11 B1 -> C1
    ]
    
    for u_local, v_local, w_local in terms_2x1:
        u_global = [0] * (n*m)
        v_global = [0] * (m*p)
        w_global = [0] * (p*n)
        
        for li in range(4):
            if u_local[li] != 0:
                i = li // 2
                j = li % 2
                u_global[i*m + j] = u_local[li]
        
        for li in range(2):
            if v_local[li] != 0:
                j = li
                v_global[j*p + k_offset] = v_local[li]
        
        for li in range(2):
            if w_local[li] != 0:
                i = li
                w_global[k_offset*n + i] = w_local[li]
        
        add_term(u_global, v_global, w_global)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
