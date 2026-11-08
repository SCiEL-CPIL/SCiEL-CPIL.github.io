def construct():
    import numpy as np
    
    n, m, p = 2, 4, 4
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen 2x2x2 coefficients
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Coefficients for A (u), B (v), C (w)
    # u: indices (i, j) in 0..1
    # v: indices (j, k) in 0..1
    # w: indices (k, i) in 0..1
    
    strassen_u = [
        [1, 1, 1, 0, 1, 1, 0], # a
        [0, 0, 0, 1, 0, -1, 1], # b
        [0, 1, 0, 0, 0, 1, 0], # c
        [1, 1, 0, 1, 0, 0, -1]  # d
    ]
    
    strassen_v = [
        [1, 1, 0, 0, 0, 1, 0], # e
        [0, 0, 1, 0, 0, 1, 0], # f
        [0, 0, 0, 1, 0, 0, 1], # g
        [1, 0, -1, -1, 1, 0, 1] # h
    ]
    
    strassen_w = [
        [1, 0, 0, 0, 0, 0, 0], # c11? No, w maps to output
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M4
        # C21 = M2 + M5
        # C22 = M1 - M2 + M3 + M6
        # w[k, i]
        # i=0, k=0 (C11): 1, 0, 0, -1, 1, 0, -1
        # i=0, k=1 (C12): 0, 0, 1, 1, 0, 0, 0
        # i=1, k=0 (C21): 0, 1, 0, 0, 1, 0, 0
        # i=1, k=1 (C22): 1, -1, 1, 0, 0, 1, 0
    ]
    
    # Correct W mapping for 2x2x2
    # Output C is 2x2.
    # W shape (p*n, rank) = (4, 7).
    # Indices (k, i).
    # k=0, i=0 -> C[0,0]
    # k=1, i=0 -> C[0,1]
    # k=0, i=1 -> C[1,0]
    # k=1, i=1 -> C[1,1]
    
    strassen_w = [
        [1, 0, 0, -1, 1, 0, -1], # k=0, i=0
        [0, 0, 1, 1, 0, 0, 0],   # k=1, i=0
        [0, 1, 0, 0, 1, 0, 0],   # k=0, i=1
        [1, -1, 1, 0, 0, 1, 0]   # k=1, i=1
    ]

    def add_block(u_offset, v_offset, w_offset, start_r):
        # u_offset: (i_start, j_start)
        # v_offset: (j_start, k_start)
        # w_offset: (k_start, i_start)
        
        i_s, j_s = u_offset
        j_s_v, k_s = v_offset
        k_s_w, i_s_w = w_offset
        
        # Verify offsets match
        assert j_s == j_s_v
        assert k_s == k_s_w
        assert i_s == i_s_w

        for r in range(7):
            idx = start_r + r
            
            # Fill U
            for di in range(2):
                for dj in range(2):
                    u_val = strassen_u[di][dj][r]
                    if u_val != 0:
                        U[(i_s + di) * m + (j_s + dj), idx] = u_val
            
            # Fill V
            for dj in range(2):
                for dk in range(2):
                    v_val = strassen_v[dj][dk][r]
                    if v_val != 0:
                        V[(j_s + dj) * p + (k_s + dk), idx] = v_val
            
            # Fill W
            for dk in range(2):
                for di in range(2):
                    w_val = strassen_w[dk][di][r]
                    if w_val != 0:
                        W[(k_s + dk) * n + (i_s + di), idx] = w_val

    # Block 1: i in 0-1, j in 0-1, k in 0-1
    add_block((0, 0), (0, 0), (0, 0), 0)
    
    # Block 2: i in 0-1, j in 0-1, k in 2-3
    add_block((0, 0), (0, 2), (2, 0), 7)
    
    # Block 3: i in 0-1, j in 2-3, k in 0-1
    add_block((0, 2), (2, 0), (0, 0), 14)
    
    # Block 4: i in 0-1, j in 2-3, k in 2-3
    add_block((0, 2), (2, 2), (2, 0), 21)

    return {"U": U, "V": V, "W": W}
