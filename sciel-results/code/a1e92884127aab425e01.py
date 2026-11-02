def construct():
    n, m, p = DIMENSIONS
    u_dim = n * m
    v_dim = m * p
    w_dim = p * n
    
    rank = 14
    U = [[0] * rank for _ in range(u_dim)]
    V = [[0] * rank for _ in range(v_dim)]
    W = [[0] * rank for _ in range(w_dim)]
    
    # Define Strassen terms for 2x2x2
    # Each term: (u_entries, v_entries, w_entries)
    # u_entries: [(i, j, val), ...]
    # v_entries: [(j, k, val), ...]
    # w_entries: [(k, i, val), ...]
    
    strassen_terms = [
        # M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1), (1,1,-1)]),
        # M3
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
        # M4
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        # M5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(1,0,1), (0,0,-1)]),
        # M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
    ]
    
    # Fill first block (k=0,1)
    for r in range(7):
        u_entries, v_entries, w_entries = strassen_terms[r]
        col = r
        for i, j, val in u_entries:
            U[i * m + j][col] += val
        for j, k, val in v_entries:
            V[j * p + k][col] += val
        for k, i, val in w_entries:
            W[k * n + i][col] += val
            
    # Fill second block (k=2,3)
    # This corresponds to multiplying A by the second half of B (cols 2,3)
    # Effectively, we shift the k-index in V and W by 2.
    for r in range(7):
        u_entries, v_entries, w_entries = strassen_terms[r]
        col = 7 + r
        for i, j, val in u_entries:
            U[i * m + j][col] += val
        for j, k, val in v_entries:
            k_new = k + 2
            V[j * p + k_new][col] += val
        for k, i, val in w_entries:
            k_new = k + 2
            W[k_new * n + i][col] += val

    return {"U": U, "V": V, "W": W}
