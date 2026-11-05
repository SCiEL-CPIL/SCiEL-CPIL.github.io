def construct():
    n, m, p = DIMENSIONS
    
    # Generate all partitions
    partitions_n = list(get_partitions(n))
    partitions_m = list(get_partitions(m))
    partitions_p = list(get_partitions(p))
    
    # Find the best partition
    best_rank = float('inf')
    best_partitions = None
    for pn in partitions_n:
        for pm in partitions_m:
            for pp in partitions_p:
                r = calculate_rank(pn, pm, pp)
                if r < best_rank:
                    best_rank = r
                    best_partitions = (pn, pm, pp)
    
    if best_partitions is None:
        best_partitions = ([n], [m], [p])
        best_rank = n * m * p
    
    pn, pm, pp = best_partitions
    
    # Calculate start indices
    n_starts = [0] * len(pn)
    m_starts = [0] * len(pm)
    p_starts = [0] * len(pp)
    for i in range(1, len(pn)):
        n_starts[i] = n_starts[i-1] + pn[i-1]
    for j in range(1, len(pm)):
        m_starts[j] = m_starts[j-1] + pm[j-1]
    for k in range(1, len(pp)):
        p_starts[k] = p_starts[k-1] + pp[k-1]
    
    # Initialize U, V, W
    U = [[0] * best_rank for _ in range(n * m)]
    V = [[0] * best_rank for _ in range(m * p)]
    W = [[0] * best_rank for _ in range(p * n)]
    
    rank_idx = 0
    
    strassen_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(0,1): 1, (1,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    for i, n_i in enumerate(pn):
        for j, m_j in enumerate(pm):
            for k, p_k in enumerate(pp):
                if n_i == 2 and m_j == 2 and p_k == 2:
                    for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
                        for (a, b), u_val in u_coeffs.items():
                            for (b2, c), v_coeffs.items():
                                if b == b2:
                                    for (c2, a2), w_coeffs.items():
                                        if c == c2 and a == a2:
                                            U[n_starts[i] + a][rank_idx] = u_val
                                            V[m_starts[j] + b][rank_idx] = v_val
                                            W[(p_starts[k] + c) * n + n_starts[i] + a][rank_idx] = w_val
                        rank_idx += 1
                else:
                    for a in range(n_i):
                        for b in range(m_j):
                            for c in range(p_k):
                                U[n_starts[i] + a][rank_idx] = 1
                                V[m_starts[j] + b][rank_idx] = 1
                                W[(p_starts[k] + c) * n + n_starts[i] + a][rank_idx] = 1
                                rank_idx += 1
    
    return {"U": U, "V": V, "W": W}
