def construct():
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    def add_term(u_entries, v_entries, w_entries):
        nonlocal r
        for idx, val in u_entries:
            U[idx][r] += val
        for idx, val in v_entries:
            V[idx][r] += val
        for idx, val in w_entries:
            W[idx][r] += val
        r += 1
    
    # Strassen block: j={0,1}, k={0,1} (7 terms)
    add_term([(0, 1), (4, 1)], [(0, 1), (7, 1)], [(0, 1), (5, 1)])
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (5, -1)])
    add_term([(0, 1)], [(1, 1), (7, -1)], [(2, 1), (5, 1)])
    add_term([(4, 1)], [(6, 1), (0, -1)], [(0, 1), (1, 1)])
    add_term([(0, 1), (1, 1)], [(7, 1)], [(0, -1), (2, 1)])
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(5, 1)])
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(0, 1)])
    
    # Strassen block: j={0,1}, k={2,3} (7 terms)
    add_term([(0, 1), (4, 1)], [(2, 1), (8, 1)], [(4, 1), (7, 1)])
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)])
    add_term([(0, 1)], [(3, 1), (8, -1)], [(6, 1), (7, 1)])
    add_term([(4, 1)], [(7, 1), (2, -1)], [(4, 1), (5, 1)])
    add_term([(0, 1), (1, 1)], [(8, 1)], [(4, -1), (6, 1)])
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)])
    add_term([(1, 1), (4, -1)], [(7, 1), (8, 1)], [(4, 1)])
    
    # k=4 column for j={0,1} (4 terms)
    add_term([(0, 1)], [(4, 1)], [(8, 1)])
    add_term([(1, 1)], [(4, 1)], [(9, 1)])
    add_term([(3, 1)], [(9, 1)], [(8, 1)])
    add_term([(4, 1)], [(9, 1)], [(9, 1)])
    
    # j=2, all k (10 terms)
    for i in range(n):
        for k in range(p):
            add_term([(i * m + 2, 1)], [(2 * p + k, 1)], [(k * n + i, 1)])
    
    return {"U": U, "V": V, "W": W}
