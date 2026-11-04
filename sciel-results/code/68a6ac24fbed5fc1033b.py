def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms (u, v, c)
    # u, v, c are lists of 4 coefficients
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    # Block definitions: (A_map, B_map, C_map)
    # Map is a list of 4 global indices corresponding to local 0,1,2,3
    blocks = [
        # Block 1: A11 * B11 -> C1
        ([0, 1, 4, 5], [0, 1, 4, 5], [0, 1, 4, 5]),
        # Block 2: A12 * B21 -> C1
        ([2, 3, 6, 7], [8, 9, 12, 13], [0, 1, 4, 5]),
        # Block 3: A11 * B12 -> C2
        ([0, 1, 4, 5], [2, 3, 6, 7], [2, 3, 6, 7]),
        # Block 4: A12 * B22 -> C2
        ([2, 3, 6, 7], [10, 11, 14, 15], [2, 3, 6, 7])
    ]
    
    col = 0
    for a_map, b_map, c_map in blocks:
        for u, v, c in strassen_terms:
            for l in range(4):
                if u[l] != 0:
                    U[a_map[l]][col] += u[l]
                if v[l] != 0:
                    V[b_map[l]][col] += v[l]
                if c[l] != 0:
                    W[c_map[l]][col] += c[l]
            col += 1
            
    return {"U": U, "V": V, "W": W}
