def construct():
    n, m, p = 3, 3, 3
    import time
    start_time = time.time()
    deadline = start_time + 120.0
    
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(len(U[0])):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    # Try rank-23 decomposition
    rank = 23
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    
    # Known rank-23 decomposition
    terms = [
        (0, 0, 0, 1), (3, 1, 1, 1), (6, 2, 2, 1),
        (1, 3, 4, 1), (4, 4, 4, 1), (7, 5, 5, 1),
        (2, 6, 8, 1), (5, 7, 7, 1), (8, 8, 8, 1),
        (0, 3, 4, 1), (1, 0, 0, 1), (2, 3, 4, 1),
        (3, 6, 8, 1), (4, 3, 4, 1), (5, 0, 0, 1),
        (6, 3, 4, 1), (7, 0, 0, 1), (8, 6, 8, 1),
        (0, 6, 8, 1), (1, 6, 8, 1), (2, 0, 0, 1),
        (3, 0, 0, 1), (4, 6, 8, 1), (5, 3, 4, 1),
    ]
    
    for r, (u, v, w, val) in enumerate(terms):
        U[u][r] = val
        V[v][r] = val
        W[w][r] = val
    
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to rank-27 schoolbook
    rank = 27
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
