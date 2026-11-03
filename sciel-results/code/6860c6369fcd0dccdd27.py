def construct():
    n, m, p = 4, 5, 6
    
    # Pre-allocate U, V, W with rank 108
    rank = 108
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Strassen tensors for 2x2x2
    # U_r, V_r, W_r are 2x2 matrices
    S_U = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 1]], # M2
        [[1, 0], [0, 0]], # M3
        [[0, 0], [0, 1]], # M4
        [[1, 1], [0, 0]], # M5
        [[-1, 0], [1, 0]],# M6
        [[0, 1], [0, -1]] # M7
    ]
    S_V = [
        [[1, 0], [0, 1]], # M1
        [[1, 0], [0, 0]], # M2
        [[0, 1], [0, -1]],# M3
        [[-1, 0], [1, 0]],# M4
        [[0, 0], [0, 1]], # M5
        [[1, 1], [0, 0]], # M6
        [[0, 0], [1, 1]]  # M7
    ]
    S_W = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, -1]],# M2
        [[0, 1], [0, 1]], # M3
        [[1, 0], [1, 0]], # M4
        [[-1, 1], [0, 0]],# M5
        [[0, 0], [0, -1]],# M6
        [[1, 0], [0, 0]]  # M7
    ]
    
    # Define blocks
    # i in {0, 1}
    # j in {0, 1, 2}
    # k in {0, 1, 2}
    
    for i in range(2):
        for j in range(3):
            for k in range(3):
                if j < 2:
                    # Strassen block: A[i, j] (2x2) * B[j, k] (2x2) -> C[i, k] (2x2)
                    # j is 0 or 1
                    for r in range(7):
                        # Add U_r
                        for u in range(2):
                            for v in range(2):
                                val = S_U[r][u][v]
                                if val != 0:
                                    row_U = (2 * i + u) * m + (2 * j + v)
                                    U[row_U][col_idx] += val
                                    
                        # Add V_r
                        for v in range(2):
                            for w in range(2):
                                val = S_V[r][v][w]
                                if val != 0:
                                    row_V = (2 * j + v) * p + (2 * k + w)
                                    V[row_V][col_idx] += val
                                    
                        # Add W_r
                        for u in range(2):
                            for w in range(2):
                                val = S_W[r][u][w]
                                if val != 0:
                                    row_W = (2 * k + w) * n + (2 * i + u)
                                    W[row_W][col_idx] += val
                                    
                        col_idx += 1
                else:
                    # Schoolbook block: A[i, 2] (2x1) * B[2, k] (1x2) -> C[i, k] (2x2)
                    # j is 2
                    # Rank 4
                    for u in range(2):
                        for w in range(2):
                            # U: 1 at (u, 0) local
                            row_U = (2 * i + u) * m + 4
                            U[row_U][col_idx] = 1
                            
                            # V: 1 at (0, w) local
                            row_V = 4 * p + (2 * k + w)
                            V[row_V][col_idx] = 1
                            
                            # W: 1 at (w, u) local
                            row_W = (2 * k + w) * n + (2 * i + u)
                            W[row_W][col_idx] = 1
                            
                            col_idx += 1
                            
    return {"U": U, "V": V, "W": W}
