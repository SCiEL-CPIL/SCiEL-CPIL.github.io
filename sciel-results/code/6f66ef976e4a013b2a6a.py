import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 factors
    # Rows: 00, 01, 10, 11
    # Cols: 7 terms
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Block coefficients
    # k: 0..6
    # u,v: 0..1
    alpha = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]]
    ]
    
    beta = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]]
    ]
    
    gamma = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, -1]],
        [[0, 1], [0, 1]],
        [[1, 0], [1, 0]],
        [[-1, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]]
    ]
    
    for k in range(7):
        for s in range(7):
            col = 7 * k + s
            for u in range(2):
                for v in range(2):
                    a_val = alpha[k][u][v]
                    if a_val != 0:
                        for r in range(2):
                            for c in range(2):
                                i = 2 * u + r
                                j = 2 * v + c
                                u_idx = i * m + j
                                U[u_idx][col] += a_val * U2[r * 2 + c][s]
                                
                    b_val = beta[k][u][v]
                    if b_val != 0:
                        for r in range(2):
                            for c in range(2):
                                j = 2 * u + r
                                k_col = 2 * v + c
                                v_idx = j * p + k_col
                                V[v_idx][col] += b_val * V2[r * 2 + c][s]
                                
                    g_val = gamma[k][u][v]
                    if g_val != 0:
                        for r in range(2):
                            for c in range(2):
                                i_row = 2 * u + r
                                k_col = 2 * v + c
                                w_idx = k_col * n + i_row
                                W[w_idx][col] += g_val * W2[c * 2 + r][s]
                                
    return {"U": U, "V": V, "W": W}
