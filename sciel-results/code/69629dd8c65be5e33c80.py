DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    rank = 52
    U = [[0] * rank for _ in range(n * m)]   # 8 x 52
    V = [[0] * rank for _ in range(m * p)]   # 28 x 52  
    W = [[0] * rank for _ in range(p * n)]   # 14 x 52
    
    col = 0
    
    # Strassen 2x2x2 templates
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    
    S_V = [
        [1, 1, 1, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, -1, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    
    S_W = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    
    # 2x4x4 part: 4 Strassen blocks, each 2x2x2
    for a_idx in range(2):
        for b_row in range(2):
            for b_col in range(2):
                for s in range(7):
                    for li in range(2):
                        for lj in range(2):
                            gi = li
                            gj = 2 * a_idx + lj
                            val = S_U[s][li * 2 + lj]
                            if val != 0:
                                U[gi * m + gj][col] += val
                    
                    for lj in range(2):
                        for lk in range(2):
                            gj = 2 * b_row + lj
                            gk = 2 * b_col + lk
                            val = S_V[s][lj * 2 + lk]
                            if val != 0:
                                V[gj * p + gk][col] += val
                    
                    for lk in range(2):
                        for li in range(2):
                            gk = 2 * b_col + lk
                            gi = li
                            val = S_W[s][lk * 2 + li]
                            if val != 0:
                                W[gk * n + gi][col] += val
                    
                    col += 1
    
    # 2x4x3 part: schoolbook for k=4,5,6
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
