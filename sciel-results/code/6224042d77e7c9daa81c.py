import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    DEADLINE = 115  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    # Part 1: 4x4x4 decomposition with rank 49 (tensor product of two Strassen)
    rank1 = 49
    U1 = [[0]*rank1 for _ in range(n*m)]  # 16 x 49
    V1 = [[0]*rank1 for _ in range(m*p)]  # 24 x 49, but we only use k in {0,1,2,3}
    W1 = [[0]*rank1 for _ in range(p*n)]  # 24 x 49, k in {0,1,2,3}, i in {0,1,2,3}
    
    for i in range(4):
        for j in range(4):
            i1, i2 = i // 2, i % 2
            j1, j2 = j // 2, j % 2
            row_U = i * m + j
            for c1 in range(7):
                for c2 in range(7):
                    col = c1 * 7 + c2
                    U1[row_U][col] = S_U[i1*2+j1][c1] * S_U[i2*2+j2][c2]
    
    for j in range(4):
        for k in range(4):
            j1, j2 = j // 2, j % 2
            k1, k2 = k // 2, k % 2
            row_V = j * p + k
            for c1 in range(7):
                for c2 in range(7):
                    col = c1 * 7 + c2
                    V1[row_V][col] = S_V[j1*2+k1][c1] * S_V[j2*2+k2][c2]
    
    for k in range(4):
        for i in range(4):
            k1, k2 = k // 2, k % 2
            i1, i2 = i // 2, i % 2
            row_W = k * n + i
            for c1 in range(7):
                for c2 in range(7):
                    col = c1 * 7 + c2
                    W1[row_W][col] = S_W[k1*2+i1][c1] * S_W[k2*2+i2][c2]
    
    # Part 2: 4x4x2 decomposition with rank 28 (tiling)
    rank2 = 28
    U2 = [[0]*rank2 for _ in range(n*m)]  # 16 x 28
    V2 = [[0]*rank2 for _ in range(m*p)]  # 24 x 28, k in {4,5}
    W2 = [[0]*rank2 for _ in range(p*n)]  # 24 x 28, k in {4,5}, i in {0,1,2,3}
    
    for bi in range(2):
        for bj in range(2):
            for c in range(7):
                col = (bi * 2 + bj) * 7 + c
                for di in range(2):
                    for dj in range(2):
                        i = bi * 2 + di
                        j = bj * 2 + dj
                        U2[i * m + j][col] = S_U[di * 2 + dj][c]
                
                for dj in range(2):
                    for dk in range(2):
                        j = bj * 2 + dj
                        k = 4 + dk  # k in {4, 5}
                        V2[j * p + k][col] = S_V[dj * 2 + dk][c]
                
                for dk in range(2):
                    for di in range(2):
                        k = 4 + dk
                        i = bi * 2 + di
                        W2[k * n + i][col] = S_W[dk * 2 + di][c]
    
    # Combine
    total_rank = rank1 + rank2
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for i in range(n * m):
        for c in range(rank1):
            U[i][c] = U1[i][c]
        for c in range(rank2):
            U[i][rank1 + c] = U2[i][c]
    
    for i in range(m * p):
        for c in range(rank1):
            V[i][c] = V1[i][c]
        for c in range(rank2):
            V[i][rank1 + c] = V2[i][c]
    
    for i in range(p * n):
        for c in range(rank1):
            W[i][c] = W1[i][c]
        for c in range(rank2):
            W[i][rank1 + c] = W2[i][c]
    
    # Remove zero columns to reduce rank using numpy
    import numpy as np
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # A column is zero if all entries in U, V, W for that column are zero
    # We can check column-wise sums of squares or just any non-zero
    col_sums = np.sum(U_np ** 2, axis=0) + np.sum(V_np ** 2, axis=0) + np.sum(W_np ** 2, axis=0)
    keep_cols = [i for i in range(total_rank) if col_sums[i] > 0]
    
    final_rank = len(keep_cols)
    U_final = U_np[:, keep_cols].tolist()
    V_final = V_np[:, keep_cols].tolist()
    W_final = W_np[:, keep_cols].tolist()
    
    return {"U": U_final, "V": V_final, "W": W_final}
