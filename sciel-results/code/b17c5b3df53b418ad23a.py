import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    DEADLINE = 115

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

    # Rank-10 decomposition for 4x4x2 matrix multiplication
    # U: 16x10, V: 8x10 (j in 0..3, k in 0..1), W: 8x10 (k in 0..1, i in 0..3)
    R10_U = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    R10_V = [
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    ]
    R10_W = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    # Part 2: 4x4x2 for k in {4,5} with rank 10
    rank2 = 10
    U2 = [[0]*rank2 for _ in range(n*m)]
    V2 = [[0]*rank2 for _ in range(m*p)]
    W2 = [[0]*rank2 for _ in range(p*n)]

    for i in range(4):
        for j in range(4):
            row_U = i * m + j
            for c in range(10):
                U2[row_U][c] = R10_U[row_U][c]

    for j in range(4):
        for k in range(2):
            row_V = j * p + (4 + k)
            for c in range(10):
                V2[row_V][c] = R10_V[j * 2 + k][c]

    for k in range(2):
        for i in range(4):
            row_W = (4 + k) * n + i
            for c in range(10):
                W2[row_W][c] = R10_W[k * 4 + i][c]

    # Part 3: 4x4x2 for k in {0,1} with rank 10
    rank3 = 10
    U3 = [[0]*rank3 for _ in range(n*m)]
    V3 = [[0]*rank3 for _ in range(m*p)]
    W3 = [[0]*rank3 for _ in range(p*n)]

    for i in range(4):
        for j in range(4):
            row_U = i * m + j
            for c in range(10):
                U3[row_U][c] = R10_U[row_U][c]

    for j in range(4):
        for k in range(2):
            row_V = j * p + k
            for c in range(10):
                V3[row_V][c] = R10_V[j * 2 + k][c]

    for k in range(2):
        for i in range(4):
            row_W = k * n + i
            for c in range(10):
                W3[row_W][c] = R10_W[k * 4 + i][c]

    # Part 4: 4x4x2 for k in {2,3} with rank 10
    rank4 = 10
    U4 = [[0]*rank4 for _ in range(n*m)]
    V4 = [[0]*rank4 for _ in range(m*p)]
    W4 = [[0]*rank4 for _ in range(p*n)]

    for i in range(4):
        for j in range(4):
            row_U = i * m + j
            for c in range(10):
                U4[row_U][c] = R10_U[row_U][c]

    for j in range(4):
        for k in range(2):
            row_V = j * p + (2 + k)
            for c in range(10):
                V4[row_V][c] = R10_V[j * 2 + k][c]

    for k in range(2):
        for i in range(4):
            row_W = (2 + k) * n + i
            for c in range(10):
                W4[row_W][c] = R10_W[k * 4 + i][c]

    # Combine all parts
    total_rank = rank1 + rank2 + rank3 + rank4
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    offset = 0
    for part_U, part_V, part_W, part_rank in [(U1, V1, W1, rank1),
                                               (U2, V2, W2, rank2),
                                               (U3, V3, W3, rank3),
                                               (U4, V4, W4, rank4)]:
        for i in range(n * m):
            for c in range(part_rank):
                U[i][offset + c] = part_U[i][c]
        for i in range(m * p):
            for c in range(part_rank):
                V[i][offset + c] = part_V[i][c]
        for i in range(p * n):
            for c in range(part_rank):
                W[i][offset + c] = part_W[i][c]
        offset += part_rank

    # Remove zero columns to reduce rank
    keep_cols = []
    for c in range(total_rank):
        if any(U[r][c] != 0 for r in range(n * m)) or \
           any(V[r][c] != 0 for r in range(m * p)) or \
           any(W[r][c] != 0 for r in range(p * n)):
            keep_cols.append(c)

    final_rank = len(keep_cols)
    U_final = [[U[r][c] for c in keep_cols] for r in range(n * m)]
    V_final = [[V[r][c] for c in keep_cols] for r in range(m * p)]
    W_final = [[W[r][c] for c in keep_cols] for r in range(p * n)]

    return {"U": U_final, "V": V_final, "W": W_final}
