def construct():
    n, m, p = 3, 3, 7
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Part 1: B cols 0-2 (3x3 block)
    # r indices 0-8
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1

    # Part 2: B cols 3-5 (3x3 block)
    # r indices 9-17
    for i in range(n):
        for j in range(m):
            for k in range(3, 6):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1

    # Part 3: B col 6 (3x1 block)
    # r indices 18-20
    for i in range(n):
        for j in range(m):
            k = 6
            U[i * m + j][r] = 1
            V[j * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1

    return {"U": U, "V": V, "W": W}
