import time

def construct():
    t0 = time.time()
    tmax = t0 + 105.0

    n = m = p = 5
    U_cols = []
    V_cols = []
    W_cols = []

    # 1. Core 4x4x4: Naive multiplication (rank 16)
    # i, j, k in 0..3
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U_cols.append({i * m + j: 1})
                V_cols.append({j * p + k: 1})
                W_cols.append({k * n + i: 1})

    # 2. A[0..3, 4] * B[4, 0..3] -> C[0..3, 0..3]
    # i in 0..3, k in 0..3
    for i in range(4):
        for k in range(4):
            U_cols.append({i * m + 4: 1})
            V_cols.append({4 * p + k: 1})
            W_cols.append({k * n + i: 1})

    # 3. A[0..3, 0..3] * B[0..3, 4] -> C[0..3, 4]
    # i in 0..3, j in 0..3
    for i in range(4):
        for j in range(4):
            U_cols.append({i * m + j: 1})
            V_cols.append({j * p + 4: 1})
            W_cols.append({4 * n + i: 1})

    # 4. A[0..3, 4] * B[4, 4] -> C[0..3, 4]
    # i in 0..3
    for i in range(4):
        U_cols.append({i * m + 4: 1})
        V_cols.append({4 * p + 4: 1})
        W_cols.append({4 * n + i: 1})

    # 5. A[4, 0..3] * B[0..3, 0..3] -> C[4, 0..3]
    # j in 0..3, k in 0..3
    for j in range(4):
        for k in range(4):
            U_cols.append({4 * m + j: 1})
            V_cols.append({j * p + k: 1})
            W_cols.append({k * n + 4: 1})

    # 6. A[4, 4] * B[4, 0..3] -> C[4, 0..3]
    # k in 0..3
    for k in range(4):
        U_cols.append({4 * m + 4: 1})
        V_cols.append({4 * p + k: 1})
        W_cols.append({k * n + 4: 1})

    # 7. A[4, 0..3] * B[0..3, 4] -> C[4, 4]
    # j in 0..3
    for j in range(4):
        U_cols.append({4 * m + j: 1})
        V_cols.append({j * p + 4: 1})
        W_cols.append({4 * n + 4: 1})

    # 8. A[4, 4] * B[4, 4] -> C[4, 4]
    U_cols.append({4 * m + 4: 1})
    V_cols.append({4 * p + 4: 1})
    W_cols.append({4 * n + 4: 1})

    rank = len(U_cols)
    
    # Build dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        for idx, val in U_cols[r].items():
            U[idx][r] = val
        for idx, val in V_cols[r].items():
            V[idx][r] = val
        for idx, val in W_cols[r].items():
            W[idx][r] = val

    return {"U": U, "V": V, "W": W}
