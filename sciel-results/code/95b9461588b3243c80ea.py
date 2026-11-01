import time

DIMENSIONS = [2, 2, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS

    # Rank 32 decomposition: 4 Strassen blocks (rank 7 each) + 1 trivial block (rank 4)
    total_rank = 32

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0

    def emit_strassen_block(ka, kb):
        nonlocal col
        # M1: (a11+a22)(b11+b22)
        U[0][col] = 1
        U[3][col] = 1
        V[ka][col] = 1
        V[p + kb][col] = 1
        W[ka][col] = 1
        W[kb + n][col] = 1
        col += 1
        # M2: (a21+a22)b11
        U[2][col] = 1
        U[3][col] = 1
        V[ka][col] = 1
        W[ka + n][col] = 1
        W[kb + n][col] = -1
        col += 1
        # M3: a11(b21-b22)
        U[0][col] = 1
        V[p + kb][col] = 1
        V[p + kb][col] = -1
        W[kb][col] = 1
        W[kb + n][col] = 1
        col += 1
        # M4: a22(b11-b12)
        U[3][col] = 1
        V[p + ka][col] = 1
        V[ka][col] = -1
        W[ka][col] = 1
        W[ka + n][col] = 1
        col += 1
        # M5: (a11+a12)b22
        U[0][col] = 1
        U[1][col] = 1
        V[p + kb][col] = 1
        W[ka][col] = -1
        W[kb][col] = 1
        col += 1
        # M6: (a21-a11)(b11+b12)
        U[2][col] = 1
        U[0][col] = -1
        V[ka][col] = 1
        V[kb][col] = 1
        W[kb + n][col] = 1
        col += 1
        # M7: (a12-a22)(b21+b22)
        U[1][col] = 1
        U[3][col] = -1
        V[p + ka][col] = 1
        V[p + kb][col] = 1
        W[ka][col] = 1
        col += 1

    def emit_trivial_block(k):
        nonlocal col
        U[0][col] = 1
        V[k][col] = 1
        W[k][col] = 1
        col += 1
        U[1][col] = 1
        V[p + k][col] = 1
        W[k][col] = 1
        col += 1
        U[2][col] = 1
        V[k][col] = 1
        W[k + n][col] = 1
        col += 1
        U[3][col] = 1
        V[p + k][col] = 1
        W[k + n][col] = 1
        col += 1

    for j in range(4):
        emit_strassen_block(2 * j, 2 * j + 1)
    emit_trivial_block(8)

    return {"U": U, "V": V, "W": W}
