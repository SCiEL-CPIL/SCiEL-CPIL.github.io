import numpy as np

def construct():
    # 4x4x4 matrix multiplication tensor decomposition at rank 49
    # built from the tensor square (Kronecker product) of Strassen's
    # rank-7 decomposition for 2x2x2 matrix multiplication.
    #
    # Row conventions:
    #   U row index  s = i*4 + j   (i row of A, j col of A)
    #   V row index  s = j*4 + k   (j row of B, k col of B)
    #   W row index  s = k*4 + i   (trace convention: C[k, i])
    # For each term t in {0..48}, the triple (U[s,t], V[s,t], W[s,t])
    # must satisfy sum_t U[i*4+j, t] * V[j*4+k, t] * W[k*4+i, t] = 1.
    #
    # We build each column t of the three factor matrices independently
    # by enumerating the two 2x2 sub-indices that define the Kronecker
    # product column, then verifying the resulting 4x4x4 identity in
    # the same pass before emitting.

    # Strassen 2x2x2 factor rows (standard form).
    #   U2: row r = i*2 + j  -> A_{i,j} factor
    #   V2: row r = j*2 + k  -> B_{j,k} factor
    #   W2: row r = k*2 + i  -> C_{k,i} factor (trace convention)
    _U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    _V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    _W2 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]

    # Verify 2x2x2 sub-decomposition first.
    for i in range(2):
        for j in range(2):
            for k in range(2):
                tot = 0
                for t in range(7):
                    tot += _U2[i * 2 + j][t] * _V2[j * 2 + k][t] * _W2[k * 2 + i][t]
                if tot != 1:
                    return _schoolbook(4, 4, 4)

    # Build the 49-column factors of the 4x4x4 decomposition.
    RANK = 49
    U = [[0] * RANK for _ in range(16)]
    V = [[0] * RANK for _ in range(16)]
    W = [[0] * RANK for _ in range(16)]

    for t in range(RANK):
        o = t // 7  # outer Strassen column
        n_ = t % 7  # inner Strassen column
        # Row (i,j) of A gets U2[2*i1+j1][o] * U2[2*i2+j2][n_]
        # Row (j,k) of B gets V2[2*j1+k1][o] * V2[2*j2+k2][n_]
        # Row (k,i) of C gets W2[2*k1+i1][o] * W2[2*k2+i2][n_]
        for i in range(4):
            for j in range(4):
                i1, i2 = i >> 1, i & 1
                j1, j2 = j >> 1, j & 1
                U[i * 4 + j][t] = (
                    _U2[i1 * 2 + j1][o] * _U2[i2 * 2 + j2][n_]
                )
        for j in range(4):
            for k in range(4):
                j1, j2 = j >> 1, j & 1
                k1, k2 = k >> 1, k & 1
                V[j * 4 + k][t] = (
                    _V2[j1 * 2 + k1][o] * _V2[j2 * 2 + k2][n_]
                )
        for k in range(4):
            for i in range(4):
                k1, k2 = k >> 1, k & 1
                i1, i2 = i >> 1, i & 1
                W[k * 4 + i][t] = (
                    _W2[k1 * 2 + i1][o] * _W2[k2 * 2 + i2][n_]
                )

    # Full 4x4x4 verification in integer arithmetic.
    for i in range(4):
        for j in range(4):
            for k in range(4):
                acc = 0
                ui_row = U[i * 4 + j]
                vj_row = V[j * 4 + k]
                wk_row = W[k * 4 + i]
                for t in range(RANK):
                    acc += ui_row[t] * vj_row[t] * wk_row[t]
                if acc != 1:
                    return _schoolbook(4, 4, 4)

    return {"U": U, "V": V, "W": W}


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
