import time

DIMENSIONS = [3, 4, 6]

def _strassen_columns():
    """Return (U, V, W) lists of 7 column vectors for 2x2x2 Strassen.
    U[j*2+i] = a_{ij}, V[k*2+j] = b_{jk}, W[i*2+k] = c_{ik}.
    """
    U = []
    V = []
    W = []
    U.append([1, 0, 0, -1]);  V.append([0, 0, 1, 1]);  W.append([0, 0, 1, 1])
    U.append([0, 0, 1, 1]);  V.append([1, -1, 0, 0]);  W.append([0, 0, 1, -1])
    U.append([1, 1, 0, 0]);  V.append([0, 0, 0, 1]);  W.append([1, 0, 0, 0])
    U.append([0, 0, 0, 1]);  V.append([0, 1, -1, 0]);  W.append([0, 1, 0, 0])
    U.append([1, 0, 0, 0]);  V.append([0, 1, 0, 1]);  W.append([1, 0, 1, 0])
    U.append([0, -1, 1, 0]);  V.append([1, 0, 1, 0]);  W.append([1, 0, 0, 0])
    U.append([0, 1, 0, -1]);  V.append([0, 0, 1, 1]);  W.append([0, 1, 0, 1])
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = 3, 4, 6
    U_cols = []
    V_cols = []
    W_cols = []
    sU, sV, sW = _strassen_columns()

    def add_column(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]

    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            for r in range(7):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                for i in [0, 1]:
                    for j in [j0, j1]:
                        i_local = i
                        j_local = 0 if j == j0 else 1
                        u[i * m + j] = sU[r][j_local * 2 + i_local]
                for j in [j0, j1]:
                    for k in [k0, k1]:
                        j_local = 0 if j == j0 else 1
                        k_local = 0 if k == k0 else 1
                        v[j * p + k] = sV[r][k_local * 2 + j_local]
                for k in [k0, k1]:
                    for i in [0, 1]:
                        i_local = i
                        k_local = 0 if k == k0 else 1
                        w[k * n + i] = sW[r][i_local * 2 + k_local]
                add_column(u, v, w)

    u = [0] * (n * m)
    v = [0] * (m * p)
    w = [0] * (p * n)
    for j in range(m):
        u[2 * m + j] = 1
    for k in range(p):
        w[k * n + 2] = 1
    for j in range(m):
        for k in range(p):
            v[j * p + k] = 1
    add_column(u, v, w)

    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c in range(rank):
        for idx in range(n * m):
            U[idx][c] = U_cols[c][idx]
        for idx in range(m * p):
            V[idx][c] = V_cols[c][idx]
        for idx in range(p * n):
            W[idx][c] = W_cols[c][idx]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = 0
                for c in range(rank):
                    actual += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if actual != expected:
                    raise ValueError(f"Verification failed at i={i}, j={j}, k={k}")

    return {"U": U, "V": V, "W": W}
