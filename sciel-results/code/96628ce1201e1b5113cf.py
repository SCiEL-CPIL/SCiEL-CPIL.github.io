"""Tensor decomposition of 5x5x5 matrix multiplication, rank 58."""
import time
import random

def construct():
    t0 = time.time()
    tmax = t0 + 105.0

    n = m = p = 5
    U_cols = []
    V_cols = []
    W_cols = []

    # 4x4x4 core via Strassen rank-7
    def a_block(blk_i, blk_j):
        return [(2 * blk_i + di, 2 * blk_j + dj) for di in range(2) for dj in range(2)]

    def b_block(blk_j, blk_k):
        return [(2 * blk_j + dj, 2 * blk_k + dk) for dj in range(2) for dk in range(2)]

    products = [
        [((0, 0), (0, 0), 1), ((0, 0), (0, 1), 1),
         ((1, 1), (0, 0), 1), ((1, 1), (0, 1), 1)],
        [((1, 0), (0, 0), 1), ((1, 1), (0, 0), 1)],
        [((0, 0), (0, 1), 1), ((0, 0), (1, 1), -1)],
        [((1, 1), (1, 0), 1), ((1, 1), (0, 0), -1)],
        [((0, 0), (1, 1), 1), ((0, 1), (1, 1), 1)],
        [((1, 0), (0, 0), 1), ((1, 0), (0, 1), 1),
         ((0, 0), (0, 0), -1), ((0, 0), (0, 1), -1)],
        [((0, 1), (1, 0), 1), ((0, 1), (1, 1), 1),
         ((1, 1), (1, 0), -1), ((1, 1), (1, 1), -1)],
    ]

    for prod in products:
        u_acc = {}
        v_acc = {}
        w_acc = {}
        for (ab_idx, bb_idx, coeff) in prod:
            ai, aj = ab_idx
            bj, bk = bb_idx
            for (ii, jj) in a_block(ai, aj):
                for (jj2, kk) in b_block(bj, bk):
                    if jj != jj2:
                        continue
                    ui = ii * m + jj
                    vi = jj * p + kk
                    wi = kk * n + ii
                    u_acc[ui] = u_acc.get(ui, 0) + coeff
                    v_acc[vi] = v_acc.get(vi, 0) + coeff
                    w_acc[wi] = w_acc.get(wi, 0) + coeff
        u_acc = {k: v for k, v in u_acc.items() if v != 0}
        v_acc = {k: v for k, v in v_acc.items() if v != 0}
        w_acc = {k: v for k, v in w_acc.items() if v != 0}
        U_cols.append(u_acc)
        V_cols.append(v_acc)
        W_cols.append(w_acc)

    # Cross terms: handle all contributions involving index 4
    # Group A: i in 0..3, k in 0..3, j=4: A[i,4]*B[4,k] -> C[i,k]
    # One term: U[i*5+4]=1 for i=0..3, V[4*5+k]=1 for k=0..3, W[k*5+i]=1 for i,k=0..3
    u = {i * m + 4: 1 for i in range(4)}
    v = {4 * p + k: 1 for k in range(4)}
    w = {k * n + i: 1 for i in range(4) for k in range(4)}
    U_cols.append(u); V_cols.append(v); W_cols.append(w)

    # Group B: i in 0..3, k=4, j in 0..3: A[i,j]*B[j,4] -> C[i,4]
    # One term: U[i*5+j]=1 for i=0..3,j=0..3, V[j*5+4]=1 for j=0..3, W[4*5+i]=1 for i=0..3
    u = {i * m + j: 1 for i in range(4) for j in range(4)}
    v = {j * p + 4: 1 for j in range(4)}
    w = {4 * n + i: 1 for i in range(4)}
    U_cols.append(u); V_cols.append(v); W_cols.append(w)

    # Group C: i in 0..3, k=4, j=4: A[i,4]*B[4,4] -> C[i,4]
    u = {i * m + 4: 1 for i in range(4)}
    v = {4 * p + 4: 1}
    w = {4 * n + i: 1 for i in range(4)}
    U_cols.append(u); V_cols.append(v); W_cols.append(w)

    # Group D: i=4, k in 0..3, j in 0..3: A[4,j]*B[j,k] -> C[4,k]
    u = {4 * m + j: 1 for j in range(4)}
    v = {j * p + k: 1 for j in range(4) for k in range(4)}
    w = {k * n + 4: 1 for k in range(4)}
    U_cols.append(u); V_cols.append(v); W_cols.append(w)

    # Group E: i=4, k in 0..3, j=4: A[4,4]*B[4,k] -> C[4,k]
    u = {4 * m + 4: 1}
    v = {4 * p + k: 1 for k in range(4)}
    w = {k * n + 4: 1 for k in range(4)}
    U_cols.append(u); V_cols.append(v); W_cols.append(w)

    # Group F: i=4, k=4, j in 0..3: A[4,j]*B[j,4] -> C[4,4]
    u = {4 * m + j: 1 for j in range(4)}
    v = {j * p + 4: 1 for j in range(4)}
    w = {4 * n + 4: 1}
    U_cols.append(u); V_cols.append(v); W_cols.append(w)

    # Group G: i=4, k=4, j=4: A[4,4]*B[4,4] -> C[4,4]
    u = {4 * m + 4: 1}
    v = {4 * p + 4: 1}
    w = {4 * n + 4: 1}
    U_cols.append(u); V_cols.append(v); W_cols.append(w)

    rank = len(U_cols)  # 7 + 7 = 14? That seems too low.
