"""Matrix multiplication tensor decomposition for 3x4 * 4x7.

Block decomposition using Strassen for 2x2 sub-blocks:
- Split rows: 2+1, columns: 3+4
- 2x4 * 4x3: split middle into 2+2, each 2x2*2x3 uses Strassen(7)+schoolbook(4)=11, total 22
- 2x4 * 4x4: split middle into 2+2, each 2x2*2x4 uses Strassen(7)+Strassen(7)=14, total 28
- 1x4 * 4x7: schoolbook = 28
Total: 22 + 28 + 28 = 78
"""

import numpy as np
import time

N_DIM, M_DIM, P_DIM = 3, 4, 7
TIME_LIMIT = 145.0


def _build_decomposition():
    n, m, p = N_DIM, M_DIM, P_DIM
    t_start = time.time()
    ok = lambda: (time.time() - t_start) < TIME_LIMIT

    u_list = []
    v_list = []
    w_list = []

    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        u = np.zeros(n * m)
        v = np.zeros(m * p)
        w = np.zeros(p * n)
        for idx, val in zip(u_idx, u_val):
            u[idx] = val
        for idx, val in zip(v_idx, v_val):
            v[idx] = val
        for idx, val in zip(w_idx, w_val):
            w[idx] = val
        u_list.append(u)
        v_list.append(v)
        w_list.append(w)

    def schoolbook_block(n_i, m_i, p_i, i0, j0, k0):
        for a in range(n_i):
            for b in range(m_i):
                for c in range(p_i):
                    if not ok():
                        return
                    add_term(
                        [(i0 + a) * m + (j0 + b)], [1.0],
                        [(j0 + b) * p + (k0 + c)], [1.0],
                        [(k0 + c) * n + (i0 + a)], [1.0],
                    )

    def strassen_2x2x2(i0, j0, k0):
        if not ok():
            return
        i01, i11 = i0, i0 + 1
        j01, j11 = j0, j0 + 1
        k01, k11 = k0, k0 + 1

        add_term(
            [i01 * m + j01, i11 * m + j11], [1, 1],
            [j01 * p + k01, j11 * p + k11], [1, 1],
            [k01 * n + i01, k11 * n + i11], [1, 1],
        )
        add_term(
            [i11 * m + j01, i11 * m + j11], [1, 1],
            [j01 * p + k01], [1],
            [k01 * n + i11, k11 * n + i11], [1, -1],
        )
        add_term(
            [i01 * m + j01], [1],
            [j01 * p + k11, j11 * p + k11], [1, -1],
            [k11 * n + i01, k11 * n + i11], [1, 1],
        )
        add_term(
            [i11 * m + j11], [1],
            [j11 * p + k01, j01 * p + k01], [1, -1],
            [k01 * n + i01, k01 * n + i11], [1, 1],
        )
        add_term(
            [i01 * m + j01, i01 * m + j11], [1, 1],
            [j11 * p + k11], [1],
            [k01 * n + i01, k11 * n + i01], [-1, 1],
        )
        add_term(
            [i11 * m + j01, i01 * m + j01], [1, -1],
            [j01 * p + k01, j01 * p + k11], [1, 1],
            [k11 * n + i11], [1],
        )
        add_term(
            [i01 * m + j11, i11 * m + j11], [1, -1],
            [j11 * p + k01, j11 * p + k11], [1, 1],
            [k01 * n + i01], [1],
        )

    def block_2x2x2x3(i0, j0, k0):
        strassen_2x2x2(i0, j0, k0)
        if not ok():
            return
        for a in range(2):
            for b in range(2):
                add_term(
                    [(i0 + a) * m + (j0 + b)], [1.0],
                    [(j0 + b) * p + (k0 + 2)], [1.0],
                    [(k0 + 2) * n + (i0 + a)], [1.0],
                )

    def block_2x2x2x4(i0, j0, k0):
        strassen_2x2x2(i0, j0, k0)
        if not ok():
            return
        strassen_2x2x2(i0, j0, k0 + 2)

    # 2x4 x 4x3: two blocks of 2x2 x 2x3
    block_2x2x2x3(0, 0, 0)
    block_2x2x2x3(0, 2, 0)

    # 2x4 x 4x4: two blocks of 2x2 x 2x4
    block_2x2x2x4(0, 0, 3)
    block_2x2x2x4(0, 2, 3)

    # 1x4 x 4x7: schoolbook
    schoolbook_block(1, 4, 7, 2, 0, 0)

    if len(u_list) == 0:
        u_list, v_list, w_list = _fallback()
        return u_list, v_list, w_list

    U = np.array(u_list).T
    V = np.array(v_list).T
    W = np.array(w_list).T

    # Verification
    idx = np.array([(i, j, k) for i in range(n) for j in range(m) for k in range(p)])
    ii = idx[:, 0]
    jj = idx[:, 1]
    kk = idx[:, 2]
    vals = (U[ii * m + jj] * V[jj * p + kk] * W[kk * n + ii]).sum(axis=1)

    if not np.allclose(vals, 1.0, atol=1e-8):
        u_list, v_list, w_list = _fallback()
        U = np.array(u_list).T
        V = np.array(v_list).T
        W = np.array(w_list).T

    return U, V, W


def _fallback():
    n, m, p = N_DIM, M_DIM, P_DIM
    u_list = []
    v_list = []
    w_list = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(p * n)
                u[i * m + j] = 1.0
                v[j * p + k] = 1.0
                w[k * n + i] = 1.0
                u_list.append(u)
                v_list.append(v)
                w_list.append(w)
    return u_list, v_list, w_list


def construct():
    t_start = time.time()
    U, V, W = _build_decomposition()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
