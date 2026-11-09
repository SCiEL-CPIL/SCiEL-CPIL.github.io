import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 16, 20, 20

    # --- Rank 65 Solution: Recursive Strassen 4x4x4 (rank 49) + Schoolbook 4x4x1 (rank 16) ---
    rank_part1 = 49
    rank_part2 = 16
    total_rank = rank_part1 + rank_part2

    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)

    # Part 2: Schoolbook for column 4 (k=4)
    col_offset = rank_part1
    for i in range(n):
        for j in range(m):
            col = col_offset + i * m + j
            U[i * m + j, col] = 1.0
            V[j * p + 4, col] = 1.0
            W[4 * n + i, col] = 1.0

    # Part 1: Recursive Strassen 4x4x4
    def a_idx(bi, u, v):
        i = (bi // 2) * 2 + u
        j = (bi % 2) * 2 + v
        return i * m + j

    def b_idx(bi, u, v):
        j = (bi // 2) * 2 + u
        k = (bi % 2) * 2 + v
        return j * p + k

    def c_idx(bi, u, v):
        i = (bi // 2) * 2 + u
        k = (bi % 2) * 2 + v
        return k * n + i

    A_side = [
        [(0, 1.0), (3, 1.0)],
        [(2, 1.0), (3, 1.0)],
        [(0, 1.0)],
        [(3, 1.0)],
        [(0, 1.0), (1, 1.0)],
        [(2, 1.0), (0, -1.0)],
        [(1, 1.0), (3, -1.0)],
    ]

    B_side = [
        [(0, 1.0), (3, 1.0)],
        [(0, 1.0)],
        [(1, 1.0), (3, -1.0)],
        [(2, 1.0), (0, -1.0)],
        [(3, 1.0)],
        [(0, 1.0), (1, 1.0)],
        [(2, 1.0), (3, 1.0)],
    ]

    out_coeff = [
        {0: 1.0, 3: 1.0},
        {2: 1.0, 3: -1.0},
        {1: 1.0, 3: 1.0},
        {0: 1.0, 2: 1.0},
        {0: -1.0, 1: 1.0},
        {3: 1.0},
        {0: 1.0},
    ]

    for mk in range(7):
        for nk in range(7):
            col = mk * 7 + nk

            for (bi, c1) in A_side[mk]:
                for (bi2, c2) in A_side[nk]:
                    u_local = bi2 // 2
                    v_local = bi2 % 2
                    U[a_idx(bi, u_local, v_local), col] += c1 * c2

            for (bi, c1) in B_side[mk]:
                for (bi2, c2) in B_side[nk]:
                    u_local = bi2 // 2
                    v_local = bi2 % 2
                    V[b_idx(bi, u_local, v_local), col] += c1 * c2

            for (bi_out, c_inner) in out_coeff[nk].items():
                for (bi_outer, c_outer) in out_coeff[mk].items():
                    u_local = bi_out // 2
                    v_local = bi_out % 2
                    W[c_idx(bi_outer, u_local, v_local), col] += c_inner * c_outer

    # Verify the decomposition
    T_target = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0

    T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
    max_err = np.max(np.abs(T_approx - T_target))

    if max_err > 1e-9:
        # Fallback to pure schoolbook (rank 80) if Strassen has a bug
        U2 = np.zeros((nm, 80), dtype=float)
        V2 = np.zeros((mp, 80), dtype=float)
        W2 = np.zeros((pn, 80), dtype=float)
        for idx, (i, j, k) in enumerate([(i, j, k) for i in range(n) for j in range(m) for k in range(p)]):
            U2[i * m + j, idx] = 1.0
            V2[j * p + k, idx] = 1.0
            W2[k * n + i, idx] = 1.0
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
