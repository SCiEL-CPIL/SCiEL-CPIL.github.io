"""Structured block-decomposition for 2x5x6 matrix multiplication tensor using Strassen blocks."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill

    # Target rank from structured Strassen + schoolbook decomposition
    rank = 54

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    def add_rank_step(u_pairs, v_pairs, w_pairs):
        nonlocal col
        if col >= rank:
            return False
        if time.time() - start_time > time_limit:
            return False
        for idx, val in u_pairs:
            U[idx][col] = val
        for idx, val in v_pairs:
            V[idx][col] = val
        for idx, val in w_pairs:
            W[idx][col] = val
        col += 1
        return True

    def add_strassen(j1, j2, k1, k2):
        """
        Adds Strassen rank-7 terms for A[:, j1:j2+1] * B[j1:j2+1, k1:k2+1]
        assuming i in {0,1}.
        """
        # m1 = (a00+a11)(b00+b11)
        add_rank_step(
            [(u_idx(0, j1), 1), (u_idx(1, j2), 1)],
            [(v_idx(j1, k1), 1), (v_idx(j2, k2), 1)],
            [(w_idx(k1, 0), 1), (w_idx(k2, 1), 1)]
        )

        # m2 = (a10+a11)*b00
        add_rank_step(
            [(u_idx(1, j1), 1), (u_idx(1, j2), 1)],
            [(v_idx(j1, k1), 1)],
            [(w_idx(k1, 1), 1), (w_idx(k2, 1), -1)]
        )

        # m3 = a00*(b01-b11)
        add_rank_step(
            [(u_idx(0, j1), 1)],
            [(v_idx(j1, k2), 1), (v_idx(j2, k2), -1)],
            [(w_idx(k2, 0), 1), (w_idx(k2, 1), 1)]
        )

        # m4 = a11*(b10-b00)
        add_rank_step(
            [(u_idx(1, j2), 1)],
            [(v_idx(j2, k1), 1), (v_idx(j1, k1), -1)],
            [(w_idx(k1, 0), 1), (w_idx(k1, 1), 1)]
        )

        # m5 = (a00+a01)*b11
        add_rank_step(
            [(u_idx(0, j1), 1), (u_idx(0, j2), 1)],
            [(v_idx(j2, k2), 1)],
            [(w_idx(k1, 0), -1), (w_idx(k2, 0), 1)]
        )

        # m6 = (a10-a00)*(b00+b01)
        add_rank_step(
            [(u_idx(1, j1), 1), (u_idx(0, j1), -1)],
            [(v_idx(j1, k1), 1), (v_idx(j1, k2), 1)],
            [(w_idx(k2, 1), 1)]
        )

        # m7 = (a01-a11)*(b10+b11)
        add_rank_step(
            [(u_idx(0, j2), 1), (u_idx(1, j2), -1)],
            [(v_idx(j2, k1), 1), (v_idx(j2, k2), 1)],
            [(w_idx(k1, 0), 1)]
        )

    def add_schoolbook_2x1x2(j, k1, k2):
        """
        Adds schoolbook rank-4 terms for A[:, j] * B[j, k1:k2+1]
        """
        add_rank_step([(u_idx(0, j), 1)], [(v_idx(j, k1), 1)], [(w_idx(k1, 0), 1)])
        add_rank_step([(u_idx(0, j), 1)], [(v_idx(j, k2), 1)], [(w_idx(k2, 0), 1)])
        add_rank_step([(u_idx(1, j), 1)], [(v_idx(j, k1), 1)], [(w_idx(k1, 1), 1)])
        add_rank_step([(u_idx(1, j), 1)], [(v_idx(j, k2), 1)], [(w_idx(k2, 1), 1)])

    # Block decomposition:
    # Part 1: 2x2x2 (j=0,1; k=0,1) -> Strassen
    add_strassen(0, 1, 0, 1)

    # Part 2: 2x2x2 (j=0,1; k=2,3) -> Strassen
    add_strassen(0, 1, 2, 3)

    # Part 3: 2x2x2 (j=0,1; k=4,5) -> Strassen
    add_strassen(0, 1, 4, 5)

    # Part 4: 2x2x2 (j=2,3; k=0,1) -> Strassen
    add_strassen(2, 3, 0, 1)

    # Part 5: 2x1x2 (j=4; k=0,1) -> Schoolbook
    add_schoolbook_2x1x2(4, 0, 1)

    # Part 6: 2x2x2 (j=2,3; k=2,3) -> Strassen
    add_strassen(2, 3, 2, 3)

    # Part 7: 2x1x2 (j=4; k=2,3) -> Schoolbook
    add_schoolbook_2x1x2(4, 2, 3)

    # Part 8: 2x2x2 (j=2,3; k=4,5) -> Strassen
    add_strassen(2, 3, 4, 5)

    # Part 9: 2x1x2 (j=4; k=4,5) -> Schoolbook
    add_schoolbook_2x1x2(4, 4, 5)

    # Verify the decomposition
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)

    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.sum(U_mat[i * m + j, :] * V_mat[j * p + k, :] * W_mat[k * n + i, :])
                if not np.isclose(total, 1.0, atol=1e-9):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback to schoolbook rank 60
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        col_sb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col_sb] = 1
                    V_sb[j * p + k, col_sb] = 1
                    W_sb[k * n + i, col_sb] = 1
                    col_sb += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {"U": U, "V": V, "W": W}
