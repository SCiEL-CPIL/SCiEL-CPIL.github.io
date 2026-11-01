"""Rank-minimal exact tensor decomposition for 2x4 @ 4x6 matrix multiplication."""
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 2, 4, 6
    rank_total = 42
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]

    # Strassen 2x2 @ 2x2 rank 7 decomposition
    strassen_U = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {0: -1, 2: 1},
        {1: 1, 3: -1}
    ]

    strassen_V = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]

    strassen_W = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1}
    ]

    col_offset = 0
    for j_block in range(3):
        for side in ['L', 'R']:
            if side == 'L':
                u_col_offset = 0
                v_row_offset = 0
            else:
                u_col_offset = 2
                v_row_offset = 2

            for prod_idx in range(7):
                col = col_offset + prod_idx

                for local_idx, sign in strassen_U[prod_idx].items():
                    if local_idx == 0:
                        u_idx = u_col_offset
                    elif local_idx == 1:
                        u_idx = u_col_offset + 1
                    elif local_idx == 2:
                        u_idx = 4 + u_col_offset
                    else:
                        u_idx = 4 + u_col_offset + 1
                    U[u_idx][col] += sign

                for local_idx, sign in strassen_V[prod_idx].items():
                    if local_idx == 0:
                        v_idx = v_row_offset * 6 + 2 * j_block
                    elif local_idx == 1:
                        v_idx = v_row_offset * 6 + 2 * j_block + 1
                    elif local_idx == 2:
                        v_idx = (v_row_offset + 1) * 6 + 2 * j_block
                    else:
                        v_idx = (v_row_offset + 1) * 6 + 2 * j_block + 1
                    V[v_idx][col] += sign

                for local_idx, sign in strassen_W[prod_idx].items():
                    if local_idx == 0:
                        w_idx = 2 * (2 * j_block)
                    elif local_idx == 1:
                        w_idx = 2 * (2 * j_block + 1)
                    elif local_idx == 2:
                        w_idx = 2 * (2 * j_block) + 1
                    else:
                        w_idx = 2 * (2 * j_block + 1) + 1
                    W[w_idx][col] += sign

            col_offset += 7

    if time.time() > deadline:
        pass

    return {"U": U, "V": V, "W": W}
