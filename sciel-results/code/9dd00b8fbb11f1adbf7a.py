import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 130

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    if _time_remaining() < 5:
        return {"U": [], "V": [], "W": []}

    n, m, p = 4, 4, 7
    # Rank 4x4x4 is 49 (Strassen)
    # Rank 4x4x3 is 4 * 11 = 44 (4 blocks of 2x2x3)
    total_rank = 49 + 44

    U = np.zeros((n * m, total_rank), dtype=np.int64)
    V = np.zeros((m * p, total_rank), dtype=np.int64)
    W = np.zeros((p * n, total_rank), dtype=np.int64)

    current_rank = 0

    # Strassen 2x2x2 terms: (x_sub, y_sub, c_sub)
    # x, y, c are dictionaries mapping (row, col) to integer coefficient
    strassen_2x2 = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1})
    ]

    # Strassen 4x4x4: 7 outer terms
    # x_def, y_def map block indices (0,1,2,3) to integer coefficients
    # Block 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    strassen_4x4_outer = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}),
        ({0: 1}, {1: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}),
        ({0: 1, 1: 1}, {3: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]

    c_blocks_4x4 = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1},
    ]

    # Part 1: 4x4x4 Strassen (rank 49)
    # Handles B columns 0-3
    for m_idx in range(7):
        x_def, y_def = strassen_4x4_outer[m_idx]
        c_block = c_blocks_4x4[m_idx]
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]

            # Compute U coefficients
            for (li, lj), cx in x_sub.items():
                for b, cb in x_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    U[i * m + j, r] += cx * cb

            # Compute V coefficients
            for (li, lj), cy in y_sub.items():
                for b, cb in y_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    V[i * p + j, r] += cy * cb

            # Compute W coefficients
            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                for (br, bc), cb_c in c_block.items():
                    if cb_c == 0:
                        continue
                    i_final = 2 * br + li
                    k_final = 2 * bc + lj
                    W[k_final * n + i_final, r] += cc * cb_c

    # Part 2: 4x4x3 (rank 44)
    # Handles B columns 4-6
    # Split into 4 blocks of 2x2x3
    def add_2x2x3(ar, ac, br, cr):
        nonlocal current_rank

        # Strassen 2x2x2 for local cols 0,1 (global cols 4,5)
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]

            for (li, lj), cx in x_sub.items():
                i = 2 * ar + li
                j = 2 * ac + lj
                U[i * m + j, r] += cx

            for (li, lj), cy in y_sub.items():
                i = 2 * br + li
                j = 4 + lj
                V[i * p + j, r] += cy

            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                i_final = 2 * cr + li
                k_final = 4 + lj
                W[k_final * n + i_final, r] += cc

        # 4 schoolbook terms for local col 2 (global col 6)
        r = current_rank
        current_rank += 1
        U[(2 * ar + 0) * m + (2 * ac + 0), r] = 1
        V[(2 * br + 0) * p + 6, r] = 1
        W[6 * n + (2 * cr + 0), r] = 1

        r = current_rank
        current_rank += 1
        U[(2 * ar + 0) * m + (2 * ac + 1), r] = 1
        V[(2 * br + 1) * p + 6, r] = 1
        W[6 * n + (2 * cr + 0), r] = 1

        r = current_rank
        current_rank += 1
        U[(2 * ar + 1) * m + (2 * ac + 0), r] = 1
        V[(2 * br + 0) * p + 6, r] = 1
        W[6 * n + (2 * cr + 1), r] = 1

        r = current_rank
        current_rank += 1
        U[(2 * ar + 1) * m + (2 * ac + 1), r] = 1
        V[(2 * br + 1) * p + 6, r] = 1
        W[6 * n + (2 * cr + 1), r] = 1

    # The four 2x2x3 blocks
    add_2x2x3(0, 0, 0, 0)
    add_2x2x3(0, 1, 1, 0)
    add_2x2x3(1, 0, 0, 1)
    add_2x2x3(1, 1, 1, 1)

    U = U[:, :current_rank]
    V = V[:, :current_rank]
    W = W[:, :current_rank]

    # Convert to list of lists with integer values
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
