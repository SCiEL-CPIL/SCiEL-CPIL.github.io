import numpy as np
import time
import sys

_START_TIME = time.time()
TIME_LIMIT = 145

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = 4, 4, 7

    total_rank = 93

    s22 = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    s44_outer = [
        ({0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}),
        ({0:1}, {1:1, 3:-1}),
        ({3:1}, {2:1, 0:-1}),
        ({0:1, 1:1}, {3:1}),
        ({2:1, 0:-1}, {0:1, 1:1}),
        ({1:1, 3:-1}, {2:1, 3:1}),
    ]

    c44_blocks = [
        {(0,0):1.0, (1,1):1.0},
        {(1,0):1.0, (1,1):-1.0},
        {(0,1):1.0, (1,1):1.0},
        {(0,0):1.0, (1,0):1.0},
        {(0,0):-1.0, (0,1):1.0},
        {(1,1):1.0},
        {(0,0):1.0},
    ]

    U = np.zeros((n*m, total_rank), dtype=float)
    V = np.zeros((m*p, total_rank), dtype=float)
    W = np.zeros((p*n, total_rank), dtype=float)

    rank = 0

    for outer_idx in range(7):
        x_def, y_def = s44_outer[outer_idx]
        c_block = c44_blocks[outer_idx]

        for inner_idx in range(7):
            r = rank
            rank += 1
            x_sub, y_sub, c_sub = s22[inner_idx]

            for (li, lj), cx in x_sub.items():
                for b, cb in x_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2*bi + li
                    j = 2*bj + lj
                    U[i*m + j, r] += cx * cb

            for (li, lj), cy in y_sub.items():
                for b, cb in y_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2*bi + li
                    j = 2*bj + lj
                    V[i*p + j, r] += cy * cb

            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                for (br, bc), cb_c in c_block.items():
                    if cb_c == 0:
                        continue
                    i_final = 2*br + li
                    k_final = 2*bc + lj
                    W[k_final*n + i_final, r] += cc * cb_c

    def add_block(ar, ac, br, cr):
        nonlocal rank

        for t in range(7):
            r = rank
            rank += 1
            x_sub, y_sub, c_sub = s22[t]

            for (li, lj), cx in x_sub.items():
                i = 2*ar + li
                j = 2*ac + lj
                U[i*m + j, r] += cx

            for (li, lj), cy in y_sub.items():
                i = 2*br + li
                j = 4 + lj
                V[i*p + j, r] += cy

            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                i_final = 2*cr + li
                k_final = 4 + lj
                W[k_final*n + i_final, r] += cc

        r = rank; rank += 1
        U[(2*ar)*m + (2*ac), r] = 1.0
        V[(2*br)*p + 6, r] = 1.0
        W[6*n + 2*cr, r] = 1.0

        r = rank; rank += 1
        U[(2*ar)*m + (2*ac+1), r] = 1.0
        V[(2*br+1)*p + 6, r] = 1.0
        W[6*n + 2*cr, r] = 1.0

        r = rank; rank += 1
        U[(2*ar+1)*m + (2*ac), r] = 1.0
        V[(2*br)*p + 6, r] = 1.0
        W[6*n + 2*cr+1, r] = 1.0

        r = rank; rank += 1
        U[(2*ar+1)*m + (2*ac+1), r] = 1.0
        V[(2*br+1)*p + 6, r] = 1.0
        W[6*n + 2*cr+1, r] = 1.0

    add_block(0, 0, 0, 0)
    add_block(0, 1, 1, 0)
    add_block(1, 0, 0, 1)
    add_block(1, 1, 1, 1)

    assert rank == 93, f"Expected rank 93, got {rank}"

    U = U[:, :rank].tolist()
    V = V[:, :rank].tolist()
    W = W[:, :rank].tolist()

    return {"U": U, "V": V, "W": W}
