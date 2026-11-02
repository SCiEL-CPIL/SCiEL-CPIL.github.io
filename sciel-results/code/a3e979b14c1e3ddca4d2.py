import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100.0

    n, m, p = 2, 2, 16
    blocks = p // 2
    total_rank = blocks * 7

    U = np.zeros((n * m, total_rank), dtype=np.int8)
    V = np.zeros((m * p, total_rank), dtype=np.int8)
    W = np.zeros((p * n, total_rank), dtype=np.int8)

    strassen = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]

    for q in range(blocks):
        b_global = [0 * p + 2 * q, 0 * p + 2 * q + 1, 1 * p + 2 * q, 1 * p + 2 * q + 1]
        c_global = [(2 * q) * n + 0, (2 * q + 1) * n + 0, (2 * q) * n + 1, (2 * q + 1) * n + 1]

        for s in range(7):
            r = q * 7 + s
            u_dict, v_dict, w_dict = strassen[s]

            for u_row, val in u_dict.items():
                U[u_row, r] = val

            for v_local, val in v_dict.items():
                V[b_global[v_local], r] = val

            for w_local, val in w_dict.items():
                W[c_global[w_local], r] = val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
