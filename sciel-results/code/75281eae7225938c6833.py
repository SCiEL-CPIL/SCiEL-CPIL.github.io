import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 4, 4, 8
    total_rank = 98

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    strassen_coeffs_u = [
        [1, 1, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, 1],
    ]
    strassen_coeffs_v = [
        [1, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
    ]
    strassen_coeffs_w = [
        [1, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 1, 0],
    ]

    block_a = [
        [(1, 0), (1, 3)],
        [(1, 2), (1, 3)],
        [(1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (-1, 0)],
        [(1, 1), (-1, 3)],
    ]
    block_b = [
        [(1, 0), (1, 3)],
        [(1, 0)],
        [(1, 1), (-1, 3)],
        [(1, 2), (-1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (1, 3)],
    ]
    block_c = [
        [(1, 0), (1, 3)],
        [(1, 2), (-1, 3)],
        [(1, 1), (1, 3)],
        [(1, 0), (1, 2)],
        [(-1, 0), (1, 1)],
        [(1, 3)],
        [(1, 0)],
    ]

    for half in range(2):
        base_col = half * 49
        for kp in range(7):
            for ms in range(7):
                col = base_col + kp * 7 + ms

                if time.time() > deadline:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

                for coeff_a, bidx in block_a[kp]:
                    br, bc = bidx // 2, bidx % 2
                    for rr in range(2):
                        for cc in range(2):
                            u_pos = (br * 2 + rr) * 4 + (bc * 2 + cc)
                            s_pos = rr * 2 + cc
                            U[u_pos, col] += coeff_a * strassen_coeffs_u[s_pos][ms]

                for coeff_b, bidx in block_b[kp]:
                    br, bc = bidx // 2, bidx % 2
                    for rr in range(2):
                        for cc in range(2):
                            v_pos = (br * 2 + rr) * 8 + (bc * 2 + cc) + half * 4
                            s_pos = rr * 2 + cc
                            V[v_pos, col] += coeff_b * strassen_coeffs_v[s_pos][ms]

                for coeff_c, cidx in block_c[kp]:
                    cr, cc = cidx // 2, cidx % 2
                    for rr in range(2):
                        for cc2 in range(2):
                            w_pos = (cc * 2 + cc2) * 4 + (cr * 2 + rr)
                            s_pos = rr * 2 + cc2
                            W[w_pos, col] += coeff_c * strassen_coeffs_w[s_pos][ms]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
