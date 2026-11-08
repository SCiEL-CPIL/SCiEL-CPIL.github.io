import time
import numpy as np

N, M, P = 2, 4, 4
U_DIM, V_DIM, W_DIM = N * M, M * P, P * N

def construct():
    RANK = 28
    U = np.zeros((U_DIM, RANK), dtype=float)
    V = np.zeros((V_DIM, RANK), dtype=float)
    W = np.zeros((W_DIM, RANK), dtype=float)

    _STRASSEN = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1},          {2: 1, 3: -1}),
        ({0: 1},          {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1},          {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1},          {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    blocks = [
        (0, 0, 0, 0, 0),
        (1, 1, 0, 0, 7),
        (0, 0, 1, 1, 14),
        (1, 1, 1, 1, 21),
    ]

    for a_off, b_r_off, b_c_off, c_c_off, start_col in blocks:
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(_STRASSEN):
            col = start_col + t_idx
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                U[r * M + a_off * 2 + c, col] += coeff
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                V[(b_r_off * 2 + r) * P + b_c_off * 2 + c, col] += coeff
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                W[(c_c_off * 2 + c) * N + r, col] += coeff

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)

    A = np.arange(N * M).reshape(N, M) + 1.0
    B = np.arange(M * P).reshape(M, P) + 2.0
    C = A @ B
    Cf = np.zeros((N, P))

    for ii in range(N):
        for kk in range(P):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(M):
                    acc += U[ii * M + jj, cc] * V[jj * P + kk, cc] * W[kk * N + ii, cc]
            Cf[ii, kk] = acc

    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
