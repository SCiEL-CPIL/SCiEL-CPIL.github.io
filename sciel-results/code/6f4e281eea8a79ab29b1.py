import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    R = 49

    # Strassen's 7 multiplications for 2x2
    a_coeffs = np.array([
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]],
    ], dtype=int)
    b_coeffs = np.array([
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]],
    ], dtype=int)
    c_coeffs = np.array([
        [[1, 0], [0, 1]],
        [[0, 0], [1, -1]],
        [[0, 1], [0, 1]],
        [[1, 0], [1, 0]],
        [[-1, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]],
    ], dtype=int)

    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)

    for s in range(7):
        for t in range(7):
            col = 7 * s + t
            # U[i*4+j, col] where i=2*I+i2, j=2*J+j2
            # = a_coeffs[s][I][J] * a_coeffs[t][i2][j2]
            for I in range(2):
                for J in range(2):
                    av = a_coeffs[s][I][J]
                    if av != 0:
                        for i2 in range(2):
                            for j2 in range(2):
                                bv = a_coeffs[t][i2][j2]
                                if bv != 0:
                                    i = 2 * I + i2
                                    j = 2 * J + j2
                                    U[i * 4 + j, col] += av * bv
            for J in range(2):
                for K in range(2):
                    bv = b_coeffs[s][J][K]
                    if bv != 0:
                        for j2 in range(2):
                            for k2 in range(2):
                                cv = b_coeffs[t][j2][k2]
                                if cv != 0:
                                    j = 2 * J + j2
                                    k = 2 * K + k2
                                    V[j * 4 + k, col] += bv * cv
            for K in range(2):
                for I in range(2):
                    cv = c_coeffs[s][I][K]
                    if cv != 0:
                        for k2 in range(2):
                            for i2 in range(2):
                                dv = c_coeffs[t][i2][k2]
                                if dv != 0:
                                    k = 2 * K + k2
                                    i = 2 * I + i2
                                    W[k * 4 + i, col] += cv * dv

            if time.time() > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
