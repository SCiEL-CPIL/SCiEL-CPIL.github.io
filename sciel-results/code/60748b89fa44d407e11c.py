import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS
    R = 98
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)

    # Strassen 2x2x2 rank 7 decomposition
    S_U = np.zeros((4, 7))
    S_V = np.zeros((4, 7))
    S_W = np.zeros((4, 7))

    # M1 = (A00+A11)(B00+B11)
    S_U[0, 0] = 1; S_U[3, 0] = 1
    S_V[0, 0] = 1; S_V[3, 0] = 1
    S_W[0, 0] = 1; S_W[3, 0] = 1

    # M2 = (A10+A11)B00
    S_U[2, 1] = 1; S_U[3, 1] = 1
    S_V[0, 1] = 1
    S_W[2, 1] = 1; S_W[3, 1] = 1

    # M3 = A00(B01-B11)
    S_U[0, 2] = 1
    S_V[1, 2] = 1; S_V[3, 2] = -1
    S_W[1, 2] = 1; S_W[3, 2] = 1

    # M4 = A11(B10-B00)
    S_U[3, 3] = 1
    S_V[2, 3] = 1; S_V[0, 3] = -1
    S_W[0, 3] = 1; S_W[2, 3] = 1

    # M5 = (A00+A01)B11
    S_U[0, 4] = 1; S_U[1, 4] = 1
    S_V[3, 4] = 1
    S_W[0, 4] = -1; S_W[1, 4] = 1

    # M6 = (A10-A00)(B00+B01)
    S_U[2, 5] = 1; S_U[0, 5] = -1
    S_V[0, 5] = 1; S_V[1, 5] = 1
    S_W[3, 5] = 1

    # M7 = (A01-A11)(B10+B11)
    S_U[1, 6] = 1; S_U[3, 6] = -1
    S_V[2, 6] = 1; S_V[3, 6] = 1
    S_W[0, 6] = 1; S_W[3, 6] = 1

    # Strassen 4x4x4: 7 outer terms, each with 7 inner terms = 49
    A_coefs = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]]
    ]

    B_coefs = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],
        [[0, -1], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 1], [0, 1]]
    ]

    C_coefs = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[0, 1], [0, 1]],
        [[1, 0], [0, 1]],
        [[-1, 1], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]]
    ]

    for t in range(7):
        for s in range(7):
            col = t * 7 + s

            for I in range(2):
                for J in range(2):
                    a_c = A_coefs[t][I][J]
                    if a_c == 0:
                        continue
                    for i_l in range(2):
                        for j_l in range(2):
                            val = a_c * S_U[i_l * 2 + j_l, s]
                            if val != 0:
                                U[I * 2 + i_l, J * 2 + j_l, col] = val

            for K in range(2):
                for J in range(2):
                    b_c = B_coefs[t][K][J]
                    if b_c == 0:
                        continue
                    for j_l in range(2):
                        for k_l in range(2):
                            val = b_c * S_V[j_l * 2 + k_l, s]
                            if val != 0:
                                V[K * 2 + j_l, J * 2 + k_l, col] = val

            for I in range(2):
                for J in range(2):
                    c_c = C_coefs[t][I][J]
                    if c_c == 0:
                        continue
                    for i_l in range(2):
                        for k_l in range(2):
                            val = c_c * S_W[i_l * 2 + k_l, s]
                            if val != 0:
                                W[I * 2 + i_l, J * 2 + k_l, col] = val

    # Second half: right 4x4 block of B
    for t in range(7):
        for s in range(7):
            col = 49 + t * 7 + s

            for I in range(2):
                for J in range(2):
                    a_c = A_coefs[t][I][J]
                    if a_c == 0:
                        continue
                    for i_l in range(2):
                        for j_l in range(2):
                            val = a_c * S_U[i_l * 2 + j_l, s]
                            if val != 0:
                                U[I * 2 + i_l, J * 2 + j_l, col] = val

            for K in range(2):
                for J in range(2):
                    b_c = B_coefs[t][K][J]
                    if b_c == 0:
                        continue
                    for j_l in range(2):
                        for k_l in range(2):
                            val = b_c * S_V[j_l * 2 + k_l, s]
                            if val != 0:
                                V[K * 2 + j_l, 4 + J * 2 + k_l, col] = val

            for I in range(2):
                for J in range(2):
                    c_c = C_coefs[t][I][J]
                    if c_c == 0:
                        continue
                    for i_l in range(2):
                        for k_l in range(2):
                            val = c_c * S_W[i_l * 2 + k_l, s]
                            if val != 0:
                                W[4 + I * 2 + i_l, J * 2 + k_l, col] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
