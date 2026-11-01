import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    num_blocks = p // 2
    R = num_blocks * 7

    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)

    U_coeffs = {
        0: {0: 1, 3: 1},
        1: {2: 1, 3: 1},
        2: {0: 1},
        3: {3: 1},
        4: {0: 1, 1: 1},
        5: {2: 1, 0: -1},
        6: {1: 1, 3: -1}
    }

    V_coeffs = {
        0: {0: 1, 3: 1},
        1: {0: 1},
        2: {1: 1, 3: -1},
        3: {2: 1, 0: -1},
        4: {3: 1},
        5: {0: 1, 1: 1},
        6: {2: 1, 3: 1}
    }

    W_contribs = {
        0: {0: 1, 3: 1},
        1: {2: 1, 3: -1},
        2: {1: 1, 3: 1},
        3: {0: 1, 2: 1},
        4: {0: -1, 1: 1},
        5: {3: 1},
        6: {0: 1}
    }

    for q in range(num_blocks):
        b11 = 2 * q
        b12 = 2 * q + 1
        b21 = 16 + 2 * q
        b22 = 16 + 2 * q + 1

        c11 = 4 * q
        c12 = 4 * q + 2
        c21 = 4 * q + 1
        c22 = 4 * q + 3

        for s in range(7):
            r = q * 7 + s
            for a_idx, coeff in U_coeffs[s].items():
                U[a_idx, r] = coeff
            for local_idx, coeff in V_coeffs[s].items():
                if local_idx == 0:
                    V[b11, r] = coeff
                elif local_idx == 1:
                    V[b12, r] = coeff
                elif local_idx == 2:
                    V[b21, r] = coeff
                else:
                    V[b22, r] = coeff
            for local_out_idx, coeff in W_contribs[s].items():
                if local_out_idx == 0:
                    W[c11, r] = coeff
                elif local_out_idx == 1:
                    W[c12, r] = coeff
                elif local_out_idx == 2:
                    W[c21, r] = coeff
                else:
                    W[c22, r] = coeff

    # Verification
    if time.time() < deadline - 10:
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, size=(n, m))
        B_test = np.random.randint(-5, 5, size=(m, p))
        C_test = A_test @ B_test

        A_flat = A_test.flatten()
        B_flat = B_test.flatten()

        UA = U.T @ A_flat
        VB = V.T @ B_flat
        terms = UA * VB
        C_decomp = W @ terms

        C_test_flat = C_test.T.flatten()

        if not np.allclose(C_decomp, C_test_flat):
            rank = n * m * p
            U = np.zeros((n * m, rank), dtype=int)
            V = np.zeros((m * p, rank), dtype=int)
            W = np.zeros((p * n, rank), dtype=int)

            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        U[u_idx, column] = 1
                        V[v_idx, column] = 1
                        W[w_idx, column] = 1
                        column += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
