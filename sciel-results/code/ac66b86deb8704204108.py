import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = 2, 2, 12

    num_pairs = p // 2  # 6
    rank = num_pairs * 7  # 42

    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    # Strassen 7-product decomposition for 2x2 * 2x2
    # A = [[a,b],[c,d]] -> U rows: a=0, b=1, c=2, d=3
    # B = [[e,f],[g,h]] -> local B: e=0, f=1, g=2, h=3
    # C = [[C00,C01],[C10,C11]] -> local C: C00=0, C01=1, C10=2, C11=3
    #
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Each term: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict A_row -> coeff
    # V_coeffs: dict local_B -> coeff
    # W_coeffs: dict local_C -> coeff
    strassen_terms = [
        # s=0: M1 = (a+d)(e+h)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # s=1: M2 = (c+d)e
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # s=2: M3 = a(f-h)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # s=3: M4 = d(g-e)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # s=4: M5 = (a+b)h
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # s=5: M6 = (c-a)(e+f)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # s=6: M7 = (b-d)(g+h)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    for t in range(num_pairs):
        base = t * 7
        k0 = 2 * t
        k1 = 2 * t + 1

        for s in range(7):
            c = base + s
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[s]

            # Set U (A indices are global, same for all blocks)
            for a_idx, val in u_coeffs.items():
                U[a_idx, c] = val

            # Set V (map local B index to global V row)
            for local_b, val in v_coeffs.items():
                if local_b == 0:
                    v_row = 0 * p + k0    # B[0, 2t]
                elif local_b == 1:
                    v_row = 0 * p + k1    # B[0, 2t+1]
                elif local_b == 2:
                    v_row = 1 * p + k0    # B[1, 2t]
                else:
                    v_row = 1 * p + k1    # B[1, 2t+1]
                V[v_row, c] = val

            # Set W (map local C index to global W row: W[k*n+i])
            for local_c, val in w_coeffs.items():
                if local_c == 0:
                    w_row = k0 * n + 0    # C[0, 2t]
                elif local_c == 1:
                    w_row = k1 * n + 0    # C[0, 2t+1]
                elif local_c == 2:
                    w_row = k0 * n + 1    # C[1, 2t]
                else:
                    w_row = k1 * n + 1    # C[1, 2t+1]
                W[w_row, c] = val

    # Verification with random test
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)

    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test

    max_err = 0.0
    for i in range(n):
        for k in range(p):
            val = np.sum(U_arr[i * m + 0, :] * V_arr[0 * p + k, :] * W_arr[k * n + i, :]
                        + U_arr[i * m + 1, :] * V_arr[1 * p + k, :] * W_arr[k * n + i, :])
            max_err = max(max_err, abs(val - C_expected[i, k]))

    if max_err > 1e-9:
        # Fallback to schoolbook rank-48 if verification fails
        rank = 48
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        r = 0
        for k in range(p):
            for i in range(n):
                for j in range(m):
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
                    r += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
