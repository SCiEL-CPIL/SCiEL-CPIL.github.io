import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Use rank 49 (Strassen tensor product)
    rank = 49

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Strassen coefficients for 2x2 matrix multiplication
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    #
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 - M3 + M5
    # C11 = M0 - M1 + M2 + M5

    A_coeff = np.zeros((7, 2, 2), dtype=int)
    B_coeff = np.zeros((7, 2, 2), dtype=int)
    C_coeff = np.zeros((7, 2, 2), dtype=int)

    # s=0: M0 = (A00+A11)(B00+B11)
    A_coeff[0, 0, 0] = 1; A_coeff[0, 1, 1] = 1
    B_coeff[0, 0, 0] = 1; B_coeff[0, 1, 1] = 1
    C_coeff[0, 0, 0] = 1; C_coeff[0, 1, 1] = 1

    # s=1: M1 = (A10+A11)B00
    A_coeff[1, 1, 0] = 1; A_coeff[1, 1, 1] = 1
    B_coeff[1, 0, 0] = 1
    C_coeff[1, 1, 0] = 1; C_coeff[1, 1, 1] = -1

    # s=2: M2 = A00(B01-B11)
    A_coeff[2, 0, 0] = 1
    B_coeff[2, 0, 1] = 1; B_coeff[2, 1, 1] = -1
    C_coeff[2, 0, 1] = 1; C_coeff[2, 1, 1] = 1

    # s=3: M3 = A11(B10-B00)
    A_coeff[3, 1, 1] = 1
    B_coeff[3, 1, 0] = 1; B_coeff[3, 0, 0] = -1
    C_coeff[3, 0, 0] = 1; C_coeff[3, 1, 0] = -1

    # s=4: M4 = (A00+A01)B11
    A_coeff[4, 0, 0] = 1; A_coeff[4, 0, 1] = 1
    B_coeff[4, 1, 1] = 1
    C_coeff[4, 0, 0] = -1; C_coeff[4, 0, 1] = 1

    # s=5: M5 = (A10-A00)(B00+B01)
    A_coeff[5, 1, 0] = 1; A_coeff[5, 0, 0] = -1
    B_coeff[5, 0, 0] = 1; B_coeff[5, 0, 1] = 1
    C_coeff[5, 1, 0] = 1; C_coeff[5, 1, 1] = 1

    # s=6: M6 = (A01-A11)(B10+B11)
    A_coeff[6, 0, 1] = 1; A_coeff[6, 1, 1] = -1
    B_coeff[6, 1, 0] = 1; B_coeff[6, 1, 1] = 1
    C_coeff[6, 0, 0] = 1

    # Build U, V, W for 4x4 matrix multiplication via tensor product of two Strassen layers
    for s in range(7):
        for t in range(7):
            r = 7 * s + t

            # U: row = i*4 + j, i = 2*I + i2, j = 2*J + j2
            for I in range(2):
                for J in range(2):
                    ca_out = A_coeff[s, I, J]
                    if ca_out == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            ca_in = A_coeff[t, i2, j2]
                            if ca_in == 0:
                                continue
                            val = ca_out * ca_in
                            i = 2 * I + i2
                            j = 2 * J + j2
                            U[i * 4 + j, r] = val

            # V: row = j*4 + k, j = 2*J + j2, k = 2*K + k2
            for J in range(2):
                for K in range(2):
                    cb_out = B_coeff[s, J, K]
                    if cb_out == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            cb_in = B_coeff[t, j2, k2]
                            if cb_in == 0:
                                continue
                            val = cb_out * cb_in
                            j = 2 * J + j2
                            k = 2 * K + k2
                            V[j * 4 + k, r] = val

            # W: row = k*4 + i (trace convention W[k*n+i, r]), i = 2*I + i2, k = 2*K + k2
            for I in range(2):
                for K in range(2):
                    cc_out = C_coeff[s, I, K]
                    if cc_out == 0:
                        continue
                    for i2 in range(2):
                        for k2 in range(2):
                            cc_in = C_coeff[t, i2, k2]
                            if cc_in == 0:
                                continue
                            val = cc_out * cc_in
                            i = 2 * I + i2
                            k = 2 * K + k2
                            W[k * 4 + i, r] = val

            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
