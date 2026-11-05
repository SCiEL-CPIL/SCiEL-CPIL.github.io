import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # 10s safety margin
    n, m, p = DIMENSIONS

    # Strassen 2x2 factors (rank 7)
    # U_S, V_S, W_S are 4x7
    U_S = np.zeros((4, 7), dtype=int)
    V_S = np.zeros((4, 7), dtype=int)
    W_S = np.zeros((4, 7), dtype=int)

    # M1: (A00+A11)(B00+B11)
    U_S[0, 0] = 1; U_S[3, 0] = 1
    V_S[0, 0] = 1; V_S[3, 0] = 1
    # M2: (A10+A11)B00
    U_S[2, 1] = 1; U_S[3, 1] = 1
    V_S[0, 1] = 1
    # M3: A00(B10-B11)
    U_S[0, 2] = 1
    V_S[2, 2] = 1; V_S[3, 2] = -1
    # M4: A11(B01-B10)
    U_S[3, 3] = 1
    V_S[1, 3] = 1; V_S[2, 3] = -1
    # M5: (A00+A01)B11
    U_S[0, 4] = 1; U_S[1, 4] = 1
    V_S[3, 4] = 1
    # M6: (A10-A00)(B00+B10)
    U_S[2, 5] = 1; U_S[0, 5] = -1
    V_S[0, 5] = 1; V_S[2, 5] = 1
    # M7: (A01-A11)(B01+B11)
    U_S[1, 6] = 1; U_S[3, 6] = -1
    V_S[1, 6] = 1; V_S[3, 6] = 1

    # W_S
    # C00 = M1 + M4 - M5 - M7
    W_S[0, 0] = 1; W_S[0, 3] = 1; W_S[0, 4] = -1; W_S[0, 6] = -1
    # C01 = M3 + M5
    W_S[1, 2] = 1; W_S[1, 4] = 1
    # C10 = M2 + M4
    W_S[2, 1] = 1; W_S[2, 3] = 1
    # C11 = M1 - M2 + M3 + M7
    W_S[3, 0] = 1; W_S[3, 1] = -1; W_S[3, 2] = 1; W_S[3, 6] = 1

    # Schoolbook 3x3 factors (rank 27)
    U_3 = np.zeros((9, 27), dtype=int)
    V_3 = np.zeros((9, 27), dtype=int)
    W_3 = np.zeros((9, 27), dtype=int)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                c = i*9 + j*3 + k
                U_3[i*3+j, c] = 1
                V_3[j*3+k, c] = 1
                W_3[i*3+k, c] = 1

    # Kronecker product
    rank = 27 * 7
    U_6 = np.zeros((36, rank), dtype=int)
    V_6 = np.zeros((36, rank), dtype=int)
    W_6 = np.zeros((36, rank), dtype=int)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                outer = i*9 + j*3 + k
                for a in range(2):
                    for b in range(2):
                        I_u = 3*i + 2*a
                        J_u = 3*j + 2*b
                        u_row = I_u*6 + J_u
                        for c_inner in range(7):
                            term = outer * 7 + c_inner
                            val = U_3[i*3+j, outer] * U_S[a*2+b, c_inner]
                            if val != 0:
                                U_6[u_row, term] += val

    for j in range(3):
        for k in range(3):
            for i in range(3):
                outer = i*9 + j*3 + k
                for b in range(2):
                    for c in range(2):
                        J_v = 3*j + 2*b
                        K_v = 3*k + 2*c
                        v_row = J_v*6 + K_v
                        for c_inner in range(7):
                            term = outer * 7 + c_inner
                            val = V_3[j*3+k, outer] * V_S[b*2+c, c_inner]
                            if val != 0:
                                V_6[v_row, term] += val

    for k in range(3):
        for i in range(3):
            for j in range(3):
                outer = i*9 + j*3 + k
                for c in range(2):
                    for a in range(2):
                        K_w = 3*k + 2*c
                        I_w = 3*i + 2*a
                        w_row = K_w*6 + I_w
                        for c_inner in range(7):
                            term = outer * 7 + c_inner
                            val = W_3[i*3+k, outer] * W_S[c*2+a, c_inner]
                            if val != 0:
                                W_6[w_row, term] += val

    if time.time() > deadline:
        pass

    return {"U": U_6, "V": V_6, "W": W_6}
# EVOLVE-BLOCK-END
