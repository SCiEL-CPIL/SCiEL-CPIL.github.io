import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    # Rank-7 Strassen decomposition for 2x2 matrix multiplication
    # U2: (4, 7), V2: (4, 7), W2: (4, 7)
    # Row indexing: row = i*2 + j for U (input A), row = j*2 + k for V (input B), row = k*2 + i for W (output C)
    U2 = np.zeros((4, 7), dtype=np.int64)
    V2 = np.zeros((4, 7), dtype=np.int64)
    W2 = np.zeros((4, 7), dtype=np.int64)

    # t=0: M0 = (a+d)(e+h) -> U: a,d ; V: e,h ; W: c00,c11
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # t=1: M1 = (c+d)e -> U: c,d ; V: e ; W: c10, c11
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1

    # t=2: M2 = a(f-h) -> U: a ; V: f,h ; W: c01, c11
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1

    # t=3: M3 = d(g-e) -> U: d ; V: g,e ; W: c00, c01
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1

    # t=4: M4 = (a+b)h -> U: a,b ; V: h ; W: c01, c00
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[1, 4] = 1; W2[0, 4] = -1

    # t=5: M5 = (c-a)(e+f) -> U: c,a ; V: e,f ; W: c11
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    # t=6: M6 = (b-d)(g+h) -> U: b,d ; V: g,h ; W: c00
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Rank-27 schoolbook decomposition for 3x3 matrix multiplication
    # U3: (9, 27), V3: (9, 27), W3: (9, 27)
    U3 = np.zeros((9, 27), dtype=np.int64)
    V3 = np.zeros((9, 27), dtype=np.int64)
    W3 = np.zeros((9, 27), dtype=np.int64)

    for i3 in range(3):
        for j3 in range(3):
            for k3 in range(3):
                t3 = i3 * 9 + j3 * 3 + k3
                U3[i3 * 3 + j3, t3] = 1
                V3[j3 * 3 + k3, t3] = 1
                W3[k3 * 3 + i3, t3] = 1

    # Kronecker product: rank = 7 * 27 = 189
    # 6x6 = 2x2 blocks of 3x3 sub-blocks
    # A is 6x6, split into 2x2 blocks each 3x3
    # B is 6x6, split into 2x2 blocks each 3x3
    # C is 6x6, split into 2x2 blocks each 3x3

    # For each Strassen term t2 and each schoolbook term t3:
    #   U[i*6+j, t2*27+t3] = U2[i2*2+j2, t2] * U3[i3*3+j3, t3]
    #   V[j*6+k, t2*27+t3] = V2[j2*2+k2, t2] * V3[j3*3+k3, t3]
    #   W[k*6+i, t2*27+t3] = W2[k2*2+i2, t2] * W3[k3*3+i3, t3]
    # where i = i3*2 + i2, j = j3*2 + j2, k = k3*2 + k2

    rank = 7 * 27
    U = np.zeros((36, rank), dtype=np.int64)
    V = np.zeros((36, rank), dtype=np.int64)
    W = np.zeros((36, rank), dtype=np.int64)

    for t2 in range(7):
        for t3 in range(27):
            c = t2 * 27 + t3
            for i2 in range(2):
                for j2 in range(2):
                    for i3 in range(3):
                        for j3 in range(3):
                            i = i3 * 2 + i2
                            j = j3 * 2 + j2
                            u_val = U2[i2 * 2 + j2, t2] * U3[i3 * 3 + j3, t3]
                            if u_val != 0:
                                U[i * 6 + j, c] = u_val
            for j2 in range(2):
                for k2 in range(2):
                    for j3 in range(3):
                        for k3 in range(3):
                            j = j3 * 2 + j2
                            k = k3 * 2 + k2
                            v_val = V2[j2 * 2 + k2, t2] * V3[j3 * 3 + k3, t3]
                            if v_val != 0:
                                V[j * 6 + k, c] = v_val
            for k2 in range(2):
                for i2 in range(2):
                    for k3 in range(3):
                        for i3 in range(3):
                            k = k3 * 2 + k2
                            i = i3 * 2 + i2
                            w_val = W2[k2 * 2 + i2, t2] * W3[k3 * 3 + i3, t3]
                            if w_val != 0:
                                W[k * 6 + i, c] = w_val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
