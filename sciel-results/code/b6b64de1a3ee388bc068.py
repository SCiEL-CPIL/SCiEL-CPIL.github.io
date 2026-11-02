import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Strassen factors for 2x2x2
    # u22[i*2+j, r], v22[j*2+k, r], w22[k*2+i, r]
    u22 = np.zeros((4, 7), dtype=float)
    v22 = np.zeros((4, 7), dtype=float)
    w22 = np.zeros((4, 7), dtype=float)

    # r=0: M1 = (A00 + A11)(B00 + B11)
    u22[0, 0] = 1; u22[3, 0] = 1
    v22[0, 0] = 1; v22[3, 0] = 1
    w22[0, 0] = 1; w22[3, 0] = 1

    # r=1: M2 = (A10 + A11)B00
    u22[2, 1] = 1; u22[3, 1] = 1
    v22[0, 1] = 1
    w22[2, 1] = 1; w22[3, 1] = -1

    # r=2: M3 = A00(B10 - B11)
    u22[0, 2] = 1
    v22[2, 2] = 1; v22[3, 2] = -1
    w22[0, 2] = 1; w22[3, 2] = 1

    # r=3: M4 = A11(B01 - B00)
    u22[3, 3] = 1
    v22[1, 3] = 1; v22[0, 3] = -1
    w22[0, 3] = 1; w22[2, 3] = 1

    # r=4: M5 = (A00 + A01)B11
    u22[0, 4] = 1; u22[1, 4] = 1
    v22[3, 4] = 1
    w22[0, 4] = -1; w22[1, 4] = 1

    # r=5: M6 = (A10 - A00)(B00 + B01)
    u22[2, 5] = 1; u22[0, 5] = -1
    v22[0, 5] = 1; v22[1, 5] = 1
    w22[3, 5] = 1

    # r=6: M7 = (A01 - A11)(B10 + B11)
    u22[1, 6] = 1; u22[3, 6] = -1
    v22[2, 6] = 1; v22[3, 6] = 1
    w22[0, 6] = 1; w22[3, 6] = 1

    col = 0
    # Part 0: j in {0,1}
    # Sub-part 0a: k in {0,1}
    for r in range(7):
        for i in range(2):
            for j in range(2):
                if u22[i*2+j, r] != 0:
                    U[i*4 + j, col] = u22[i*2+j, r]
        for j in range(2):
            for k in range(2):
                if v22[j*2+k, r] != 0:
                    V[j*4 + k, col] = v22[j*2+k, r]
        for k in range(2):
            for i in range(2):
                if w22[k*2+i, r] != 0:
                    W[k*2 + i, col] = w22[k*2+i, r]
        col += 1

    # Sub-part 0b: k in {2,3}
    for r in range(7):
        for i in range(2):
            for j in range(2):
                if u22[i*2+j, r] != 0:
                    U[i*4 + j, col] = u22[i*2+j, r]
        for j in range(2):
            for k in range(2):
                if v22[j*2+k, r] != 0:
                    V[j*4 + (k+2), col] = v22[j*2+k, r]
        for k in range(2):
            for i in range(2):
                if w22[k*2+i, r] != 0:
                    W[(k+2)*2 + i, col] = w22[k*2+i, r]
        col += 1

    # Part 1: j in {2,3}
    # Sub-part 1a: k in {0,1}
    for r in range(7):
        for i in range(2):
            for j in range(2):
                if u22[i*2+j, r] != 0:
                    U[i*4 + (j+2), col] = u22[i*2+j, r]
        for j in range(2):
            for k in range(2):
                if v22[j*2+k, r] != 0:
                    V[(j+2)*4 + k, col] = v22[j*2+k, r]
        for k in range(2):
            for i in range(2):
                if w22[k*2+i, r] != 0:
                    W[k*2 + i, col] = w22[k*2+i, r]
        col += 1

    # Sub-part 1b: k in {2,3}
    for r in range(7):
        for i in range(2):
            for j in range(2):
                if u22[i*2+j, r] != 0:
                    U[i*4 + (j+2), col] = u22[i*2+j, r]
        for j in range(2):
            for k in range(2):
                if v22[j*2+k, r] != 0:
                    V[(j+2)*4 + (k+2), col] = v22[j*2+k, r]
        for k in range(2):
            for i in range(2):
                if w22[k*2+i, r] != 0:
                    W[(k+2)*2 + i, col] = w22[k*2+i, r]
        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
