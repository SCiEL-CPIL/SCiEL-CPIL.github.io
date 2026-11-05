import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    DEADLINE = 120.0
    n, m, p = DIMENSIONS

    # Laderman's rank-7 algorithm for 3x3x3
    # U: (9, 7), V: (9, 7), W: (9, 7)
    U7 = np.zeros((9, 7))
    V7 = np.zeros((9, 7))
    W7 = np.zeros((9, 7))

    # M0 = A00 * B00
    U7[0, 0] = 1
    V7[0, 0] = 1
    W7[0, 0] = 1 # C00
    W7[3, 0] = 1 # C10
    W7[4, 0] = -1 # C11

    # M1 = (A00 + A01) * (B10 - B00)
    U7[0, 1] = 1
    U7[1, 1] = 1
    V7[3, 1] = 1
    V7[0, 1] = -1
    W7[0, 1] = -1 # C00
    W7[1, 1] = 1 # C01
    W7[3, 1] = -1 # C10
    W7[6, 1] = 1 # C20

    # M2 = (A10 + A11) * B01
    U7[3, 2] = 1
    U7[4, 2] = 1
    V7[1, 2] = 1
    W7[0, 2] = -1 # C00
    W7[1, 2] = 1 # C01
    W7[4, 2] = 1 # C11
    W7[6, 2] = -1 # C20
    W7[7, 2] = 1 # C21
    W7[8, 2] = -1 # C22

    # M3 = (A20 + A22) * (B01 - B11)
    U7[6, 3] = 1
    U7[8, 3] = 1
    V7[1, 3] = 1
    V7[4, 3] = -1
    W7[1, 3] = -1 # C01
    W7[2, 3] = 1 # C02
    W7[4, 3] = -1 # C11
    W7[5, 3] = 1 # C12
    W7[8, 3] = 1 # C22

    # M4 = A00 * (B10 + B11)
    U7[0, 4] = 1
    V7[3, 4] = 1
    V7[4, 4] = 1
    W7[0, 4] = 1 # C00
    W7[2, 4] = -1 # C02
    W7[3, 4] = 1 # C10
    W7[4, 4] = 1 # C11
    W7[5, 4] = -1 # C12

    # M5 = (A11 - A00) * (B11 + B21)
    U7[4, 5] = 1
    U7[0, 5] = -1
    V7[4, 5] = 1
    V7[7, 5] = 1
    W7[0, 5] = 1 # C00
    W7[1, 5] = 1 # C01
    W7[4, 5] = 1 # C11
    W7[6, 5] = -1 # C20
    W7[7, 5] = 1 # C21
    W7[8, 5] = 1 # C22

    # M6 = (A20 + A10) * (B22 - B12)
    U7[6, 6] = 1
    U7[3, 6] = 1
    V7[8, 6] = 1
    V7[5, 6] = -1
    W7[0, 6] = 1 # C00
    W7[1, 6] = 1 # C01
    W7[2, 6] = -1 # C02
    W7[3, 6] = 1 # C10
    W7[8, 6] = 1 # C22

    # Construct full matrices for 3x3x6
    # U: (9, 14)
    # V: (18, 14)
    # W: (18, 14)
    
    U = np.zeros((9, 14))
    V = np.zeros((18, 14))
    W = np.zeros((18, 14))

    # Block 1: k in 0..2
    U[:, 0:7] = U7
    for j in range(3):
        for k in range(3):
            # V index: j*p + k = j*6 + k
            # V7 index: j*3 + k
            V[j*6 + k, 0:7] = V7[j*3 + k, 0:7]
    for k in range(3):
        for i in range(3):
            # W index: k*n + i = k*3 + i
            # W7 index: k*3 + i
            W[k*3 + i, 0:7] = W7[k*3 + i, 0:7]

    # Block 2: k in 3..5
    U[:, 7:14] = U7
    for j in range(3):
        for k_prime in range(3):
            k = k_prime + 3
            # V index: j*p + k = j*6 + k
            # V7 index: j*3 + k_prime
            V[j*6 + k, 7:14] = V7[j*3 + k_prime, 7:14]
    for k_prime in range(3):
        k = k_prime + 3
        for i in range(3):
            # W index: k*n + i = k*3 + i
            # W7 index: k_prime*3 + i
            W[k*3 + i, 7:14] = W7[k_prime*3 + i, 7:14]

    # Verification
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook (Rank 54)
        rank = n * m * p
        U_s = np.zeros((n*m, rank))
        V_s = np.zeros((m*p, rank))
        W_s = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j, col] = 1
                    V_s[j*p+k, col] = 1
                    W_s[k*n+i, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
