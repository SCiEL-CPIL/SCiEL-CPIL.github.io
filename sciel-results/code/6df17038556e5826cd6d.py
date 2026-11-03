"""Matrix multiplication 4,4,6 rank-minimal decomposition via Kronecker product."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    n, m, p = DIMENSIONS  # 4, 4, 6

    # Strassen decomposition for 2x2x2 (rank 7)
    # A (2x2), B (2x2), C (2x2)
    # U1: rows i*2+j, V1: rows j*2+k, W1: rows k*2+i
    U1 = np.zeros((4, 7), dtype=float)
    V1 = np.zeros((4, 7), dtype=float)
    W1 = np.zeros((4, 7), dtype=float)

    # M1 = (a11+a22)(b11+b22)
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1; W1[3, 0] = 1

    # M2 = (a21+a22)b11
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[1, 1] = 1; W1[3, 1] = -1

    # M3 = a11(b12-b22)
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    W1[2, 2] = 1; W1[3, 2] = 1

    # M4 = a22(b21-b11)
    U1[3, 3] = 1
    V1[2, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = 1; W1[1, 3] = 1

    # M5 = (a11+a12)b22
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = -1; W1[2, 4] = 1

    # M6 = (a21-a11)(b11+b12)
    U1[2, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1

    # M7 = (a12-a22)(b21+b22)
    U1[1, 6] = 1; U1[3, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = 1

    # 2x2x3 decomposition (rank 11)
    # A (2x2), B (2x3), C (2x3)
    # U2: rows i*2+j, V2: rows j*3+k, W2: rows k*2+i
    U2 = np.zeros((4, 11), dtype=float)
    V2 = np.zeros((6, 11), dtype=float)
    W2 = np.zeros((6, 11), dtype=float)

    # Strassen part for first 2 columns (terms 0-6)
    for c in range(7):
        U2[0, c] = U1[0, c]
        U2[1, c] = U1[1, c]
        U2[2, c] = U1[2, c]
        U2[3, c] = U1[3, c]

        V2[0, c] = V1[0, c]  # j=0, k=0
        V2[1, c] = V1[1, c]  # j=0, k=1
        V2[3, c] = V1[2, c]  # j=1, k=0
        V2[4, c] = V1[3, c]  # j=1, k=1
        V2[2, c] = 0         # j=0, k=2
        V2[5, c] = 0         # j=1, k=2

        W2[0, c] = W1[0, c]  # k=0, i=0
        W2[1, c] = W1[1, c]  # k=0, i=1
        W2[2, c] = W1[2, c]  # k=1, i=0
        W2[3, c] = W1[3, c]  # k=1, i=1
        W2[4, c] = 0         # k=2, i=0
        W2[5, c] = 0         # k=2, i=1

    # Trivial part for column 2 (terms 7-10)
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]

    # Term 7: A[0,0]*B[0,2] -> C[0,2]
    U2[0, 7] = 1
    V2[2, 7] = 1  # j=0, k=2
    W2[4, 7] = 1  # k=2, i=0

    # Term 8: A[0,1]*B[1,2] -> C[0,2]
    U2[1, 8] = 1
    V2[5, 8] = 1  # j=1, k=2
    W2[4, 8] = 1  # k=2, i=0

    # Term 9: A[1,0]*B[0,2] -> C[1,2]
    U2[2, 9] = 1
    V2[2, 9] = 1  # j=0, k=2
    W2[5, 9] = 1  # k=2, i=1

    # Term 10: A[1,1]*B[1,2] -> C[1,2]
    U2[3, 10] = 1
    V2[5, 10] = 1  # j=1, k=2
    W2[5, 10] = 1  # k=2, i=1

    # Kronecker product construction
    # A (4x4) = A1 (2x2) ⊗ A2 (2x2)
    # B (4x6) = B1 (2x2) ⊗ B2 (2x3)
    # C (4x6) = C1 (2x2) ⊗ C2 (2x3)
    #
    # Total rank = 7 * 11 = 77
    rank = 77
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((24, rank), dtype=float)
    W = np.zeros((24, rank), dtype=float)

    # Index mappings:
    # U row: i*4+j where i=i1*2+i2, j=j1*2+j2 -> i1*8 + i2*4 + j1*2 + j2
    # V row: j*6+k where j=j1*2+j2, k=k1*3+k2 -> j1*12 + j2*6 + k1*3 + k2
    # W row: k*4+i where k=k1*3+k2, i=i1*2+i2 -> k1*12 + k2*4 + i1*2 + i2

    for c1 in range(7):
        for c2 in range(11):
            c = c1 * 11 + c2

            # U construction
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u1 = U1[i1*2 + j1, c1]
                            u2 = U2[i2*2 + j2, c2]
                            if u1 != 0 and u2 != 0:
                                row = i1*8 + i2*4 + j1*2 + j2
                                U[row, c] = u1 * u2

            # V construction
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            v1 = V1[j1*2 + k1, c1]
                            v2 = V2[j2*3 + k2, c2]
                            if v1 != 0 and v2 != 0:
                                row = j1*12 + j2*6 + k1*3 + k2
                                V[row, c] = v1 * v2

            # W construction
            for k1 in range(2):
                for k2 in range(3):
                    for i1 in range(2):
                        for i2 in range(2):
                            w1 = W1[k1*2 + i1, c1]
                            w2 = W2[k2*2 + i2, c2]
                            if w1 != 0 and w2 != 0:
                                row = k1*12 + k2*4 + i1*2 + i2
                                W[row, c] = w1 * w2

    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Should not happen for this fast computation, but guard just in case
        pass

    return {"U": U, "V": V, "W": W}
