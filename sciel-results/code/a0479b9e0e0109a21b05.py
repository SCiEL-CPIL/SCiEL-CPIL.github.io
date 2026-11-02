import time
import numpy as np

_start = time.time()
_DEADLINE = _start + 130.0

def construct():
    n, m, p = 2, 2, 15
    total_rank = 53

    U = np.zeros((n * m, total_rank), dtype=np.int64)
    V = np.zeros((m * p, total_rank), dtype=np.int64)
    W = np.zeros((p * n, total_rank), dtype=np.int64)

    # Strassen 2x2x2 block: multiplies [[a,b],[c,d]] * [[e,f],[g,h]]
    # Produces 7 products, reconstructs 4 output entries
    def strassen_2x2x2(base, k0, k1):
        # r indices within this block
        r = [base + i for i in range(7)]

        # P1 = (a+d)(e+h)
        U[0, r[0]] += 1
        U[3, r[0]] += 1
        V[0 * p + k0, r[0]] += 1
        V[1 * p + k1, r[0]] += 1

        # P2 = (c+d)e
        U[2, r[1]] += 1
        U[3, r[1]] += 1
        V[0 * p + k0, r[1]] += 1

        # P3 = a(f-h)
        U[0, r[2]] += 1
        V[0 * p + k1, r[2]] += 1
        V[1 * p + k1, r[2]] += -1

        # P4 = d(g-e)
        U[3, r[3]] += 1
        V[1 * p + k0, r[3]] += 1
        V[0 * p + k0, r[3]] += -1

        # P5 = (a+b)h
        U[0, r[4]] += 1
        U[1, r[4]] += 1
        V[1 * p + k1, r[4]] += 1

        # P6 = (c-a)(e+f)
        U[2, r[5]] += 1
        U[0, r[5]] += -1
        V[0 * p + k0, r[5]] += 1
        V[0 * p + k1, r[5]] += 1

        # P7 = (b-d)(g+h)
        U[1, r[6]] += 1
        U[3, r[6]] += -1
        V[1 * p + k0, r[6]] += 1
        V[1 * p + k1, r[6]] += 1

        # Reconstruct outputs
        # C[i,k] = W[k*n + i, r]
        # C0,k0 = P1 + P4 - P5 + P7
        W[2 * k0 + 0, r[0]] += 1
        W[2 * k0 + 0, r[3]] += 1
        W[2 * k0 + 0, r[4]] += -1
        W[2 * k0 + 0, r[6]] += 1

        # C1,k0 = P2 + P4
        W[2 * k0 + 1, r[1]] += 1
        W[2 * k0 + 1, r[3]] += 1

        # C0,k1 = P3 + P5
        W[2 * k1 + 0, r[2]] += 1
        W[2 * k1 + 0, r[4]] += 1

        # C1,k1 = P1 - P2 + P3 + P6
        W[2 * k1 + 1, r[0]] += 1
        W[2 * k1 + 1, r[1]] += -1
        W[2 * k1 + 1, r[2]] += 1
        W[2 * k1 + 1, r[5]] += 1

    # Place 7 Strassen blocks for column pairs (0,1), (2,3), ..., (12,13)
    for blk in range(7):
        col0 = 2 * blk
        col1 = 2 * blk + 1
        strassen_2x2x2(7 * blk, col0, col1)

    # Trivial 2x2x1 block for column 14 (rank 4)
    b0 = 49
    # a * B[0,14] -> C[0,14]
    U[0, b0] = 1
    V[0 * p + 14, b0] = 1
    W[2 * 14 + 0, b0] = 1

    # b * B[1,14] -> C[0,14]
    U[1, b0 + 1] = 1
    V[1 * p + 14, b0 + 1] = 1
    W[2 * 14 + 0, b0 + 1] = 1

    # c * B[0,14] -> C[1,14]
    U[2, b0 + 2] = 1
    V[0 * p + 14, b0 + 2] = 1
    W[2 * 14 + 1, b0 + 2] = 1

    # d * B[1,14] -> C[1,14]
    U[3, b0 + 3] = 1
    V[1 * p + 14, b0 + 3] = 1
    W[2 * 14 + 1, b0 + 3] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
