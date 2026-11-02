import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 8, 32, 16

    # We use a rank-56 decomposition based on 8 independent 2x2 Strassen blocks.
    # This is a safe upper bound.
    rank0 = 56
    U0 = np.zeros((nm, rank0), dtype=float)
    V0 = np.zeros((mp, rank0), dtype=float)
    W0 = np.zeros((pn, rank0), dtype=float)

    for a in range(2):
        for b in range(4):
            base = 7 * (4 * a + b)

            a00 = 0 * m + 2 * a
            a01 = 0 * m + 2 * a + 1
            a10 = 1 * m + 2 * a
            a11 = 1 * m + 2 * a + 1

            b00 = (2 * a) * p + 2 * b
            b01 = (2 * a) * p + 2 * b + 1
            b10 = (2 * a + 1) * p + 2 * b
            b11 = (2 * a + 1) * p + 2 * b + 1

            c00 = (2 * b) * n + 0
            c01 = (2 * b + 1) * n + 0
            c10 = (2 * b) * n + 1
            c11 = (2 * b + 1) * n + 1

            U0[a00, base] += 1; U0[a11, base] += 1
            V0[b00, base] += 1; V0[b11, base] += 1
            W0[c00, base] += 1; W0[c11, base] += 1

            U0[a10, base+1] += 1; U0[a11, base+1] += 1
            V0[b00, base+1] += 1
            W0[c10, base+1] += 1; W0[c11, base+1] -= 1

            U0[a00, base+2] += 1
            V0[b01, base+2] += 1; V0[b11, base+2] -= 1
            W0[c01, base+2] += 1; W0[c11, base+2] += 1

            U0[a11, base+3] += 1
            V0[b10, base+3] += 1; V0[b00, base+3] -= 1
            W0[c00, base+3] += 1; W0[c10, base+3] += 1

            U0[a00, base+4] += 1; U0[a01, base+4] += 1
            V0[b11, base+4] += 1
            W0[c00, base+4] -= 1; W0[c01, base+4] += 1

            U0[a10, base+5] += 1; U0[a00, base+5] -= 1
            V0[b00, base+5] += 1; V0[b01, base+5] += 1
            W0[c11, base+5] += 1

            U0[a01, base+6] += 1; U0[a11, base+6] -= 1
            V0[b10, base+6] += 1; V0[b11, base+6] += 1
            W0[c00, base+6] += 1

    def verify(U, V, W, tol=1e-7):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    if not verify(U0, V0, W0):
        rank = 64
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
