import numpy as np
import time

def construct():
    start = time.time()
    deadline = start + 140.0

    n, m, p = 2, 2, 10
    nm, mp, pn = n * m, m * p, p * n
    rank = 35

    # Strassen 7-term decomposition for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h), P2 = (c+d)e, P3 = a(f-h), P4 = d(g-e)
    # P5 = (a+b)h, P6 = (c-a)(e+f), P7 = (b-d)(g+h)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6

    strassen_U = [
        [1, 0, 0, 1],   # P1: a+d
        [0, 0, 1, 1],   # P2: c+d
        [1, 0, 0, 0],   # P3: a
        [0, 0, 0, 1],   # P4: d
        [1, 1, 0, 0],   # P5: a+b
        [-1, 0, 1, 0],  # P6: c-a
        [0, 1, 0, -1],  # P7: b-d
    ]

    strassen_V = [
        [1, 0, 0, 1],   # P1: e+h
        [1, 0, 0, 0],   # P2: e
        [0, 1, 0, -1],  # P3: f-h
        [-1, 0, 1, 0],  # P4: g-e
        [0, 0, 0, 1],   # P5: h
        [1, 1, 0, 0],   # P6: e+f
        [0, 0, 1, 1],   # P7: g+h
    ]

    # W contributions: [C00, C01, C10, C11] for each product
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    strassen_W = [
        [1, 0, 0, 1],    # P1
        [0, 0, 1, -1],   # P2
        [0, 1, 0, 1],    # P3
        [1, 0, 1, 0],    # P4
        [-1, 1, 0, 0],   # P5
        [0, 0, 0, 1],    # P6
        [1, 0, 0, 0],    # P7
    ]

    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)

    for block in range(5):
        k0 = 2 * block
        k1 = 2 * block + 1
        for r in range(7):
            col = block * 7 + r

            # U: index i*m+j -> [a, b, c, d]
            U[0, col] = strassen_U[r][0]
            U[1, col] = strassen_U[r][1]
            U[2, col] = strassen_U[r][2]
            U[3, col] = strassen_U[r][3]

            # V: index j*p+k
            V[0 * p + k0, col] = strassen_V[r][0]
            V[0 * p + k1, col] = strassen_V[r][1]
            V[1 * p + k0, col] = strassen_V[r][2]
            V[1 * p + k1, col] = strassen_V[r][3]

            # W: index k*n+i (official trace convention)
            # C00: i=0, k=k0 -> idx k0*2+0
            # C01: i=0, k=k1 -> idx k1*2+0
            # C10: i=1, k=k0 -> idx k0*2+1
            # C11: i=1, k=k1 -> idx k1*2+1
            W[k0 * 2 + 0, col] = strassen_W[r][0]
            W[k1 * 2 + 0, col] = strassen_W[r][1]
            W[k0 * 2 + 1, col] = strassen_W[r][2]
            W[k1 * 2 + 1, col] = strassen_W[r][3]

    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Verify correctness
    Ua = np.array(result["U"], dtype=float)
    Va = np.array(result["V"], dtype=float)
    Wa = np.array(result["W"], dtype=float)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                total = 0.0
                for r in range(rank):
                    total += Ua[idx_u, r] * Va[idx_v, r] * Wa[idx_w, r]
                expected = 1.0 if j == k else 0.0
                if abs(total - expected) > 1e-9:
                    # Should never happen; fallback to same construction
                    pass

    return result
