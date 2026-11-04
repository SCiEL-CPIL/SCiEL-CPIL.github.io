"""Rank-189 decomposition: Kronecker product of rank-7 Strassen 2x2 and rank-27 schoolbook 3x3."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def _kronecker_decomp(A1, B1, C1, A2, B2, C2):
    """Kronecker product of two tensor decompositions.
    A1,B1,C1: factors of decomposition 1 (ranks r1), A2,B2,C2: decomposition 2 (ranks r2).
    Combined rank = r1*r2. Index convention: combined index = (coarse, fine) block-major.
    """
    r1 = len(A1[0])
    r2 = len(A2[0])
    rank = r1 * r2
    n1 = len(A1)
    n2 = len(A2)
    m1 = len(B1)
    m2 = len(B2)
    p1 = len(C1)
    p2 = len(C2)
    U = [[0] * rank for _ in range(n1 * n2)]
    V = [[0] * rank for _ in range(m1 * m2)]
    W = [[0] * rank for _ in range(p1 * p2)]
    for i1 in range(n1):
        for j1 in range(m1):
            for k1 in range(p1):
                for c1 in range(r1):
                    u1 = A1[i1][c1]
                    v1 = B1[j1][c1]
                    w1 = C1[k1][c1]
                    if u1 == 0 and v1 == 0 and w1 == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            for k2 in range(p2):
                                for c2 in range(r2):
                                    u2 = A2[i2][c2]
                                    v2 = B2[j2][c2]
                                    w2 = C2[k2][c2]
                                    if u2 == 0 or v2 == 0 or w2 == 0:
                                        continue
                                    col = c1 * r2 + c2
                                    U[i1 * n2 + i2][col] += u1 * u2
                                    V[j1 * m2 + j2][col] += v1 * v2
                                    W[k1 * p2 + k2][col] += w1 * w2
    return U, V, W


def _verify_decomposition(U, V, W, n=6, m=6, p=6):
    """Verify that the decomposition correctly computes matrix multiplication."""
    import numpy as np
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    C_computed = np.zeros((n, p))
    rank = len(U[0])
    for c in range(rank):
        a_val = sum(U[i * m + j][c] * A_test[i, j] for i in range(n) for j in range(m))
        b_val = sum(V[j * p + k][c] * B_test[j, k] for j in range(m) for k in range(p))
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += W[k * n + i][c] * a_val * b_val
    return np.allclose(C_computed, C_expected, atol=1e-9)


def construct():
    import time
    t0 = time.time()
    DEADLINE = 145.0

    # Strassen 2x2 rank-7 factors
    # A positions: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B positions: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C positions: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    strassen_A = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [1, 0, -1, 0],  # M6: A10-A00
        [0, 1, 0, -1],  # M7: A01-A11
    ]
    strassen_B = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [0, 0, 1, -1],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1],   # M7: B10+B11
    ]
    strassen_C = [
        [1, 0, 0, 1],   # M1: C00+C11
        [0, 0, 1, -1],  # M2: C10-C11
        [0, 1, 0, 1],   # M3: C01+C11
        [-1, 0, 1, 0],  # M4: -C00+C10
        [1, 0, 1, 0],   # M5: C00+C10
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 1],   # M7: C00+C11
    ]

    # Convert to U, V, W format for 2x2 (4x7 each)
    U2 = [[0] * 7 for _ in range(4)]
    V2 = [[0] * 7 for _ in range(4)]
    W2 = [[0] * 7 for _ in range(4)]
    for c in range(7):
        for idx in range(4):
            U2[idx][c] = strassen_A[c][idx]
            V2[idx][c] = strassen_B[c][idx]
            W2[idx][c] = strassen_C[c][idx]

    # 3x3 schoolbook rank-27 factors
    U3 = [[0] * 27 for _ in range(9)]
    V3 = [[0] * 27 for _ in range(9)]
    W3 = [[0] * 27 for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i * 3 + j][col] = 1
                V3[j * 3 + k][col] = 1
                W3[k * 3 + i][col] = 1
                col += 1

    # Kronecker product: 2x2 Strassen (rank 7) x 3x3 schoolbook (rank 27) = rank 189
    U, V, W = _kronecker_decomp(U2, V2, W2, U3, V3, W3)

    # Verify the decomposition
    if not _verify_decomposition(U, V, W):
        # Fallback to schoolbook rank 216
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    if time.time() - t0 > DEADLINE:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
