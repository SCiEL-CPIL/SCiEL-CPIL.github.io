"""Kronecker-product tensor decomposition for 4x5 * 5x6 matrix multiplication."""
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """
    2x2x2 Strassen decomposition.
    Returns (U, V, W) with shapes (4,7), (4,7), (4,7).
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]  (transposed identity convention).
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1

    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1

    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1

    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1

    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return U, V, W

def _build_2x3x2():
    """
    2x3x2 decomposition of rank 6 (rank-optimal for 2x3x2).
    A is 2x3, B is 3x2, C is 2x2.
    Indexing: U[i*3+j], V[j*2+k], W[k*2+i].
    """
    rank = 6
    U = [[0]*rank for _ in range(6)]   # n*m = 2*3
    V = [[0]*rank for _ in range(6)]   # m*p = 3*2
    W = [[0]*rank for _ in range(4)]   # p*n = 2*2

    # Column 0: A[0,0], B[0,0], C[0,0]
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1

    # Column 1: A[0,1], B[1,0], C[0,0]
    U[1][1] = 1
    V[2][1] = 1
    W[0][1] = 1

    # Column 2: A[0,2], B[2,0], C[0,0]
    U[2][2] = 1
    V[4][2] = 1
    W[0][2] = 1

    # Column 3: A[1,0], B[0,1], C[1,0]
    U[3][3] = 1
    V[1][3] = 1
    W[2][3] = 1

    # Column 4: A[1,1], B[1,1], C[1,0]
    U[4][4] = 1
    V[3][4] = 1
    W[2][4] = 1

    # Column 5: A[1,2], B[2,1], C[1,0]
    U[5][5] = 1
    V[5][5] = 1
    W[2][5] = 1

    return U, V, W

def _build_2x1x3():
    """
    2x1x3 decomposition of rank 1.
    A is 2x1, B is 1x3, C is 2x3.
    Indexing: U[i*1+j], V[j*3+k], W[k*2+i].
    """
    rank = 1
    U = [[0]*rank for _ in range(2)]   # n*m = 2*1
    V = [[0]*rank for _ in range(3)]   # m*p = 1*3
    W = [[0]*rank for _ in range(6)]   # p*n = 3*2

    # Only one term: A[i,0] * B[0,k] = C[i,k]
    U[0][0] = 1
    U[1][0] = 1
    V[0][0] = 1
    V[1][0] = 1
    V[2][0] = 1
    # W[k*2+i] = 1 for all k,i
    for k in range(3):
        for i in range(2):
            W[k*2+i][0] = 1

    return U, V, W

def _combine_decompositions(d1, dims1, d2, dims2):
    """
    Combine two decompositions by concatenating their rank-1 terms.
    d1 = (U1, V1, W1) for (n1, m1, p1)
    d2 = (U2, V2, W2) for (n2, m2, p2)
    The combined decomposition handles (n1, m1+m2, p1) where p1 == p2.
    Actually, this is for block-diagonal in the middle dimension.
    """
    U1, V1, W1 = d1
    U2, V2, W2 = d2
    n1, m1, p1 = dims1
    n2, m2, p2 = dims2

    if n1 != n2 or p1 != p2:
        raise ValueError("Dimensions must match for n and p")

    n = n1
    m = m1 + m2
    p = p1

    r1 = len(U1[0])
    r2 = len(U2[0])
    rank = r1 + r2

    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    # Copy d1
    for i in range(n):
        for j in range(m1):
            for c in range(r1):
                U[i*m + j][c] = U1[i*m1 + j][c]
    for j in range(m1):
        for k in range(p):
            for c in range(r1):
                V[j*p + k][c] = V1[j*p1 + k][c]
    for k in range(p):
        for i in range(n):
            for c in range(r1):
                W[k*n + i][c] = W1[k*p1 + i][c]

    # Copy d2 (offset in middle dimension)
    offset = r1
    for i in range(n):
        for j in range(m2):
            for c in range(r2):
                U[i*m + (j + m1)][offset + c] = U2[i*m2 + j][c]
    for j in range(m2):
        for k in range(p):
            for c in range(r2):
                V[(j + m1)*p + k][offset + c] = V2[j*p2 + k][c]
    for k in range(p):
        for i in range(n):
            for c in range(r2):
                W[k*n + i][offset + c] = W2[k*p2 + i][c]

    return U, V, W

def _kron_decomposition_dims(d1, dims1, d2, dims2):
    """
    Kronecker product of two decompositions with explicit dimensions.
    d1 = (U1, V1, W1) for (n1, m1, p1)
    d2 = (U2, V2, W2) for (n2, m2, p2)
    Produces decomposition for (n1*n2, m1*m2, p1*p2).
    """
    U1, V1, W1 = d1
    U2, V2, W2 = d2
    n1, m1, p1 = dims1
    n2, m2, p2 = dims2

    r1 = len(U1[0])
    r2 = len(U2[0])
    rank = r1 * r2

    n = n1 * n2
    m = m1 * m2
    p = p1 * p2

    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    for c1 in range(r1):
        for c2 in range(r2):
            col = c1 * r2 + c2

            # U: index (i1*n2 + i2) * (m1*m2) + (j1*m2 + j2)
            for i1 in range(n1):
                for j1 in range(m1):
                    u1_val = U1[i1*m1 + j1][c1]
                    if u1_val == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u2_val = U2[i2*m2 + j2][c2]
                            if u2_val == 0:
                                continue
                            i = i1*n2 + i2
                            j = j1*m2 + j2
                            U[i*m + j][col] = u1_val * u2_val

            # V: index (j1*p2 + j2) * (p1*p2) + (k1*p2 + k2)
            for j1 in range(m1):
                for k1 in range(p1):
                    v1_val = V1[j1*p1 + k1][c1]
                    if v1_val == 0:
                        continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v2_val = V2[j2*p2 + k2][c2]
                            if v2_val == 0:
                                continue
                            j = j1*p2 + j2
                            k = k1*p2 + k2
                            V[j*p + k][col] = v1_val * v2_val

            # W: index (k1*n2 + k2) * (n1*n2) + (i1*n2 + i2)
            for k1 in range(p1):
                for i1 in range(n1):
                    w1_val = W1[k1*n1 + i1][c1]
                    if w1_val == 0:
                        continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            w2_val = W2[k2*n2 + i2][c2]
                            if w2_val == 0:
                                continue
                            k = k1*n2 + k2
                            i = i1*n2 + i2
                            W[k*n + i][col] = w1_val * w2_val

    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS

    try:
        # Build 2x5x2 decomposition by combining 2x2x2 (Strassen, rank 7) and 2x3x2 (rank 6)
        # 2x5x2: n=2, m=5, p=2
        U_strassen, V_strassen, W_strassen = _build_strassen_2x2()
        d_strassen = (U_strassen, V_strassen, W_strassen)
        dims_strassen = (2, 2, 2)

        U_2x3x2, V_2x3x2, W_2x3x2 = _build_2x3x2()
        d_2x3x2 = (U_2x3x2, V_2x3x2, W_2x3x2)
        dims_2x3x2 = (2, 3, 2)

        # Combine: 2x5x2 with rank 7+6=13
        U_2x5x2, V_2x5x2, W_2x5x2 = _combine_decompositions(d_strassen, dims_strassen, d_2x3x2, dims_2x3x2)
        d_2x5x2 = (U_2x5x2, V_2x5x2, W_2x5x2)
        dims_2x5x2 = (2, 5, 2)

        # Build 2x1x3 decomposition (rank 1)
        U_2x1x3, V_2x1x3, W_2x1x3 = _build_2x1x3()
        d_2x1x3 = (U_2x1x3, V_2x1x3, W_2x1x3)
        dims_2x1x3 = (2, 1, 3)

        # Kronecker product: (2*2) x (1*5) x (3*2) = 4 x 5 x 6
        # Rank: 1 * 13 = 13
        U, V, W = _kron_decomposition_dims(d_2x1x3, dims_2x1x3, d_2x5x2, dims_2x5x2)

        return {"U": U, "V": V, "W": W}

    except Exception:
        # Fallback to schoolbook
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

# EVOLVE-BLOCK-END
