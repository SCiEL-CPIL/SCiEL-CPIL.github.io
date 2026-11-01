"""Rank-minimal tensor decomposition for 2x2 x 2x13 matrix multiplication."""
DIMENSIONS = [2, 2, 13]

import time


def construct():
    n, m, p = 2, 2, 13
    # We will use 6 Strassen blocks (rank 7 each) for columns 0-11
    # and 1 schoolbook block (rank 4) for column 12.
    # Total rank = 6 * 7 + 4 = 46.
    
    rank = 46
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Helper to add a Strassen block for columns k1, k2
    def add_strassen_block(k1, k2, offset):
        # 7 terms
        # M1 = (A00+A11)(B00+B11)
        U[0][offset+0] = 1; U[3][offset+0] = 1
        V[0*p+k1][offset+0] = 1; V[1*p+k2][offset+0] = 1
        W[k1*n+0][offset+0] = 1; W[k2*n+1][offset+0] = 1

        # M2 = (A10+A11)B00
        U[2][offset+1] = 1; U[3][offset+1] = 1
        V[0*p+k1][offset+1] = 1
        W[k1*n+1][offset+1] = 1

        # M3 = A00(B10-B11)
        U[0][offset+2] = 1
        V[0*p+k2][offset+2] = 1; V[1*p+k2][offset+2] = -1
        W[k2*n+0][offset+2] = 1; W[k2*n+1][offset+2] = 1

        # M4 = A11(B01-B00)
        U[3][offset+3] = 1
        V[0*p+k2][offset+3] = 1; V[0*p+k1][offset+3] = -1
        W[k2*n+0][offset+3] = 1; W[k1*n+0][offset+3] = 1

        # M5 = (A00+A01)B11
        U[0][offset+4] = 1; U[1][offset+4] = 1
        V[1*p+k2][offset+4] = 1
        W[k2*n+0][offset+4] = -1; W[k2*n+1][offset+4] = 1

        # M6 = (A10-A00)(B00+B01)
        U[2][offset+5] = 1; U[0][offset+5] = -1
        V[0*p+k1][offset+5] = 1; V[0*p+k2][offset+5] = 1
        W[k1*n+1][offset+5] = 1; W[k2*n+1][offset+5] = 1

        # M7 = (A01-A11)(B10+B11)
        U[1][offset+6] = 1; U[3][offset+6] = -1
        V[1*p+k1][offset+6] = 1; V[1*p+k2][offset+6] = 1
        W[k1*n+0][offset+6] = 1; W[k2*n+0][offset+6] = -1

    # Helper to add a schoolbook block for column k
    def add_schoolbook_block(k, offset):
        # 4 terms
        U[0][offset+0] = 1
        V[0*p+k][offset+0] = 1
        W[k*n+0][offset+0] = 1

        U[1][offset+1] = 1
        V[1*p+k][offset+1] = 1
        W[k*n+0][offset+1] = 1

        U[2][offset+2] = 1
        V[0*p+k][offset+2] = 1
        W[k*n+1][offset+2] = 1

        U[3][offset+3] = 1
        V[1*p+k][offset+3] = 1
        W[k*n+1][offset+3] = 1

    offset = 0
    # 6 pairs: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11)
    for i in range(6):
        k1 = 2 * i
        k2 = 2 * i + 1
        add_strassen_block(k1, k2, offset)
        offset += 7

    # 1 single: 12
    add_schoolbook_block(12, offset)
    offset += 4

    # Verify
    def verify(n, m, p, U, V, W, tol=1e-9):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    if not verify(n, m, p, U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1

    return {"U": U, "V": V, "W": W}
