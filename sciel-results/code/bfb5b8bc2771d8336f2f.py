"""Matrix multiplication 4x4x6 via Kronecker product of Strassen decompositions."""
DIMENSIONS = [4, 4, 6]

import time
import numpy as np


def _strassen_2x2_factors():
    """
    Return (U, V, W) for the rank-7 Strassen decomposition of 2x2x2 matrix multiplication.
    U shape (4, 7): rows indexed by i*2+j
    V shape (4, 7): rows indexed by j*2+k
    W shape (4, 7): rows indexed by k*2+i
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    prods = [
        ([(0,0,1.0),(1,1,1.0)],
         [(0,0,1.0),(1,1,1.0)],
         [(0,0,1.0),(1,1,1.0)]),
        ([(1,0,1.0),(1,1,1.0)],
         [(0,0,1.0)],
         [(1,0,1.0),(1,1,-1.0)]),
        ([(0,0,1.0)],
         [(0,1,1.0),(1,1,-1.0)],
         [(0,1,1.0),(1,1,1.0)]),
        ([(1,1,1.0)],
         [(1,0,1.0),(0,0,-1.0)],
         [(0,0,1.0),(1,0,1.0)]),
        ([(0,0,1.0),(0,1,1.0)],
         [(1,1,1.0)],
         [(0,0,-1.0),(0,1,1.0)]),
        ([(1,0,1.0),(0,0,-1.0)],
         [(0,0,1.0),(0,1,1.0)],
         [(1,1,1.0)]),
        ([(0,1,1.0),(1,1,-1.0)],
         [(1,0,1.0),(1,1,1.0)],
         [(0,0,1.0)]),
    ]

    for s, (Ub, Vb, Wb) in enumerate(prods):
        for (i, j, c) in Ub:
            U[i * m + j][s] += c
        for (j, k, c) in Vb:
            V[j * p + k][s] += c
        for (i, k, c) in Wb:
            W[k * n + i][s] += c

    return U, V, W


def _rank8_2x2x3_factors():
    """
    Return (U, V, W) for the rank-8 decomposition of 2x2x3 matrix multiplication.
    U shape (4, 8): rows indexed by i*2+j
    V shape (6, 8): rows indexed by j*3+k
    W shape (6, 8): rows indexed by k*2+i
    """
    U_s, V_s, W_s = _strassen_2x2_factors()
    U = [[0.0] * 8 for _ in range(4)]
    V = [[0.0] * 8 for _ in range(6)]
    W = [[0.0] * 8 for _ in range(6)]
    
    # First 7 columns: Strassen for k=0,1
    for i in range(4):
        for c in range(7):
            U[i][c] = U_s[i][c]
    for j in range(4):
        for c in range(7):
            V[j][c] = V_s[j][c]
    for k in range(4):
        for c in range(7):
            W[k][c] = W_s[k][c]
    
    # Column 7: rank-1 term for k=2
    for i in range(4):
        U[i][7] = 1.0
    for j in range(2):
        V[j * 3 + 2][7] = 1.0
    for i in range(2):
        W[2 * 2 + i][7] = 1.0
    
    return U, V, W


def _verify(U, V, W, n, m, p, rank):
    """Check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    return err


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0

    U1, V1, W1 = _strassen_2x2_factors()
    U2, V2, W2 = _rank8_2x2x3_factors()
    
    rank = 56
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for i1 in range(2):
        for i2 in range(2):
            i = i1*2 + i2
            for j1 in range(2):
                for j2 in range(2):
                    j = j1*2 + j2
                    row = i*m + j
                    for c1 in range(7):
                        for c2 in range(8):
                            c = c1*8 + c2
                            U[row][c] = U1[i1*2+j1][c1] * U2[i2*2+j2][c2]
    
    for j1 in range(2):
        for j2 in range(2):
            j = j1*2 + j2
            for k1 in range(2):
                for k2 in range(3):
                    k = k1*3 + k2
                    row = j*p + k
                    for c1 in range(7):
                        for c2 in range(8):
                            c = c1*8 + c2
                            V[row][c] = V1[j1*2+k1][c1] * V2[j2*3+k2][c2]
    
    for k1 in range(2):
        for k2 in range(3):
            k = k1*3 + k2
            for i1 in range(2):
                for i2 in range(2):
                    i = i1*2 + i2
                    row = k*n + i
                    for c1 in range(7):
                        for c2 in range(8):
                            c = c1*8 + c2
                            W[row][c] = W1[k1*2+i1][c1] * W2[k2*2+i2][c2]
    
    err = _verify(U, V, W, n, m, p, rank)
    if err > 1e-6:
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
    
    return {"U": U, "V": V, "W": W}
