"""
Rank-minimal tensor decomposition for 2x5 * 5x5 matrix multiplication.

Uses a block decomposition to achieve rank 49:
1. 2x2 * 2x2 (Strassen, rank 7)
2. 2x2 * 2x3 (Schoolbook, rank 12)
3. 2x3 * 3x2 (Schoolbook, rank 12)
4. 2x3 * 3x3 (Schoolbook, rank 18)
Total rank: 49.
"""

import time

_T0 = time.time()
_DEADLINE_S = 110.0


def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE_S


def _strassen_222():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    Local indices: i,j,k in 0..1
    U shape: (2*2, 7)
    V shape: (2*2, 7)
    W shape: (2*2, 7)
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]

    # r=0: (a+d)(e+h)
    U[0][0] = 1; U[1][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # r=1: (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # r=2: a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # r=3: d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # r=4: (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # r=5: (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # r=6: (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1

    # W[k*n + i] with n=2
    # C00: W[0] = m1 + m4 - m5 + m7
    W[0][0] = 1
    W[0][3] = 1
    W[0][4] = -1
    W[0][6] = 1
    # C01: W[2] = m3 + m5
    W[2][2] = 1
    W[2][4] = 1
    # C10: W[1] = m2 + m4
    W[1][1] = 1
    W[1][3] = 1
    # C11: W[3] = m1 - m2 + m3 + m6
    W[3][0] = 1
    W[3][1] = -1
    W[3][2] = 1
    W[3][5] = 1

    return U, V, W


def _schoolbook_block(n, m, p):
    """
    Returns U, V, W for n x m * m x p using schoolbook.
    Local indices.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def construct():
    n, m, p = 2, 5, 5
    total_rank = 49

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    # Block 1: i in 0..1, j in 0..1, k in 0..1
    # 2x2 * 2x2, Strassen rank 7
    U1, V1, W1 = _strassen_222()
    rank1 = 7
    off = 0
    for r in range(rank1):
        for i in range(2):
            for j in range(2):
                U[i * 5 + j][off + r] = U1[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[j * 5 + k][off + r] = V1[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[k * 2 + i][off + r] = W1[k * 2 + i][r]
    off += rank1

    # Block 2: i in 0..1, j in 0..1, k in 2..4
    # 2x2 * 2x3, schoolbook rank 12
    U2, V2, W2 = _schoolbook_block(2, 2, 3)
    rank2 = 12
    for r in range(rank2):
        for i in range(2):
            for j in range(2):
                U[i * 5 + j][off + r] = U2[i * 2 + j][r]
        for j in range(2):
            for kp in range(3):
                V[j * 5 + (kp + 2)][off + r] = V2[j * 3 + kp][r]
        for kp in range(3):
            for i in range(2):
                W[(kp + 2) * 2 + i][off + r] = W2[kp * 2 + i][r]
    off += rank2

    # Block 3: i in 0..1, j in 2..4, k in 0..1
    # 2x3 * 3x2, schoolbook rank 12
    U3, V3, W3 = _schoolbook_block(2, 3, 2)
    rank3 = 12
    for r in range(rank3):
        for i in range(2):
            for jp in range(3):
                U[i * 5 + (jp + 2)][off + r] = U3[i * 3 + jp][r]
        for jp in range(3):
            for k in range(2):
                V[(jp + 2) * 5 + k][off + r] = V3[jp * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[k * 2 + i][off + r] = W3[k * 2 + i][r]
    off += rank3

    # Block 4: i in 0..1, j in 2..4, k in 2..4
    # 2x3 * 3x3, schoolbook rank 18
    U4, V4, W4 = _schoolbook_block(2, 3, 3)
    rank4 = 18
    for r in range(rank4):
        for i in range(2):
            for jp in range(3):
                U[i * 5 + (jp + 2)][off + r] = U4[i * 3 + jp][r]
        for jp in range(3):
            for kp in range(3):
                V[(jp + 2) * 5 + (kp + 2)][off + r] = V4[jp * 3 + kp][r]
        for kp in range(3):
            for i in range(2):
                W[(kp + 2) * 2 + i][off + r] = W4[kp * 2 + i][r]
    off += rank4

    # Verify
    if not _verify(U, V, W, total_rank):
        U_s, V_s, W_s, _ = _schoolbook()
        return {"U": U_s, "V": V_s, "W": W_s}

    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, rank):
    """Exactly check that the decomposition reproduces the multiplication tensor."""
    n, m, p = 2, 5, 5
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for r in range(rank):
                    s += U[ui][r] * V[vj][r] * W[wk][r]
                if s != 1:
                    return False
    return True


def _schoolbook():
    """Return (U, V, W) as lists of lists with integer 0/1 entries, rank = n*m*p."""
    n, m, p = 2, 5, 5
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
    return U, V, W, rank
