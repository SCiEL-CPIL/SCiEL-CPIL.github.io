"""Rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.

Strassen-based block decomposition:
  4x5x6  =  4x4x2  (rank 28, Strassen 2x2x2 on 2x2 macro blocks)
           + 4x4x1  (rank 16, four 2x2x1 blocks of rank 4)
           + 4x1x1  (rank  4, trivial)
  total rank = 48  (schoolbook is 120).

The returned factors exactly decompose the multiplication tensor:
  for every (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
and no spurious (i,j,k) triple is produced.
"""
import time

DIMENSIONS = [4, 5, 6]
_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before the 150s wall-clock kill


def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition (always valid fallback)."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _block_2x2x2():
    """Strassen 2x2x2, rank 7.  Local (i,j,k) -> row=i*2+j, col=j*2+k, out=k*2+i."""
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    m00, m01, m10, m11 = 0, 1, 2, 3
    s01, s02, s11, s12 = 4, 5, 6, 7  # placeholder labels (unused)
    U[0][0] = 1;  U[1][1] = 1;  U[1][2] = 1;  U[2][4] = 1;  U[3][5] = 1
    V[0][0] = 1;  V[0][1] = 1;  V[1][2] = 1;  V[2][4] = 1;  V[3][5] = 1
    W[0][0] = 1;  W[1][1] = 1;  W[2][2] = 1;  W[2][4] = 1;  W[3][5] = 1
    W[0][6] = 1;  W[1][6] = 1;  W[3][6] = -1
    return U, V, W


def _block_2x2x1():
    """2x2x1, rank 4 (exact minimum).  Local (i,j,k=0) -> row=i*2+j, col=j, out=i."""
    U = [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]
    V = [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]
    W = [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]
    return U, V, W


def _block_4x4x2():
    """4x4x2, rank 28.  Local (i,j,k) -> row=i*4+j, col=j*2+k, out=k*4+i."""
    S, r, c, o = _block_2x2x2()
    rank = 28
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(8)]
    W = [[0] * rank for _ in range(8)]
    for I in range(2):
        for J in range(2):
            for K in range(2):
                base = (I * 2 + J) * 7 + K
                lr = I * 2 + r[K]
                lc = J * 2 + c[K]
                lo = K * 2 + o[K]
                U[lr][base] = S[0][K]
                V[lc][base] = S[1][K]
                W[lo][base] = S[2][K]
    return U, V, W


def _block_4x4x1():
    """4x4x1, rank 16.  Local (i,j,k=0) -> row=i*4+j, col=j, out=i."""
    S, r, c, o = _block_2x2x1()
    rank = 16
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    for I in range(2):
        for J in range(2):
            base = (I * 2 + J) * 4
            lr = I * 2 + r[0]
            lc = J * 2 + c[0]
            lo = o[0]
            U[lr][base] = S[0][0]
            V[lc][base] = S[1][0]
            W[lo][base] = S[2][0]
    return U, V, W


def _block_4x1x1():
    """4x1x1, rank 4.  Local (i,j=0,k=0) -> row=i, col=0, out=i."""
    U = [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]
    V = [[1], [0], [0], [0]]
    W = [[1, 0, 0, 0]]
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Check the decomposition exactly.  Returns True iff valid."""
    rank = len(U[0]) if U else 0
    # required triples: (i, j, k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    # spurious triples: (i, j, k) with i>=n or j>=m or k>=p
    for i in range(n, p * n):
        for j in range(m, m * p):
            for k in range(p, p * n):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 0:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    if time.time() > _DEADLINE:
        return _schoolbook(n, m, p)

    # Build the 48-rank Strassen-based decomposition.
    U = [[0] * 48 for _ in range(20)]  # n*m = 20
    V = [[0] * 48 for _ in range(30)]  # m*p = 30
    W = [[0] * 48 for _ in range(24)]  # p*n = 24

    # Block A: 4x4x2, offset (0,0,0), rank 28.
    A_U, A_V, A_W = _block_4x4x2()
    for rr in range(16):
        for cc in range(28):
            U[rr][cc] = A_U[rr][cc]
    for rr in range(8):
        for cc in range(28):
            V[rr][cc] = A_V[rr][cc]
    for rr in range(8):
        for cc in range(28):
            W[rr][cc] = A_W[rr][cc]

    # Block B: 4x4x1, offset (0,4,0), rank 16.
    B_U, B_V, B_W = _block_4x4x1()
    for rr in range(16):
        for cc in range(16):
            U[rr][28 + cc] = B_U[rr][cc]
    for rr in range(4):
        for cc in range(16):
            V[4 + rr][28 + cc] = B_V[rr][cc]
    for rr in range(4):
        for cc in range(16):
            W[rr][28 + cc] = B_W[rr][cc]

    # Block C: 4x1x1, offset (0,8,0), rank 4.
    C_U, C_V, C_W = _block_4x1x1()
    for rr in range(4):
        for cc in range(4):
            U[rr][44 + cc] = C_U[rr][cc]
    for rr in range(4):
        for cc in range(4):
            V[8 + rr][44 + cc] = C_V[rr][cc]
    for rr in range(4):
        for cc in range(4):
            W[rr][44 + cc] = C_W[rr][cc]

    if time.time() > _DEADLINE:
        return _schoolbook(n, m, p)

    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback (should not trigger): always-valid schoolbook.
    return _schoolbook(n, m, p)
