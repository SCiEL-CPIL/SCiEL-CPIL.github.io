"""Block decomposition: Strassen (rank-7) on 2x2x2 blocks + schoolbook elsewhere,
with a low-rank residual pass to shave rank and a fallback to pure schoolbook."""
import time

DIMENSIONS = [3, 5, 6]
_DEADLINE = 145.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _time_left():
    return _DEADLINE - _elapsed()


def _schoolbook(n, m, p):
    """Pure schoolbook rank n*m*p decomposition for (n,m,p)."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W


def _strassen222():
    """Rank-7 Strassen decomposition of 2x2x2 (schoolbook would be rank-8).
    Indices: U row = i*2+j, V row = j*2+k, W row = k*2+i."""
    U = [[0.0] * 7 for _ in range(4)]
    V = [[0.0] * 7 for _ in range(4)]
    W = [[0.0] * 7 for _ in range(4)]
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1.0; U[3][0] = 1.0
    V[0][0] = 1.0; V[3][0] = 1.0
    W[0][0] = 1.0; W[3][0] = 1.0; W[2][0] = -1.0; W[1][0] = -1.0
    # M2 = (A10+A11)B00
    U[2][1] = 1.0; U[3][1] = 1.0
    V[0][1] = 1.0
    W[2][1] = 1.0
    # M3 = A00(B10-B11)
    U[0][2] = 1.0
    V[2][2] = 1.0; V[3][2] = -1.0
    W[0][2] = 1.0
    # M4 = (A01-A10)(B00+B10)
    U[1][3] = 1.0; U[2][3] = -1.0
    V[0][3] = 1.0; V[2][3] = 1.0
    W[1][3] = 1.0; W[2][3] = 1.0
    # M5 = (A00+A01)B11
    U[0][4] = 1.0; U[1][4] = 1.0
    V[3][4] = 1.0
    W[1][4] = 1.0
    # M6 = (A10-A00)(B01+B11)
    U[2][5] = 1.0; U[0][5] = -1.0
    V[1][5] = 1.0; V[3][5] = 1.0
    W[0][5] = 1.0; W[1][5] = 1.0
    # M7 = (A11-A10)(B10-B00)
    U[3][6] = 1.0; U[2][6] = -1.0
    V[2][6] = 1.0; V[0][6] = -1.0
    W[3][6] = 1.0; W[2][6] = -1.0
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check U*V*W reproduces the multiplication tensor via random sampling."""
    import random
    random.seed(12345)
    for _ in range(200):
        A = [[random.uniform(-1, 1) for _ in range(m)] for _ in range(n)]
        B = [[random.uniform(-1, 1) for _ in range(p)] for _ in range(m)]
        C = [[0.0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += A[i][j] * B[j][k]
                C[i][k] = s
        rank = len(W[0])
        # accumulate tensor: T[i,k,j] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
        out = [[0.0] * p for _ in range(n)]
        for c in range(rank):
            for i in range(n):
                for k in range(p):
                    wv = W[k * n + i][c]
                    if wv == 0:
                        continue
                    for j in range(m):
                        uv = U[i * m + j][c]
                        if uv == 0:
                            continue
                        vk = V[j * p + k][c]
                        if vk == 0:
                            continue
                        out[i][k] += uv * vk * wv * A[i][j] * B[j][k]
        for i in range(n):
            for k in range(p):
                if abs(out[i][k] - C[i][k]) > tol * (1.0 + abs(C[i][k])):
                    return False
    return True


def _build_block(n, m, p):
    """Build a block decomposition using Strassen on a 2x2x2 corner + schoolbook
    for the remaining rows/cols. Returns (U,V,W)."""
    if n < 2 or m < 2 or p < 2:
        return _schoolbook(n, m, p)
    U = [[0.0] * 0 for _ in range(n * m)]
    V = [[0.0] * 0 for _ in range(m * p)]
    W = [[0.0] * 0 for _ in range(p * n)]
    sU, sV, sW = _strassen222()
    # Strassen covers i in {0,1}, j in {0,1}, k in {0,1}
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i * m + j].append(sU[i * 2 + j][c])
        for j in range(2):
            for k in range(2):
                V[j * p + k].append(sV[j * 2 + k][c])
        for k in range(2):
            for i in range(2):
                W[k * n + i].append(sW[k * 2 + i][c])
    # Schoolbook for the rest: all (i,j,k) except the 2x2x2 corner
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * m + j].append(1.0)
                V[j * p + k].append(1.0)
                W[k * n + i].append(1.0)
    return U, V, W


def _shave_low_rank(U, V, W, n, m, p):
    """Attempt to reduce rank by detecting linearly dependent columns in U (or V, W).
    If a column of U is a linear combo of others, drop it and adjust.
    This is a conservative exact pass; returns (U,V,W) possibly with lower rank."""
    import numpy as np
    rank = len(U[0])
    # Find the matrix with the most rank deficiency
    def deficiency(M):
        arr = np.array(M, dtype=float)
        r = np.linalg.matrix_rank(arr, tol=1e-9)
        return rank - r
    dU = deficiency(U)
    dV = deficiency(V)
    dW = deficiency(W)
    best = max(dU, dV, dW)
    if best <= 0:
        return U, V, W
    # Greedy: drop dependent columns one at a time, re-verify
    U2, V2, W2 = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    changed = True
    while changed and _time_left() > 15.0:
        changed = False
        # try dropping each column
        for c in range(len(U2[0])):
            candU = [row[:c] + row[c + 1:] for row in U2]
            candV = [row[:c] + row[c + 1:] for row in V2]
            candW = [row[:c] + row[c + 1:] for row in W2]
            # check: is column c of each a linear combo of the others?
            # Simpler: just verify the decomposition still holds (it won't unless col was redundant)
            if _verify(candU, candV, candW, n, m, p, tol=1e-8):
                U2, V2, W2 = candU, candV, candW
                changed = True
                break
    return U2, V2, W2


def construct():
    n, m, p = DIMENSIONS
    # Fallback: pure schoolbook (always valid)
    U, V, W = _schoolbook(n, m, p)
    best = (U, V, W)
    # Try block decomposition
    try:
        bU, bV, bW = _build_block(n, m, p)
        if _verify(bU, bV, bW, n, m, p, tol=1e-9):
            best = (bU, bV, bW)
            # Try shaving
            sU, sV, sW = _shave_low_rank(bU, bV, bW, n, m, p)
            if _verify(sU, sV, sW, n, m, p, tol=1e-9) and len(sU[0]) < len(bU[0]):
                best = (sU, sV, sW)
    except Exception:
        pass
    if _elapsed() > _DEADLINE - 5.0:
        U, V, W = _schoolbook(n, m, p)
    U, V, W = best
    return {"U": U, "V": V, "W": W}
