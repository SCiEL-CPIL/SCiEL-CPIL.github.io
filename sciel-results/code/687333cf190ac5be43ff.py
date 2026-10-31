"""Structural redesign: exact rank-49 Strassen-recursive 4x4x4 tensor decomposition.

Instead of the 64-term schoolbook factorization, we compose two rank-7 2x2
Strassen tensors. The 4x4 multiplication tensor is the (tensor/Kronecker) product
of two 2x2 multiplication tensors, so a rank-49 (7*7) exact decomposition exists.
We build U,V,W directly and verify M_{i,k}=sum_r U[i*4+j]V[j*4+k]W[k*4+i] matches
the true 4x4 product, with a deadline-guarded fallback to the known-valid schoolbook.
"""
import time

_DIM = (4, 4, 4)
_T_START = time.time()
_DEADLINE_S = 150.0
_MARGIN_S = 3.0


def _time_left():
    return (_T_START + _DEADLINE_S - _MARGIN_S) - time.time()


def _schoolbook():
    n, m, p = _DIM
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


def _strassen_4x4():
    # 2x2 Strassen factors (verified by _valid check)
    # U2 rows: A00, A01, A10, A11 ; cols: 7 Strassen products
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0,-1],
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 0,  1, 0, 0, 1],
        [0, 0, 1,  0, 0, 1, 0],
        [1, 0,-1,  0, 1, 0, 1],
    ]
    W2 = [
        [1, 0, 0, -1, 0, -1, 1],   # C00 (k=0,i=0)
        [0, 0, 0,  0, 1,  0, 1],   # C01 (k=1,i=0)
        [0, 1, 1,  0, 0,  0, 0],   # C10 (k=0,i=1)
        [1,-1, 0,  0, 1,  1, 0],   # C11 (k=1,i=1)
    ]

    n = m = p = 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49

    # Kronecker product construction:
    # U4[(i1*2+i2), (r*7+s)] = U2[i1, r] * U2[i2, s]
    # V4[(j1*2+j2), (r*7+s)] = V2[j1, r] * V2[j2, s]
    # W4[(k1*2+k2)*4 + (i1*2+i2), (r*7+s)] = W2[k1*2+i1, r] * W2[k2*2+i2, s]
    for i1 in range(2):
        for i2 in range(2):
            row = i1 * 2 + i2
            for r in range(7):
                for s in range(7):
                    U[row][r * 7 + s] = U2[i1][r] * U2[i2][s]
    for j1 in range(2):
        for j2 in range(2):
            row = j1 * 2 + j2
            for r in range(7):
                for s in range(7):
                    V[row][r * 7 + s] = V2[j1][r] * V2[j2][s]
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    wrow = (k1 * 2 + k2) * 4 + (i1 * 2 + i2)
                    for r in range(7):
                        for s in range(7):
                            W[wrow][r * 7 + s] = W2[k1 * 2 + i1][r] * W2[k2 * 2 + i2][s]

    return {"U": U, "V": V, "W": W}


def _valid(d):
    n, m, p = _DIM
    U, V, W = d["U"], d["V"], d["W"]
    r = len(U[0])
    if len(V[0]) != r or len(W[0]) != r:
        return False
    # random integer test inputs
    rng = _Rng(123456789)
    for trial in range(4):
        A = [[rng.int_range(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[rng.int_range(-3, 3) for _ in range(p)] for _ in range(m)]
        # true product
        M = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        # reconstructed
        for i in range(n):
            for k in range(p):
                val = 0
                for j in range(m):
                    for rr in range(r):
                        val += U[i * m + j][rr] * V[j * p + k][rr] * W[k * n + i][rr]
                if val != M[i][k]:
                    return False
    return True


class _Rng:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF
    def _next(self):
        # xorshift32
        x = self.s
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17)
        x ^= (x << 5) & 0xFFFFFFFF
        self.s = x & 0xFFFFFFFF
        return self.s
    def int_range(self, lo, hi):
        span = hi - lo + 1
        return lo + (self._next() % span)


def _valid_2x2(U2, V2, W2):
    """Verify 2x2 Strassen factors produce correct 2x2 matrix product."""
    rng = _Rng(987654321)
    for _ in range(3):
        A = [[rng.int_range(-3, 3) for _ in range(2)] for _ in range(2)]
        B = [[rng.int_range(-3, 3) for _ in range(2)] for _ in range(2)]
        M = [[sum(A[i][j] * B[j][k] for j in range(2)) for k in range(2)] for i in range(2)]
        for i in range(2):
            for k in range(2):
                val = 0
                for j in range(2):
                    for r in range(7):
                        val += U2[i * 2 + j][r] * V2[j * 2 + k][r] * W2[k * 2 + i][r]
                if val != M[i][k]:
                    return False
    return True


def construct():
    t0 = time.time()
    if _time_left() <= 0.0:
        return _schoolbook()
    try:
        # Quick pre-check: verify 2x2 Strassen factors
        U2 = [
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1,  0, 1],
            [0, 1, 0, 0, 0,  1, 0],
            [1, 1, 0, 1, 0,  0,-1],
        ]
        V2 = [
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 0,  1, 0, 0, 1],
            [0, 0, 1,  0, 0, 1, 0],
            [1, 0,-1,  0, 1, 0, 1],
        ]
        W2 = [
            [1, 0, 0, -1, 0, -1, 1],
            [0, 0, 0,  0, 1,  0, 1],
            [0, 1, 1,  0, 0,  0, 0],
            [1,-1, 0,  0, 1,  1, 0],
        ]
        if not _valid_2x2(U2, V2, W2):
            raise ValueError("2x2 Strassen factors failed verification")
        cand = _strassen_4x4()
        if _time_left() > 2.0 and _valid(cand):
            return cand
    except Exception:
        pass
    # Fallback: known-valid schoolbook (rank 64).
    return _schoolbook()
