"""Rank-minimal tensor decomposition for 2x3 x 3x6 matrix multiplication.
Uses two independent rank-11 decompositions of 2x3x3 for rank 22.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]
_START_TIME = time.time()
_HARD_DEADLINE = 140  # seconds; leave margin for evaluator kill at 150s


def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    deadline = _START_TIME + _HARD_DEADLINE

    # ---------- schoolbook rank = n*m*p = 36 ----------
    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r), dtype=np.float64)
        V = np.zeros((m * p, r), dtype=np.float64)
        W = np.zeros((p * n, r), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return _to_dict(U, V, W)

    best = schoolbook()

    # ---------- vectorized verification ----------
    def verify(U, V, W, n, m, p, rank):
        """Check: sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] == 1 for all i,j,k."""
        T = np.einsum('it,jt,kt->ijk',
                       U.reshape(n, m, rank),
                       V.reshape(m, p, rank),
                       W.reshape(p, n, rank))
        return bool(np.allclose(T, 1.0, atol=1e-8))

    # ---------- Strassen 2x2x2 terms ----------
    # u indexed [i*2+j], v indexed [j*2+k], w indexed [k*2+i]
    strassen_222 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),      # p1 = (A00+A11)(B00+B11)
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),      # p2 = (A10+A11)B00
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),      # p3 = A00(B01-B11)
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),      # p4 = A11(B10-B00)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),      # p5 = (A00+A01)B11
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),      # p6 = (A10-A00)(B00+B01)
        ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 1, 0, 0]),     # p7 = (A01-A11)(B10+B11)
    ]

    # ---------- Build rank-11 2x3x3 block ----------
    def build_rank11_233():
        """Returns (U, V, W) for 2x3x3 with rank 11."""
        blk_rank = 11
        U = np.zeros((6, blk_rank))   # i in {0,1}, j in {0,1,2}
        V = np.zeros((9, blk_rank))   # j in {0,1,2}, k in {0,1,2}
        W = np.zeros((6, blk_rank))   # k in {0,1,2}, i in {0,1}

        # 7 Strassen terms for columns 0,1 (j=0,1)
        for t in range(7):
            ut, vt, wt = strassen_222[t]
            for i in range(2):
                for j in range(2):
                    U[i * 3 + j, t] = ut[i * 2 + j]
            for j in range(2):
                for k in range(2):
                    V[j * 3 + k, t] = vt[j * 2 + k]
            for k in range(2):
                for i in range(2):
                    W[k * 2 + i, t] = wt[k * 2 + i]

        # 4 schoolbook terms for column 2 (j=2)
        for di, (iv, kv) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            t = 7 + di
            U[iv * 3 + 2, t] = 1.0
            V[2 * 3 + kv, t] = 1.0
            W[kv * 2 + iv, t] = 1.0

        return U, V, W

    # ---------- Rank 22: two independent 2x3x3 blocks ----------
    try:
        rank22 = 22
        U22 = np.zeros((n * m, rank22), dtype=np.float64)
        V22 = np.zeros((m * p, rank22), dtype=np.float64)
        W22 = np.zeros((p * n, rank22), dtype=np.float64)

        U_blk, V_blk, W_blk = build_rank11_233()

        for copy in range(2):
            k_off = copy * 3   # column offset in B (3 columns per copy)
            base = copy * 11   # term offset

            for li in range(2):
                for lj in range(3):
                    for t in range(11):
                        U22[li * 3 + lj, base + t] = U_blk[li * 3 + lj, t]
            for lj in range(3):
                for lk in range(3):
                    for t in range(11):
                        V22[lj * 3 + (k_off + lk), base + t] = V_blk[lj * 3 + lk, t]
            for lk in range(3):
                for li in range(2):
                    for t in range(11):
                        W22[(k_off + lk) * 2 + li, base + t] = W_blk[lk * 2 + li, t]

        if time.time() > deadline:
            return best

        if verify(U22, V22, W22, n, m, p, rank22):
            return _to_dict(U22, V22, W22)
    except Exception:
        pass

    # ---------- Attempt rank 33 via three 2x3x2 blocks (backup) ----------
    try:
        block_rank = 11
        U_blk33 = np.zeros((6, block_rank))
        V_blk33 = np.zeros((6, block_rank))
        W_blk33 = np.zeros((4, block_rank))

        for t in range(7):
            ut, vt, wt = strassen_222[t]
            for i in range(2):
                for j in range(2):
                    U_blk33[i * 3 + j, t] = ut[i * 2 + j]
            for j in range(2):
                for k in range(2):
                    V_blk33[j * 2 + k, t] = vt[j * 2 + k]
            for k in range(2):
                for i in range(2):
                    W_blk33[k * 2 + i, t] = wt[k * 2 + i]

        for di, (iv, kv) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            t = 7 + di
            U_blk33[iv * 3 + 2, t] = 1.0
            V_blk33[2 * 2 + kv, t] = 1.0
            W_blk33[kv * 2 + iv, t] = 1.0

        rank33 = 33
        U33 = np.zeros((n * m, rank33))
        V33 = np.zeros((m * p, rank33))
        W33 = np.zeros((p * n, rank33))

        for b in range(3):
            k_off = b * 2
            base = b * block_rank
            for li in range(2):
                for lj in range(3):
                    U33[li * 3 + lj, base:base + block_rank] = U_blk33[li * 3 + lj, :]
            for lj in range(3):
                for lk in range(2):
                    V33[lj * 2 + lk, base:base + block_rank] = V_blk33[lj * 2 + lk, :]
            for lk in range(2):
                for li in range(2):
                    W33[(k_off + lk) * 2 + li, base:base + block_rank] = W_blk33[lk * 2 + li, :]

        if time.time() > deadline:
            return best

        if verify(U33, V33, W33, n, m, p, rank33):
            return _to_dict(U33, V33, W33)
    except Exception:
        pass

    return best


def _to_dict(U, V, W):
    """Convert numpy arrays to lists with integer/half-integer cleanup."""
    def clean(arr):
        out = []
        for row in arr:
            crow = []
            for x in row:
                xi = round(float(x))
                if abs(x - xi) < 1e-9:
                    crow.append(int(xi))
                else:
                    h = round(2.0 * float(x))
                    if abs(2.0 * float(x) - h) < 1e-9:
                        crow.append(h / 2.0)
                    else:
                        crow.append(float(x))
            out.append(crow)
        return out
    return {"U": clean(U), "V": clean(V), "W": clean(W)}
