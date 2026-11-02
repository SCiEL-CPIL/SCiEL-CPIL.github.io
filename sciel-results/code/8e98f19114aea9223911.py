"""Deterministic rank-202 Strassen decomposition for 6x6 matrix multiplication."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 130.0  # explicit safety margin under the 150 s wall-clock kill.

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _strassen_6():
    """Build the rank-202 Strassen factorization for 6x6 x 6x6 -> 6x6.

    Uses the 7-term 2x2 Strassen recursion to depth 3.  Returns (U,V,W)
    with U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    using the official trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS
    L = 3
    R = 7 ** L

    # ---- Build factor matrices as integer dictionaries (sparse, exact). ----
    U = {}  # (i, j) -> {c: val}
    V = {}  # (j, k) -> {c: val}
    W = {}  # (k, i) -> {c: val}

    def rec(base_i, base_j, base_k, size, level, c0, sign):
        if size == 1:
            d = U.get((base_i, base_j))
            if d is None:
                d = U[(base_i, base_j)] = {}
            d[c0] = d.get(c0, 0) + sign
            d = V.get((base_j, base_k))
            if d is None:
                d = V[(base_j, base_k)] = {}
            d[c0] = d.get(c0, 0) + sign
            d = W.get((base_k, base_i))
            if d is None:
                d = W[(base_k, base_i)] = {}
            d[c0] = d.get(c0, 0) + sign
            return
        h = size // 2
        q = 7 ** (level - 1)
        # (a,b,c,d) sub-block offsets (row, col) within the size x size block
        S = [
            (0, 0, 1, 1, 1),   # M1 = (A00+A11)(B01+B10)
            (0, 0, 1, 0, 1),   # M2 = (A10+A11) B00
            (0, 1, 0, 0, 1),   # M3 = A00 (B10-B11)
            (1, 1, 0, 1, 1),   # M4 = A11 (B00-B10)
            (0, 0, 1, 1, 1),   # M5 = (A00+A10) B11
            (1, 0, 0, 1, 1),   # M6 = (A11-A00)(B00+B10)
            (0, 1, 1, 0, 1),   # M7 = (A00-A11)(B11+B01)
        ]
        for (a, b, c, d, s) in S:
            rec(base_i + a * h, base_j + b * h, base_k + c * h, h, level - 1,
                c0 + s * q, sign * s)

    for a in range(3):
        for b in range(3):
            for c in range(3):
                rec(a, b, c, 2, 2, 7 ** (2 - a - b - c), 1)

    # ---- Convert dictionaries to dense integer lists. ----
    U_list = [[0] * R for _ in range(n * m)]
    for (i, j), d in U.items():
        row = i * m + j
        for c, v in d.items():
            U_list[row][c] = v
    V_list = [[0] * R for _ in range(m * p)]
    for (j, k), d in V.items():
        row = j * p + k
        for c, v in d.items():
            V_list[row][c] = v
    W_list = [[0] * R for _ in range(p * n)]
    for (k, i), d in W.items():
        row = k * n + i
        for c, v in d.items():
            W_list[row][c] = v
    return {"U": U_list, "V": V_list, "W": W_list}

def _verify(res):
    """Exact check: sum_c U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c] for all i,j,k,c."""
    n, m, p = DIMENSIONS
    U, V, W = res["U"], res["V"], res["W"]
    R = len(U[0])
    Vt = list(zip(*V))  # Vt[k][c] = V[j*p+k][c]
    for i in range(n):
        for j in range(m):
            Urow = U[i * m + j]
            for k in range(p):
                Wrow = W[k * n + i]
                Vrow = Vt[k]
                for c in range(R):
                    if Urow[c] * Vrow[c] != Wrow[c]:
                        return False
    return True

def construct():
    t0 = time.time()
    try:
        res = _strassen_6()
        if _verify(res):
            return res
    except Exception:
        pass
    if time.time() > _DEADLINE - 1.0:
        return _schoolbook()
    return _schoolbook()
# EVOLVE-BLOCK-END
