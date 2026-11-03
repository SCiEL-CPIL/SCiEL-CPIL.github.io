import time
import numpy as np

DIMENSIONS = [4, 4, 9]


# ---------------------------------------------------------------------------
# Strassen 2x2x2: 7 rank-1 terms.  Block index order: 0=TL,1=TR,2=BL,3=BR.
# ---------------------------------------------------------------------------
_STRASSEN_222 = [
    ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # m1
    ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),   # m2
    ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),  # m3
    ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]),  # m4
    ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # m5
    ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # m6
    ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # m7
]


def _add_strassen_block(U, V, W, col, a_base, b_base, c_base):
    """Add 7 rank-1 Strassen terms for a 2x2x2 block.  Bases are global row-major
    linear indices of the top-left element of each 2x2 block."""
    for (u, v, w) in _STRASSEN_222:
        for t in range(4):
            if u[t]:
                U[a_base + t][col] += u[t]
            if v[t]:
                V[b_base + t][col] += v[t]
            if w[t]:
                W[c_base + t][col] += w[t]
        col += 1
    return col


def _add_444(U, V, W, col, a_base, b_base, c_base):
    """Exact rank-49 decomposition for a 4x4x4 block via 7 Strassen block-sums.
    Block index order (2x2 blocks of a 4x4, row-major): 0=TL,1=TR,2=BL,3=BR."""
    U_sums = [
        [1, 0, 0, 1],   # A00 + A11
        [0, 0, 1, 1],   # A10 + A11
        [1, 0, 0, 0],   # A00
        [0, 0, 0, 1],   # A11
        [1, 1, 0, 0],   # A00 + A01
        [-1, 0, 1, 0],  # A10 - A00
        [0, 1, 0, -1],  # A01 - A11
    ]
    V_sums = [
        [1, 0, 0, 1],   # B00 + B11
        [1, 0, 0, 0],   # B00
        [0, 1, 0, -1],  # B01 - B11
        [0, -1, 1, 0],  # B10 - B00
        [0, 0, 0, 1],   # B11
        [1, 1, 0, 0],   # B00 + B01
        [0, 0, 1, 1],   # B10 + B11
    ]
    W_sums = [
        [1, 0, 0, 1],   # C00 + C11
        [0, 0, 1, 0],   # C10
        [0, 1, 0, 0],   # C01
        [1, 0, 1, 0],   # C00 + C10
        [-1, 1, 0, 0],  # C01 - C00
        [0, 0, 0, 1],   # C11
        [1, 0, 0, 0],   # C00
    ]
    for t in range(7):
        u, v, w = U_sums[t], V_sums[t], W_sums[t]
        for s in range(4):
            if u[s]:
                U[a_base + s][col] += u[s]
            if v[s]:
                V[b_base + s][col] += v[s]
            if w[s]:
                W[c_base + s][col] += w[s]
        col += 1
    return col


def _schoolbook_block(U, V, W, col, a_base, b_base, c_base, ar, ac, bc):
    """Schoolbook for A(ar x ac) * B(ac x bc) -> C(ar x bc).  Bases are global
    row-major linear indices of the top-left element; A rows stride ac, B and C
    rows stride bc."""
    for i in range(ar):
        for j in range(ac):
            for k in range(bc):
                U[a_base + i * ac + j][col] += 1
                V[b_base + j * bc + k][col] += 1
                W[c_base + i * bc + k][col] += 1
                col += 1
    return col


def _verify(U, V, W, n, m, p):
    """Return True iff sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for every
    (i,j,k) and 0 elsewhere.  Fast: single einsum contraction."""
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    # target: T[a=i*m+j, b=j*p+k, c=k*n+i] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i * m + j, j * p + k, k * n + i] - 1.0) > 1e-9:
                    return False
    # ensure no spurious nonzeros
    T2 = T.copy()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T2[i * m + j, j * p + k, k * n + i] = 0.0
    if np.max(np.abs(T2)) > 1e-9:
        return False
    return True


def _build_rank114(n, m, p):
    """Strassen(49) for cols 0-3 + Strassen(49) for cols 4-7 + schoolbook(16) for col 8 = 114."""
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    # Part 1: A(4x4) * B[0:4](4x4) -> C[0:4](4x4). Strassen, rank 49.
    col = _add_444(U, V, W, col, 0, 0, 0)

    # Part 2: A(4x4) * B[4:8](4x4) -> C[4:8](4x4). Strassen, rank 49.
    # B columns 4-7: global B base = 4
    # C columns 4-7: global C base = 4
    col = _add_444(U, V, W, col, 0, 4, 4)

    # Part 3: A(4x4) * B[8:9](4x1) -> C[8:9](4x1). Schoolbook, rank 16.
    # B column 8: global B base = 8 (4 rows x 1 col)
    # C column 8: global C base = 8 (4 rows x 1 col)
    col = _schoolbook_block(U, V, W, col, 0, 8, 8, 4, 4, 1)

    if col != rank:
        rank = col
        U = [row[:rank] for row in U]
        V = [row[:rank] for row in V]
        W = [row[:rank] for row in W]
    return {"U": U, "V": V, "W": W}


def _build_schoolbook(n, m, p):
    """Guaranteed rank n*m*p decomposition."""
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


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin under 150 s

    # Primary: rank-114 decomposition.
    res = _build_rank114(n, m, p)
    if time.time() < deadline and _verify(res["U"], res["V"], res["W"], n, m, p):
        return res

    # Fallback: guaranteed schoolbook rank 144.
    return _build_schoolbook(n, m, p)
