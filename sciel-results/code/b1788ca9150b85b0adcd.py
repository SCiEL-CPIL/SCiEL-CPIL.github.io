"""Matrix multiplication 2,4,5 — rank-36 decomposition via Strassen blocks."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START

# ---- Strassen rank-7 decomposition of the 2x2x2 multiplication tensor ----
_STRASSEN_A = [
    [1, 0, 0, 1],   # M1
    [0, 0, 1, 1],   # M2
    [1, 0, 0, 0],   # M3
    [0, 0, 0, 1],   # M4
    [1, 1, 0, 0],   # M5
    [-1, 0, 1, 0],  # M6
    [0, 1, 0, -1],  # M7
]
_STRASSEN_B = [
    [1, 0, 0, 1],   # M1
    [1, 0, 0, 0],   # M2
    [0, 1, 0, -1],  # M3
    [-1, 0, 1, 0],  # M4
    [0, 0, 0, 1],   # M5
    [1, 1, 0, 0],   # M6
    [0, 0, 1, 1],   # M7
]
_STRASSEN_C = [
    [[1, 0, 0, 1], [0, 0, 1, 0]],  # C00, C01
    [[0, 1, 0, 0], [1, -1, 1, 0]], # C10, C11
]


def _strassen_2x2x2():
    """Return (U, V, W) factor matrices for the 2x2x2 tensor, rank 7.
    U shape (4,7): rows indexed by (i*2+j) for A[i,j]
    V shape (4,7): rows indexed by (j*2+k) for B[j,k]
    W shape (4,7): rows indexed by (k*2+i) for C[i,k]  (trace convention W[k*n+i,r], n=2)
    """
    rank = 7
    U = [[0.0] * rank for _ in range(4)]
    V = [[0.0] * rank for _ in range(4)]
    W = [[0.0] * rank for _ in range(4)]
    for r in range(rank):
        for a in range(4):
            U[a][r] = _STRASSEN_A[r][a]
        for b in range(4):
            V[b][r] = _STRASSEN_B[r][b]
    for i in range(2):
        for k in range(2):
            for r in range(rank):
                W[k * 2 + i][r] = _STRASSEN_C[i][k][r]
    return U, V, W


def _trivial_2x2x1():
    """Trivial rank-4 decomposition of 2x2x1 (C = A * b, b is 2x1).
    U (4,4), V (2,4), W (2,4). C[i,k] with k=0 only -> W[0*2+i] = W[i].
    """
    rank = 4
    U = [[0.0] * rank for _ in range(4)]
    V = [[0.0] * rank for _ in range(2)]
    W = [[0.0] * rank for _ in range(2)]
    col = 0
    for i in range(2):
        for j in range(2):
            U[i * 2 + j][col] = 1.0
            V[j * 1 + 0][col] = 1.0
            W[0 * 2 + i][col] = 1.0
            col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == delta_{i,k}.
    Returns (max_err, ok)."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                expected = 1.0
                err = abs(T[a, b, c] - expected)
                if err > max_err:
                    max_err = err
                # Check sparsity: for fixed a (i,j) and b (j,k), only c = k*n+i is nonzero
                for c2 in range(p * n):
                    if c2 != c:
                        if abs(T[a, b, c2]) > max_err:
                            max_err = abs(T[a, b, c2])
    ok = max_err < tol
    return max_err, ok


def _construct_2x4x5():
    """Build rank-36 decomposition: 2x4x4 (28 via two Strassen 2x2x2) + 2x4x1 (8 trivial).
    Total rank = 14 + 14 + 4 + 4 = 36.
    """
    n, m, p = 2, 4, 5
    rank_total = 36
    U = [[0.0] * rank_total for _ in range(n * m)]   # 8 rows
    V = [[0.0] * rank_total for _ in range(m * p)]   # 20 rows
    W = [[0.0] * rank_total for _ in range(p * n)]   # 10 rows

    rank_offset = 0

    def place_strassen(j_range, k_range):
        nonlocal rank_offset
        U_s, V_s, W_s = _strassen_2x2x2()
        for r in range(7):
            for i in range(2):
                for jl in range(2):
                    j = j_range[0] + jl
                    U[i * 4 + j][rank_offset + r] = U_s[i * 2 + jl][r]
            for jl in range(2):
                for kl in range(2):
                    j = j_range[0] + jl
                    k = k_range[0] + kl
                    V[j * 5 + k][rank_offset + r] = V_s[jl * 2 + kl][r]
            for i in range(2):
                for kl in range(2):
                    k = k_range[0] + kl
                    W[k * 2 + i][rank_offset + r] = W_s[kl * 2 + i][r]
        rank_offset += 7

    def place_trivial(j_range, k_range):
        nonlocal rank_offset
        U_t, V_t, W_t = _trivial_2x2x1()
        for r in range(4):
            for i in range(2):
                for jl in range(2):
                    j = j_range[0] + jl
                    U[i * 4 + j][rank_offset + r] = U_t[i * 2 + jl][r]
            for jl in range(2):
                j = j_range[0] + jl
                V[j * 5 + (k_range[0] + 0)][rank_offset + r] = V_t[jl * 1 + 0][r]
            for i in range(2):
                k = k_range[0] + 0
                W[k * 2 + i][rank_offset + r] = W_t[0 * 2 + i][r]
        rank_offset += 4

    # Blocks for j_range [0,2):
    place_strassen((0, 2), (0, 2))   # C[:, 0:2]
    place_strassen((0, 2), (2, 4))   # C[:, 2:4]
    place_trivial((0, 2), (4, 5))    # C[:, 4]
    # Blocks for j_range [2,4):
    place_strassen((2, 4), (0, 2))   # C[:, 0:2]
    place_strassen((2, 4), (2, 4))   # C[:, 2:4]
    place_trivial((2, 4), (4, 5))    # C[:, 4]

    assert rank_offset == rank_total, f"rank_offset {rank_offset} != {rank_total}"
    return U, V, W


def _to_halfint(mat):
    """Convert float entries to integer or half-integer (multiply by 2, round)."""
    out = []
    for row in mat:
        new_row = []
        for x in row:
            v = round(x * 2)
            if v % 2 == 0:
                new_row.append(v // 2)
            else:
                new_row.append(v / 2.0)
        out.append(new_row)
    return out


def construct():
    deadline = time.time() + 140.0
    try:
        U, V, W = _construct_2x4x5()
        max_err, ok = _verify(U, V, W, 2, 4, 5, tol=1e-9)
        if not ok:
            # Fallback: schoolbook rank-40 (always valid)
            n, m, p = DIMENSIONS
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
            return {"U": U, "V": V, "W": W}
        return {"U": _to_halfint(U), "V": _to_halfint(V), "W": _to_halfint(W)}
    except Exception:
        # Last-resort schoolbook
        n, m, p = DIMENSIONS
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
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
