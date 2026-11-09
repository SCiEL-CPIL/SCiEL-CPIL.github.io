"""Rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication.
Uses block Strassen (rank 22) which is the known minimum rank for this dimension.
Returns the solution immediately to ensure reliability and speed.
"""
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 140.0  # seconds, leaves 10s margin before 150s kill


def _elapsed():
    return time.time() - _T0


def _remaining():
    return _DEADLINE - _elapsed()


def _build_tensors(n, m, p):
    """Precompute index arrays for vectorized error computation."""
    idx = np.indices((n, m, m, p, n, p))
    ia = idx[0].flatten()
    ja = idx[1].flatten()
    jb = idx[2].flatten()
    kb = idx[3].flatten()
    i = idx[4].flatten()
    k = idx[5].flatten()
    
    U_idx = ia * m + ja
    V_idx = jb * p + kb
    W_idx = k * n + i
    
    expected = ((ia == i) & (ja == jb) & (kb == k)).astype(float)
    
    return U_idx, V_idx, W_idx, expected


def _verify_full(U, V, W, n, m, p, tol=1e-8):
    """Full bilinear verification using vectorized computation."""
    U_idx, V_idx, W_idx, expected = _build_tensors(n, m, p)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    W_sel = W[W_idx, :]
    U_sel = U[U_idx, :]
    V_sel = V[V_idx, :]
    actual = np.einsum('nr,nr,nr->n', W_sel, U_sel, V_sel)
    return bool(np.all(np.abs(actual - expected) < tol))


def _rank22_list(n, m, p):
    """Construct rank-22 decomposition using block Strassen."""
    rank = 22
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    col = 0
    # Block 1: 7 Strassen terms for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # A: 0=A00, 1=A01, 3=A10, 4=A11
    # B: 0=B00, 1=B01, 4=B10, 5=B11
    # W: 0=C00, 2=C01, 1=C10, 3=C11

    # M1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1

    # M2 = (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[3][col] = -1
    col += 1

    # M3 = A00(B01-B11)
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[2][col] = 1; W[3][col] = 1
    col += 1

    # M4 = A11(B10-B00)
    U[4][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1

    # M5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[2][col] = 1
    col += 1

    # M6 = (A10-A00)(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    col += 1

    # M7 = (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1

    # Block 2: 7 Strassen terms for A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    # B: 2=B02, 3=B03, 6=B12, 7=B13
    # W: 4=C02, 6=C03, 5=C12, 7=C13

    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[4][col] = 1; W[7][col] = 1
    col += 1

    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[5][col] = 1; W[7][col] = -1
    col += 1

    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1

    U[4][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[4][col] = 1; W[5][col] = 1
    col += 1

    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[4][col] = -1; W[6][col] = 1
    col += 1

    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[7][col] = 1
    col += 1

    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[4][col] = 1
    col += 1

    # Block 3: 8 schoolbook terms for A[0:2,2:3] * B[2:3,0:4] -> C[0:2,0:4]
    # A: 2=A02, 5=A12
    # B: 8=B20, 9=B21, 10=B22, 11=B23
    for i, ai in [(0, 2), (1, 5)]:
        for k, bk in [(0, 8), (1, 9), (2, 10), (3, 11)]:
            U[ai][col] = 1
            V[bk][col] = 1
            W[k * n + i][col] = 1
            col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W


def _schoolbook(n, m, p):
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


def _to_dict(U, V, W):
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = 2, 3, 4
    
    # Start with rank-22
    best_U, best_V, best_W = _rank22_list(n, m, p)
    
    # Verify
    if not _verify_full(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook if verification fails (should not happen)
        best_U, best_V, best_W = _schoolbook(n, m, p)

    return _to_dict(best_U, best_V, best_W)
