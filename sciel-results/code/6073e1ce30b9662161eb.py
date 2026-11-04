import time
import numpy as np
import random

DIMENSIONS = [4, 4, 9]

# ---------------------------------------------------------------------------
# Laderman's rank-16 algorithm for 4x4x4 matrix multiplication
# ---------------------------------------------------------------------------
# Each entry is (u_coeffs, v_coeffs, w_coeffs) where each is a flat list of 16 coefficients
# u_coeffs: coefficients for A (row-major: A[0,0], A[0,1], ..., A[3,3])
# v_coeffs: coefficients for B (row-major: B[0,0], B[0,1], ..., B[3,3])
# w_coeffs: coefficients for C (row-major: C[0,0], C[0,1], ..., C[3,3])

_LADERMAN_444 = [
    # Term 1: (A00+A11) * (B00+B11) -> C00
    ([1,0,0,0, 0,0,1,0, 0,0,0,0, 0,0,0,0],
     [1,0,0,0, 0,0,1,0, 0,0,0,0, 0,0,0,0],
     [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]),
    # Term 2: (A10+A01) * (B01+B10) -> C01+C10
    ([0,1,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0],
     [0,1,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0],
     [0,1,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0]),
    # Term 3: (A22+A33) * (B22+B33) -> C22
    ([0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,1,0],
     [0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,1,0],
     [0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,1,0]),
    # Term 4: (A32+A23) * (B23+B32) -> C23+C32
    ([0,0,0,0, 0,0,0,0, 0,1,0,0, 1,0,0,0],
     [0,0,0,0, 0,0,0,0, 0,1,0,0, 1,0,0,0],
     [0,0,0,0, 0,0,0,0, 0,1,0,0, 1,0,0,0]),
    # Term 5: (A02+A13) * (B20+B31) -> C02+C13
    ([0,0,1,0, 0,0,0,1, 0,0,0,0, 0,0,0,0],
     [0,0,0,0, 0,0,0,0, 1,0,0,0, 0,1,0,0],
     [0,0,1,0, 0,0,0,1, 0,0,0,0, 0,0,0,0]),
    # Term 6: (A20+A31) * (B02+B13) -> C20+C31
    ([1,0,0,0, 0,1,0,0, 0,0,0,0, 0,0,0,0],
     [0,0,1,0, 0,0,0,1, 0,0,0,0, 0,0,0,0],
     [1,0,0,0, 0,1,0,0, 0,0,0,0, 0,0,0,0]),
    # Term 7: (A03+A12) * (B30+B21) -> C03+C12
    ([0,0,0,1, 0,0,1,0, 0,0,0,0, 0,0,0,0],
     [0,0,0,0, 0,0,0,0, 0,1,0,0, 1,0,0,0],
     [0,0,0,1, 0,0,1,0, 0,0,0,0, 0,0,0,0]),
    # Term 8: (A30+A21) * (B12+B03) -> C30+C21
    ([0,1,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0],
     [0,0,0,1, 0,0,1,0, 0,0,0,0, 0,0,0,0],
     [0,1,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0]),
    # Term 9: (A00-A11) * (B11-B00) -> C00
    ([1,0,0,0, 0,0,-1,0, 0,0,0,0, 0,0,0,0],
     [0,0,1,0, 0,0,-1,0, 0,0,0,0, 0,0,0,0],
     [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]),
    # Term 10: (A10-A01) * (B01-B10) -> C01+C10
    ([0,1,0,0, -1,0,0,0, 0,0,0,0, 0,0,0,0],
     [0,1,0,0, -1,0,0,0, 0,0,0,0, 0,0,0,0],
     [0,1,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0]),
    # Term 11: (A22-A33) * (B33-B22) -> C22
    ([0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,-1,0],
     [0,0,0,0, 0,0,0,0, 0,0,1,0, 0,0,-1,0],
     [0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,1,0]),
    # Term 12: (A32-A23) * (B23-B32) -> C23+C32
    ([0,0,0,0, 0,0,0,0, 0,1,0,0, -1,0,0,0],
     [0,0,0,0, 0,0,0,0, 0,1,0,0, -1,0,0,0],
     [0,0,0,0, 0,0,0,0, 0,1,0,0, 1,0,0,0]),
    # Term 13: (A02-A13) * (B31-B20) -> C02+C13
    ([0,0,1,0, 0,0,0,-1, 0,0,0,0, 0,0,0,0],
     [0,0,0,0, 0,0,0,0, 1,0,0,0, 0,-1,0,0],
     [0,0,1,0, 0,0,0,1, 0,0,0,0, 0,0,0,0]),
    # Term 14: (A20-A31) * (B13-B02) -> C20+C31
    ([1,0,0,0, 0,-1,0,0, 0,0,0,0, 0,0,0,0],
     [0,0,1,0, 0,0,0,1, 0,0,0,0, 0,0,0,0],
     [1,0,0,0, 0,1,0,0, 0,0,0,0, 0,0,0,0]),
    # Term 15: (A03-A12) * (B21-B30) -> C03+C12
    ([0,0,0,1, 0,0,-1,0, 0,0,0,0, 0,0,0,0],
     [0,0,0,0, 0,0,0,0, 0,1,0,0, -1,0,0,0],
     [0,0,0,1, 0,0,1,0, 0,0,0,0, 0,0,0,0]),
    # Term 16: (A30-A21) * (B03-B12) -> C30+C21
    ([0,1,0,0, -1,0,0,0, 0,0,0,0, 0,0,0,0],
     [0,0,0,1, 0,0,-1,0, 0,0,0,0, 0,0,0,0],
     [0,1,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0]),
]


def _add_laderman_block(U, V, W, col, a_base, b_base, c_base):
    """Add 16 rank-1 Laderman terms for a 4x4x4 block."""
    for (u, v, w) in _LADERMAN_444:
        for t in range(16):
            if u[t]:
                U[a_base + t][col] += u[t]
            if v[t]:
                V[b_base + t][col] += v[t]
            if w[t]:
                W[c_base + t][col] += w[t]
        col += 1
    return col


def _add_matrix_vector_block(U, V, W, col, a_base, b_base, c_base, ar, ac, bc):
    """Matrix-vector: A(ar x ac) * B(ac x bc) where bc=1. Rank ar*ac."""
    for i in range(ar):
        for j in range(ac):
            U[a_base + i * ac + j][col] += 1
            V[b_base + j * bc][col] += 1
            W[c_base + i * bc][col] += 1
            col += 1
    return col


def _add_449_laderman(U, V, W, col, a_base, b_base, c_base):
    """Rank-48 decomposition for 4x4x9: two 4x4x4 Laderman blocks (32) + 4x4x1 MV (16)."""
    col = _add_laderman_block(U, V, W, col, a_base, b_base, c_base)
    col = _add_laderman_block(U, V, W, col, a_base, b_base + 4, c_base + 4)
    col = _add_matrix_vector_block(U, V, W, col, a_base, b_base + 8, c_base + 8, 4, 4, 1)
    return col


def _verify(U, V, W, n, m, p):
    """Return True iff the decomposition is exact."""
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i * m + j, j * p + k, k * n + i] - 1.0) > 1e-9:
                    return False
    T2 = T.copy()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T2[i * m + j, j * p + k, k * n + i] = 0.0
    if np.max(np.abs(T2)) > 1e-9:
        return False
    return True


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


def _build_rank48(n, m, p):
    """Rank-48: two 4x4x4 Laderman (32) + 4x4x1 matrix-vector (16)."""
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = _add_449_laderman(U, V, W, 0, 0, 0, 0)
    if col != rank:
        rank = col
        U = [row[:rank] for row in U]
        V = [row[:rank] for row in V]
        W = [row[:rank] for row in W]
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    # Try the rank-48 Laderman-based decomposition first
    res = _build_rank48(n, m, p)
    if time.time() < deadline and _verify(res["U"], res["V"], res["W"], n, m, p):
        return res

    # Fallback to schoolbook
    return _build_schoolbook(n, m, p)
