"""Rank-minimal exact tensor decomposition for 2x4 @ 4x6 matrix multiplication.

Uses block-partition + Strassen (rank 42) as baseline, then attempts
border-rank / ALS search for rank 40 or 41 with half-integer coefficients.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]


def _strassen_coefficients():
    products = [
        {'U': [(0,0,1), (1,1,1)], 'V': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)]},
        {'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)], 'W': [(1,0,1), (1,1,-1)]},
        {'U': [(0,0,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(0,1,1), (1,1,1)]},
        {'U': [(1,1,1)], 'V': [(1,0,1), (0,0,-1)], 'W': [(0,0,1), (1,0,1)]},
        {'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,-1), (0,1,1)]},
        {'U': [(1,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(1,1,1)]},
        {'U': [(0,1,1), (1,1,-1)], 'V': [(1,0,1), (1,1,1)], 'W': [(0,0,1)]},
    ]
    return products


def _build_rank42_block():
    n, m, p = 2, 4, 6
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    strassen = _strassen_coefficients()
    col = 0
    for r in range(3):
        for sub in range(2):
            a_offset = sub * 2
            b_row_offset = sub * 2
            for prod_idx in range(7):
                prod = strassen[prod_idx]
                for (i, j, coeff) in prod['U']:
                    U[i * m + (a_offset + j), col] += coeff
                for (j, kb, coeff) in prod['V']:
                    V[(b_row_offset + j) * p + (2 * r + kb), col] += coeff
                for (i, kb, coeff) in prod['W']:
                    W[(2 * r + kb) * n + i, col] += coeff
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for s in range(rank):
                    val += U[i * m + j, s] * V[j * p + k, s] * W[k * n + i, s]
                if abs(val - 1.0) > tol:
                    return False, (i, j, k, val)
    return True, None


def _to_output(mat):
    return [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in mat]


def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                U[ii * m + jj][col] = 1
                V[jj * p + kk][col] = 1
                W[kk * n + ii][col] = 1
                col += 1
    return U, V, W


def _als_decompose(n, m, p, r, time_budget=45.0, n_trials=50, n_iters=40):
    """Alternating least squares for rank-r tensor decomposition."""
    start = time.time()
    n_dim = n * m
    m_dim = m * p
    p_dim = p * n
    np.random.seed(12345)
    best_res = float('inf')
    best = None

    for trial in range(n_trials):
        if time.time() - start > time_budget:
            break
        U = np.random.randn(n_dim, r) * 0.3
        V = np.random.randn(m_dim, r) * 0.3
        W = np.random.randn(p_dim, r) * 0.3

        for it in range(n_iters):
            if time.time() - start > time_budget:
                break
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                    x, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[i*m+j, :] = x
            # Update V
            for j in range(m):
                for k in range(p):
                    B = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                    x, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                    V[j*p+k, :] = x
            # Update W
            for k in range(p):
                for i in range(n):
                    C = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                    x, _, _, _ = np.linalg.lstsq(C, np.ones(m), rcond=None)
                    W[k*n+i, :] = x

        # Residual
        res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    v = sum(U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] for s in range(r))
                    res += (v - 1.0)**2
        if res < best_res:
            best_res = res
            best = (U.copy(), V.copy(), W.copy())
        if best_res < 1e-10:
            break

    if best is not None and best_res < 1e-8:
        return best[0], best[1], best[2], best_res
    return None, None, None, best_res


def _round_halfint(U, V, W):
    return np.round(U*2)/2, np.round(V*2)/2, np.round(W*2)/2


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Try rank 40
    if time.time() < deadline - 50:
        U40, V40, W40, res = _als_decompose(n, m, p, 40, time_budget=45.0)
        if U40 is not None:
            U40r, V40r, W40r = _round_halfint(U40, V40, W40)
            valid, _ = _verify(U40r, V40r, W40r, n, m, p)
            if valid:
                return {"U": _to_output(U40r), "V": _to_output(V40r), "W": _to_output(W40r)}

    # Try rank 41
    if time.time() < deadline - 40:
        U41, V41, W41, res = _als_decompose(n, m, p, 41, time_budget=35.0)
        if U41 is not None:
            U41r, V41r, W41r = _round_halfint(U41, V41, W41)
            valid, _ = _verify(U41r, V41r, W41r, n, m, p)
            if valid:
                return {"U": _to_output(U41r), "V": _to_output(V41r), "W": _to_output(W41r)}

    # Fallback: rank 42
    U, V, W = _build_rank42_block()
    valid, _ = _verify(U, V, W, n, m, p)
    if valid:
        return {"U": _to_output(U), "V": _to_output(V), "W": _to_output(W)}

    U_fb, V_fb, W_fb = _build_schoolbook(n, m, p)
    return {"U": U_fb, "V": V_fb, "W": W_fb}
