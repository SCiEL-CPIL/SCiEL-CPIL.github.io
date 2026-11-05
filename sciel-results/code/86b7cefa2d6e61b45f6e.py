"""
Exact rank-78 tensor decomposition for M_{3,4,7} matrix multiplication.
Constructed as block sum: 6 Strassen 2x2x2 blocks + 36 schoolbook residual terms.
"""
import time
import numpy as np

_DIMS = (3, 4, 7)


def construct():
    t0 = time.monotonic()
    T_LIM = 105.0

    n, m, p = _DIMS
    R = 78

    U = np.zeros((n * m, R), dtype=np.float64)
    V = np.zeros((m * p, R), dtype=np.float64)
    W = np.zeros((p * n, R), dtype=np.float64)

    # Precompute all 42 Strassen terms as (col_offset, [(U_idx, val), ...], [(V_idx, val), ...], [(W_idx, val), ...])
    _STRASSEN = _build_strassen_terms(n, m, p)
    _SCHOOLBOOK = _build_residual_terms(n, m, p)

    for c, u_specs, v_specs, w_specs in _STRASSEN:
        for ui, uval in u_specs:
            U[ui, c] = uval
        for vi, vval in v_specs:
            V[vi, c] = vval
        for wi, wval in w_specs:
            W[wi, c] = wval

    for c, u_idx, v_idx, w_idx in _SCHOOLBOOK:
        U[u_idx, c] = 1.0
        V[v_idx, c] = 1.0
        W[w_idx, c] = 1.0

    if time.monotonic() - t0 > T_LIM:
        return _fallback(n, m, p)

    if _check(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return _fallback(n, m, p)


def _build_strassen_terms(n, m, p):
    """Generate the 42 rank-1 terms from six 2x2x2 Strassen decompositions."""
    blocks = [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 2, 0), (0, 2, 2), (0, 2, 4)]
    terms = []
    col = 0
    for (i0, j0, k0) in blocks:
        i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
        terms.append((col,
                      [(i0 * m + j0, 1), (i1 * m + j1, 1)],
                      [(j0 * p + k0, 1), (j1 * p + k1, 1)],
                      [(k0 * n + i0, 1), (k1 * n + i1, 1)]))
        col += 1
        terms.append((col,
                      [(i1 * m + j0, 1), (i1 * m + j1, 1)],
                      [(j0 * p + k0, 1)],
                      [(k0 * n + i1, 1), (k1 * n + i1, -1)]))
        col += 1
        terms.append((col,
                      [(i0 * m + j0, 1)],
                      [(j0 * p + k1, 1), (j1 * p + k1, -1)],
                      [(k1 * n + i0, 1), (k1 * n + i1, 1)]))
        col += 1
        terms.append((col,
                      [(i1 * m + j1, 1)],
                      [(j1 * p + k0, 1), (j0 * p + k0, -1)],
                      [(k0 * n + i0, 1), (k0 * n + i1, 1)]))
        col += 1
        terms.append((col,
                      [(i0 * m + j0, 1), (i0 * m + j1, 1)],
                      [(j1 * p + k1, 1)],
                      [(k0 * n + i0, -1), (k1 * n + i0, 1)]))
        col += 1
        terms.append((col,
                      [(i1 * m + j0, 1), (i0 * m + j0, -1)],
                      [(j0 * p + k0, 1), (j0 * p + k1, 1)],
                      [(k1 * n + i1, 1)]))
        col += 1
        terms.append((col,
                      [(i0 * m + j1, 1), (i1 * m + j1, -1)],
                      [(j1 * p + k0, 1), (j1 * p + k1, 1)],
                      [(k0 * n + i0, 1)]))
        col += 1
    return terms


def _build_residual_terms(n, m, p):
    """Generate schoolbook terms for entries not covered by Strassen blocks."""
    covered = set()
    for (i0, j0, k0) in [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 2, 0), (0, 2, 2), (0, 2, 4)]:
        for ii in range(i0, i0 + 2):
            for jj in range(j0, j0 + 2):
                for kk in range(k0, k0 + 2):
                    covered.add((ii, jj, kk))

    terms = []
    col = 42
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    terms.append((col, i * m + j, j * p + k, k * n + i))
                    col += 1
    return terms


def _check(U, V, W, n, m, p):
    """Vectorized verification: check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
    Uf = U
    Vf = V
    Wf = W
    R = Uf.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(Uf[i * m + j], Vf[j * p + k] * Wf[k * n + i])
                if abs(val - 1.0) > 1e-7:
                    return False
    return True


def _fallback(n, m, p):
    R = n * m * p
    U = np.zeros((n * m, R), dtype=np.float64)
    V = np.zeros((m * p, R), dtype=np.float64)
    W = np.zeros((p * n, R), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
