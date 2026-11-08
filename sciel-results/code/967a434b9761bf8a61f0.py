import time
import numpy as np

_DIMS = (3, 4, 7)
_RANK = 78

_BLOCKS = [
    (0, 1, 0, 1, 0, 1),
    (0, 1, 0, 1, 2, 3),
    (0, 1, 0, 1, 4, 5),
    (0, 1, 2, 3, 0, 1),
    (0, 1, 2, 3, 2, 3),
    (0, 1, 2, 3, 4, 5),
]


def _strassen_pattern_rows(i0, i1, j0, j1, m):
    """Rows (i*m+j) touched per Strassen factor index, with signs in U."""
    return [
        [(i0, j0, 1)],
        [(i1, j0, 1), (i1, j1, 1)],
        [(i0, j0, 1)],
        [(i1, j1, 1)],
        [(i0, j0, 1), (i0, j1, 1)],
        [(i1, j0, 1), (i0, j0, -1)],
        [(i0, j1, 1), (i1, j1, -1)],
    ]


def _strassen_pattern_cols_v(j0, j1, k0, k1, p):
    """Columns (j*p+k) touched per Strassen factor index, with signs in V."""
    return [
        [(j0, k0, 1)],
        [(j0, k0, 1)],
        [(j0, k1, 1), (j1, k1, -1)],
        [(j1, k0, 1), (j0, k0, -1)],
        [(j1, k1, 1)],
        [(j0, k0, 1), (j0, k1, 1)],
        [(j1, k0, 1), (j1, k1, 1)],
    ]


def _strassen_pattern_rows_w(k0, k1, i0, i1, n):
    """Rows (k*n+i) touched per Strassen factor index, with signs in W."""
    return [
        [(k0, i0, 1)],
        [(k0, i1, 1), (k1, i1, -1)],
        [(k1, i0, 1), (k1, i1, 1)],
        [(k0, i0, 1), (k0, i1, 1)],
        [(k0, i0, -1), (k1, i0, 1)],
        [(k1, i1, 1)],
        [(k0, i0, 1)],
    ]


def _verify(U, V, W, n, m, p):
    """Check U * V == tensor == W under the trace convention."""
    rank = U.shape[1]
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1

    # Sum over rank: U[:,r] * V[:,r] * W[:,r]
    # Use einsum: sum_r U[a,r] * V[b,r] * W[c,r]
    T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    return np.array_equal(T_recon, T)


def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    n, m, p = _DIMS
    rank = _RANK

    rows_u = n * m
    rows_v = m * p
    rows_w = p * n

    U = np.zeros((rows_u, rank), dtype=np.int64)
    V = np.zeros((rows_v, rank), dtype=np.int64)
    W = np.zeros((rows_w, rank), dtype=np.int64)

    col = 0

    # Place Strassen blocks
    for (i0, i1, j0, j1, k0, k1) in _BLOCKS:
        u_pat = _strassen_pattern_rows(i0, i1, j0, j1, m)
        v_pat = _strassen_pattern_cols_v(j0, j1, k0, k1, p)
        w_pat = _strassen_pattern_rows_w(k0, k1, i0, i1, n)

        for r in range(7):
            for (ii, jj, sign) in u_pat[r]:
                U[ii * m + jj, col + r] += sign
            for (jj, kk, sign) in v_pat[r]:
                V[jj * p + kk, col + r] += sign
            for (kk, ii, sign) in w_pat[r]:
                W[kk * n + ii, col + r] += sign
        col += 7

        if time.time() > deadline:
            break

    # Determine covered triples
    covered = set()
    for (i0, i1, j0, j1, k0, k1) in _BLOCKS:
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covered.add((ii, jj, kk))

    # Fill remaining triples with schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                if col >= rank:
                    raise RuntimeError("Rank overflow during construction")
                if time.time() > deadline:
                    raise RuntimeError("Deadline exceeded during construction")
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    if col != rank:
        raise RuntimeError(f"Expected rank {rank}, got {col}")

    # Verify correctness before returning
    if not _verify(U, V, W, n, m, p):
        raise RuntimeError("Verification failed: decomposition does not match tensor")

    # Convert to list-of-lists interface expected by the task
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist(),
    }
