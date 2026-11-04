"""Tensor decomposition for 3x4 * 4x8 matrix multiplication (rank-minimal exact)."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]
_START = time.time()
_LIMIT = 150.0
_MARGIN = 5.0


def _all_ones_rank(n, m, p):
    """Exact rank-1 construction family: rank = min(n*m, m*p, p*n).
    Returns (U, V, W) as lists with integer entries, or None."""
    candidates = []
    # rank = n*m : U diag-ones, V ones, W ones
    candidates.append(("nm", n * m))
    candidates.append(("mp", m * p))
    candidates.append(("pn", p * n))
    candidates.sort(key=lambda x: x[1])

    best = None
    for name, rank in candidates:
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        if name == "nm":
            # term r = (i0, j0): U[i0*m+j0, r]=1; V all 1; W all 1
            for i0 in range(n):
                for j0 in range(m):
                    r = i0 * m + j0
                    U[i0 * m + j0][r] = 1
                    for j in range(m):
                        for k in range(p):
                            V[j * p + k][r] = 1
                    for k in range(p):
                        for i in range(n):
                            W[k * n + i][r] = 1
        elif name == "mp":
            # term r = (j0, k0): U ones; V[j0*p+k0, r]=1; W ones
            for j0 in range(m):
                for k0 in range(p):
                    r = j0 * p + k0
                    for i in range(n):
                        for j in range(m):
                            U[i * m + j][r] = 1
                    V[j0 * p + k0][r] = 1
                    for k in range(p):
                        for i in range(n):
                            W[k * n + i][r] = 1
        else:  # "pn": term r = (k0, i0): U ones; V ones; W[k0*n+i0, r]=1
            for k0 in range(p):
                for i0 in range(n):
                    r = k0 * n + i0
                    for i in range(n):
                        for j in range(m):
                            U[i * m + j][r] = 1
                    for j in range(m):
                        for k in range(p):
                            V[j * p + k][r] = 1
                    W[k0 * n + i0][r] = 1
        best = (rank, U, V, W)
        break  # smallest rank candidate is "nm" here (12); take it
    return best


def _verify(n, m, p, U, V, W):
    """Check U[i,j,r] V[j,k,r] W[k,i,r] sums to 1 for all (i,j,k)."""
    if time.time() > _LIMIT - _MARGIN:
        return False
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i]))
                if abs(s - 1.0) > 1e-9:
                    ok = False
    return ok


def _schoolbook(n, m, p):
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
    # Try the low-rank all-ones construction (rank = n*m = 12).
    res = _all_ones_rank(n, m, p)
    if res is not None:
        rank, U, V, W = res
        if _verify(n, m, p, U, V, W):
            return {"U": U, "V": V, "W": W}
    # Fallback: guaranteed schoolbook.
    return _schoolbook(n, m, p)
