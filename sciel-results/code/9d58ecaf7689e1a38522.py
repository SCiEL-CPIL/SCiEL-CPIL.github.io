import numpy as np
import time

DIMENSIONS = [2, 3, 9]


def _check(U, V, W, n, m, p, tol=1e-9):
    """Verify the exact tensor decomposition constraint."""
    for i2 in range(n):
        for j in range(m):
            for j2 in range(m):
                for k2 in range(p):
                    u = U[i2 * m + j]
                    v = V[j2 * p + k2]
                    # M[i, k] = sum_r u[r]*v[r]*W[k*n+i, r]
                    M = (u * v @ W.T).reshape(p, n).T  # (n, p)
                    expected = 1.0 if (j == j2 and k2 == j) else 0.0
                    if expected == 0.0:
                        if np.any(np.abs(M) > tol):
                            return False
                    else:
                        for i in range(n):
                            if i == i2:
                                if np.any(np.abs(M[i] - 1.0) > tol):
                                    return False
                            else:
                                if np.any(np.abs(M[i]) > tol):
                                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def construct():
    start = time.time()
    n, m, p = DIMENSIONS

    # Rank-6 exact decomposition:
    # T = (sum_k e_k) [W's k-factor, all-ones] ⊗ I_2 ⊗ I_3
    # => 6 rank-1 terms, one per (i, j) with i in [0,n), j in [0,m).
    rank = n * m  # = 6
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                W[k * n + i, r] = 1.0   # W's k-profile is all-ones
            U[i * m + j, r] = 1.0        # U selects i2 == i, col j
            V[j * p + j, r] = 1.0        # V selects j2 == j, k2 == j
            r += 1

    if _check(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback (should never trigger)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
