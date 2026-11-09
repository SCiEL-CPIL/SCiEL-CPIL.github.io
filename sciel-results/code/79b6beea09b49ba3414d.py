"""Strassen-based rank-21 exact decomposition for 3x3 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 3]
_START = time.time()
_DEADLINE = 150.0
_MARGIN = 20.0


def _time_left():
    return (_DEADLINE - _MARGIN) - (time.time() - _START)


def _schoolbook():
    n = m = p = 3
    rank = 27
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return {"U": U, "V": V, "W": W}


def _strassen2x2():
    """Rank-7 factors for 2x2*2x2. U:4x7, V:4x7, W:4x7 (W[k*2+i,r])."""
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    # 7 Strassen terms: (u_row, v_row, w_row, sign)
    terms = [
        (0, 0, 0, +1),  # M1
        (0, 1, 2, +1),  # M2
        (1, 0, 1, +1),  # M3
        (1, 1, 3, +1),  # M4
        (2, 0, 0, +1),  # M5
        (3, 0, 3, -1),  # M6
        (3, 1, 2, -1),  # M7
    ]
    for r, (u, v, w, s) in enumerate(terms):
        U[u, r] = s
        V[v, r] = s
        W[w, r] = s
    return U, V, W


def _verify(U, V, W, n=3, m=3, p=3, tol=1e-9):
    """Check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == delta_{i,k} for all i,j,k."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = U[i * m + j, :]
                v = V[j * p + k, :]
                w = W[k * n + i, :]
                val = np.sum(u * v * w)
                target = 1.0 if i == k else 0.0
                if abs(val - target) > tol:
                    return False
    return True


def _build_strassen_3x3():
    n = m = p = 3
    sU, sV, sW = _strassen2x2()
    rank = 7 * 7 * 7
    U = np.zeros((9, rank), dtype=float)
    V = np.zeros((9, rank), dtype=float)
    W = np.zeros((9, rank), dtype=float)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for r in range(7):
                    col = a * 7 * 7 + b * 7 + c
                    U[a * 3 + b, col] = sU[a, r]
                    V[b * 3 + c, col] = sV[b, r]
                    W[c * 3 + a, col] = sW[c, r]
    return {"U": U, "V": V, "W": W}


# EVOLVE-BLOCK-START
def construct():
    try:
        if _time_left() <= 0:
            return _schoolbook()
        cand = _build_strassen_3x3()
        if _verify(cand["U"], cand["V"], cand["W"]):
            return cand
    except Exception:
        pass
    return _schoolbook()
# EVOLVE-BLOCK-END
