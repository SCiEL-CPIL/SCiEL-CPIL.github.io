import numpy as np
import time

DIMENSIONS = [4, 5, 5]
_START = time.time()
_TIMEOUT = 150.0
_MARGIN = 10.0

def _time_left():
    return _TIMEOUT - _MARGIN - (time.time() - _START)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-9):
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    T = U @ np.diag(np.ones(r)) @ V.T  # (nm, mp)
    for i in range(n):
        for k in range(p):
            for j in range(m):
                expected = 1.0 if (i * m + j) * (p * n) + (k * n + i) < 0 else 0.0
            # build full expected tensor
    full = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                full[i * m + j, j * p + k, k * n + i] = 1.0
    # T contracted with W: T[i,j,k] should equal sum_t U[i,t] V[j,t] W[k,t]
    got = np.einsum('it,jt,kt->ijk', U, V, W)
    return bool(np.allclose(got, full, atol=tol, rtol=0))

def _winograd_1x5x5():
    # Return (U,V,W) each shape (5, 15) for 1x5x5 = cyclic convolution x^4+x^3+x^2+x+1
    phi = [1, 1, 1, 1, 1]
    phi_rev = phi[::-1]
    def circ(a, b):
        out = [0] * 5
        for i in range(5):
            for j in range(5):
                out[i] += a[i] * b[(i - j) % 5]
        return out
    # factor A: x^4+x^3+x^2+x+1 = (x^2 + alpha x + 1)(x^2 + beta x + 1)
    # alpha + beta = -1, alpha*beta + 2 = 1 -> alpha*beta = -1
    alpha = (1 - 5 ** 0.5) / 2.0
    beta = (1 + 5 ** 0.5) / 2.0
    A1 = [1.0, alpha, 1.0]
    A2 = [1.0, beta, 1.0]
    B1 = [1.0, 0.0]
    B2 = [1.0, 0.0]
    C1 = [1.0, alpha, 1.0]
    C2 = [1.0, beta, 1.0]
    A0 = [1.0, 1.0, 1.0, 1.0, 1.0]
    B0 = [1.0]
    C0 = phi_rev
    factors = [(A0, B0, C0), (A1, B1, C1), (A2, B2, C2)]
    U = []
    V = []
    W = []
    for (A, B, C) in factors:
        La = len(A)
        Lb = len(B)
        Lc = len(C)
        for i in range(5):
            for j in range(5):
                u = 1.0
                v = 1.0
                w = 1.0
                for t in range(La):
                    if t <= i:
                        u *= A[t]
                for t in range(Lb):
                    if t <= j:
                        v *= B[t]
                for t in range(Lc):
                    if t <= (4 - j) % 5:
                        w *= C[t]
                U.append(u)
                V.append(v)
                W.append(w)
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # verify 1x5x5
    got = np.einsum('it,jt,kt->ijk', U, V, W)
    expected = np.zeros((5, 5, 5))
    for i in range(5):
        for j in range(5):
            for k in range(5):
                if i == (j - k) % 5:
                    expected[i, j, k] = 1.0
    if not np.allclose(got, expected, atol=1e-8, rtol=0):
        return None
    return U, V, W

def _construct_winograd():
    n, m, p = DIMENSIONS
    base = _winograd_1x5x5()
    if base is None:
        return None
    Ub, Vb, Wb = base
    rank = n * 15
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    for i in range(n):
        off = i * 15
        for j in range(m):
            U[i * m + j, off:off + 15] = Ub
        for k in range(p):
            W[k * n + i, off:off + 15] = Wb
    for j in range(m):
        for k in range(p):
            V[j * p + k, :] = Vb
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    if _time_left() <= 1:
        return _schoolbook(n, m, p)
    try:
        cand = _construct_winograd()
        if cand is not None and _verify(cand["U"], cand["V"], cand["W"], n, m, p):
            return cand
    except Exception:
        pass
    return _schoolbook(n, m, p)
