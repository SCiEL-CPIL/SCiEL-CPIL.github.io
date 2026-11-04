import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return bool(np.allclose(T, 1.0, atol=tol, rtol=0.0))

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _snap_half(x):
    # snap to nearest half-integer
    return np.round(2.0 * x) / 2.0

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 112.0

    # Baseline
    U0, V0, W0 = _schoolbook(n, m, p)
    best = (U0, V0, W0, n * m * p)
    assert _verify(U0, V0, W0, n, m, p)

    rng = np.random.default_rng(20240611)
    found = None

    # Search ranks below schoolbook.
    for r in range(n * m * p - 1, max(50, n * m * p - 40), -1):
        if time.time() > deadline - 15.0:
            break
        for trial in range(400):
            if time.time() > deadline - 15.0:
                break
            # Random small-integer A, B
            A = rng.integers(-1, 2, size=(n, m, r)).astype(float)
            B = rng.integers(-1, 2, size=(m, p, r)).astype(float)
            M = np.einsum('ijc,jkc->ikc', A, B)  # (n,p,r)
            # Linear system for C: for each (i,k): sum_c M[i,k,c] C[k,i,c] = 1
            A_mat = np.zeros((n * p, p * n * r))
            for i in range(n):
                for k in range(p):
                    for c in range(r):
                        A_mat[i * p + k, (k * n + i) * r + c] = M[i, k, c]
            rhs = np.ones(n * p)
            try:
                x, *_ = np.linalg.lstsq(A_mat, rhs, rcond=None)
            except Exception:
                continue
            C = x.reshape(p, n, r)
            # Snap and verify
            for snap in (C, _snap_half(C)):
                U = A.reshape(n * m, r)
                V = B.reshape(m * p, r)
                W = snap.reshape(p * n, r)
                if _verify(U, V, W, n, m, p):
                    found = (U, V, W, r)
                    break
            if found is not None:
                break
        if found is not None:
            break

    if found is not None:
        U, V, W, r = found
        return {"U": U, "V": V, "W": W}
    else:
        U, V, W, r = best
        return {"U": U, "V": V, "W": W}
