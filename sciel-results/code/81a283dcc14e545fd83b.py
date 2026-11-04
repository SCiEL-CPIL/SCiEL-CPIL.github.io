"""Search for rank-minimal exact tensor decomposition of 3x3 * 3x5 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(U[i*m+j], V[j*p+k] * W[k*n+i]))
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol, max_err

def _try_rank(r, n, m, p, time_limit, rng):
    deadline = time.time() + time_limit
    while time.time() < deadline:
        U = rng.standard_normal((n*m, r))
        V = rng.standard_normal((m*p, r))
        W = np.zeros((p*n, r))
        success = True
        for i in range(n):
            for k in range(p):
                M = np.array([[U[i*m+j, rp] * V[j*p+k, rp] for rp in range(r)] for j in range(m)])
                b = np.ones(m)
                try:
                    x, res, _, _ = np.linalg.lstsq(M, b, rcond=None)
                    if res > 1e-8:
                        success = False
                        break
                    W[k*n+i, :] = x
                except:
                    success = False
                    break
            if not success:
                break
        if not success:
            continue
        ok, err = _verify(U, V, W, n, m, p, tol=1e-8)
        if ok:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 135
    rng = np.random.RandomState(42)
    
    # Schoolbook
    rs = n*m*p
    Us = [[0]*rs for _ in range(n*m)]
    Vs = [[0]*rs for _ in range(m*p)]
    Ws = [[0]*rs for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Us[i*m+j][c] = 1
                Vs[j*p+k][c] = 1
                Ws[k*n+i][c] = 1
                c += 1
    best = {"U": Us, "V": Vs, "W": Ws}
    
    for rank in range(3, 45):
        if time.time() > deadline - 5:
            break
        tpr = min(20, (deadline - time.time()) / max(1, 45-rank))
        res = _try_rank(rank, n, m, p, tpr, rng)
        if res is not None:
            best = res
            break
    return best
