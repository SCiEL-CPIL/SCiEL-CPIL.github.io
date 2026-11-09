import numpy as np
import time

DIMENSIONS = [3, 3, 6]
START = time.time()
DEADLINE = 150.0 - 5.0


def _ok():
    return time.time() < DEADLINE


def _verify_3x3x2(u, v, w, iters=12, seed=999):
    # u: (rank, 3, 3), v: (rank, 3, 2), w: (rank, 3, 2)
    # Check: sum_t (u[t] * A) * (v[t] * B) * w[t] == A @ B
    rng = np.random.default_rng(seed)
    for _ in range(iters):
        a = rng.integers(-5, 6, size=(3, 3))
        b = rng.integers(-5, 6, size=(3, 2))
        c_true = a @ b
        c = np.zeros((3, 2))
        for t in range(u.shape[0]):
            alpha = float(np.sum(u[t] * a))
            beta = float(np.sum(v[t] * b))
            c += alpha * beta * w[t]
        if not np.allclose(c, c_true, atol=1e-9, rtol=0.0):
            return False
    return True


def _find_3x3x2(rank, seed, trials=1000):
    rng = np.random.default_rng(seed)
    vals = np.array([-2, -1, 0, 1, 2])
    for _ in range(trials):
        if not _ok():
            return None
        u = rng.choice(vals, size=(rank, 3, 3))
        v = rng.choice(vals, size=(rank, 3, 2))
        rows = []
        for i in range(3):
            for k in range(3):
                for j in range(2):
                    row = np.zeros(rank * 6)
                    for t in range(rank):
                        row[t * 6 + i * 2 + j] = u[t, i, k] * v[t, k, j]
                    rows.append(row)
        A = np.array(rows)
        try:
            w, *_ = np.linalg.lstsq(A, np.ones(18), rcond=None)
        except np.linalg.LinAlgError:
            continue
        w = w.reshape(rank, 3, 2)
        wr = np.round(w * 2) / 2
        if _verify_3x3x2(u, v, wr):
            return u.astype(float), v.astype(float), wr
    return None


def _build_3x6_from_3x3x2(r, d0, d1, d2):
    n, m, p = 3, 3, 6
    R = 3 * r
    U = np.zeros((9, R))
    V = np.zeros((18, R))
    W = np.zeros((18, R))
    for q, (u, v, w) in enumerate((d0, d1, d2)):
        base = q * r
        for t in range(r):
            col = base + t
            for i in range(3):
                for k in range(3):
                    U[i * 3 + k, col] = u[t, i, k]
            for k in range(3):
                for j in range(2):
                    V[k * 6 + (q * 2 + j), col] = v[t, k, j]
            for i in range(3):
                for j in range(2):
                    W[(q * 2 + j) * 3 + i, col] = w[t, i, j]
    return U, V, W


def _verify_3x6(U, V, W, iters=12, seed=555):
    n, m, p = 3, 3, 6
    R = U.shape[1]
    rng = np.random.default_rng(seed)
    for _ in range(iters):
        a = rng.integers(-4, 5, size=(n, m))
        b = rng.integers(-4, 5, size=(m, p))
        C = a @ b
        Cc = np.zeros((n, p))
        for c in range(R):
            ua = np.array([[U[i * m + k, c] for k in range(m)] for i in range(n)])
            vb = np.array([[V[k * p + j, c] for j in range(p)] for k in range(m)])
            wa = np.array([[W[j * n + i, c] for i in range(n)] for j in range(p)])
            Cc += float(np.sum(ua * a)) * float(np.sum(vb * b)) * wa
        if not np.allclose(Cc, C, atol=1e-9, rtol=0.0):
            return False
    return True


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    # Try to find 3x3x2 decomposition of rank 15, 14, 13, 12, 11, 10, 9
    for r in range(15, 8, -1):
        if not _ok():
            break
        d0 = _find_3x3x2(r, seed=1000 + r)
        if d0 is None:
            continue
        d1 = _find_3x3x2(r, seed=3000 + r)
        if d1 is None:
            continue
        d2 = _find_3x3x2(r, seed=5000 + r)
        if d2 is None:
            continue
        U, V, W = _build_3x6_from_3x3x2(r, d0, d1, d2)
        if _verify_3x6(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to 3x3x3 rank 23 if possible (rank 46)
    for r in range(23, 20, -1):
        if not _ok():
            break
        d0 = _find_3x3(r, seed=1000 + r)
        if d0 is None:
            continue
        d1 = _find_3x3(r, seed=3000 + r)
        if d1 is None:
            continue
        U, V, W = _build_3x6(r, d0, d1)
        if _verify_3x6(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    return _schoolbook()
