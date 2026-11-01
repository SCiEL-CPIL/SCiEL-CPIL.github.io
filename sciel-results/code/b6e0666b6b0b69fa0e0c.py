import numpy as np
import time

DIMENSIONS = [2, 2, 11]

def _build_tensor(n, m, p):
    # T[a, b, c] with a = i*m+j (A index), b = j*p+k (B index), c = k*n+i (C index)
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _round_half(x):
    return np.round(2.0 * x) / 2.0

def _verify(U, V, W, T, tol=1e-7):
    U = _round_half(U); V = _round_half(V); W = _round_half(W)
    M = np.einsum('ar,br,cr->abc', U, V, W)
    return bool(np.max(np.abs(M - T)) < tol), U, V, W

def _als(T, rank, iters, seed, deadline):
    rng = np.random.default_rng(seed)
    A, B, C = T.shape
    U = rng.normal(size=(A, rank))
    V = rng.normal(size=(B, rank))
    W = rng.normal(size=(C, rank))
    # normalize columns
    for r in range(rank):
        nU = np.linalg.norm(U[:, r]); nV = np.linalg.norm(V[:, r]); nW = np.linalg.norm(W[:, r])
        if nU > 0 and nV > 0 and nW > 0:
            U[:, r] /= nU; V[:, r] /= nV; W[:, r] /= nW
    M = np.einsum('ar,br,cr->abc', U, V, W)
    for it in range(iters):
        if time.time() > deadline:
            break
        for r in range(rank):
            term = U[:, r, None, None] * V[:, r][None, :, None] * W[:, r][None, None, :]
            R = T - M + term
            denom = (np.linalg.norm(V[:, r]) ** 2) * (np.linalg.norm(W[:, r]) ** 2)
            U[:, r] = np.einsum('br,cr,abc->a', V[:, r], W[:, r], R) / denom
            M = np.einsum('ar,br,cr->abc', U, V, W)
        for r in range(rank):
            term = U[:, r, None, None] * V[:, r][None, :, None] * W[:, r][None, None, :]
            R = T - M + term
            denom = (np.linalg.norm(U[:, r]) ** 2) * (np.linalg.norm(W[:, r]) ** 2)
            V[:, r] = np.einsum('ar,cr,abc->b', U[:, r], W[:, r], R) / denom
            M = np.einsum('ar,br,cr->abc', U, V, W)
        for r in range(rank):
            term = U[:, r, None, None] * V[:, r][None, :, None] * W[:, r][None, None, :]
            R = T - M + term
            denom = (np.linalg.norm(U[:, r]) ** 2) * (np.linalg.norm(V[:, r]) ** 2)
            W[:, r] = np.einsum('ar,br,abc->c', U[:, r], V[:, r], R) / denom
            M = np.einsum('ar,br,cr->abc', U, V, W)
    return U, V, W

def _to_lists(U, V, W):
    def conv(M):
        out = []
        for row in M:
            r = []
            for v in row:
                fv = float(v)
                # snap to integer or half-integer
                r.append(fv)
            out.append(r)
        return out
    return {"U": conv(U), "V": conv(V), "W": conv(W)}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 105.0  # safety margin under 120/150s
    T = _build_tensor(n, m, p)
    schoolbook_rank = n * m * p

    # flattening lower bound
    LB = max(n * m, m * p, p * n)

    # search ranks from LB upward
    for rank in range(LB, schoolbook_rank + 1):
        if time.time() > deadline - 5.0:
            break
        # more effort for lower ranks
        if rank <= 26:
            seeds = [0, 1, 2, 3, 4, 5]
            iters = 700
        elif rank <= 33:
            seeds = [0, 1, 2, 3]
            iters = 500
        else:
            seeds = [0, 1, 2]
            iters = 350
        for seed in seeds:
            if time.time() > deadline - 5.0:
                break
            U, V, W = _als(T, rank, iters, seed, deadline)
            ok, Ur, Vr, Wr = _verify(U, V, W, T)
            if ok:
                return _to_lists(Ur, Vr, Wr)
    # fallback: exact schoolbook
    return _schoolbook(n, m, p)
