"""Matrix multiplication 2x2x9 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def _build_rank32(n, m, p):
    """Deterministic rank-32 decomposition: 4 Strassen 2x2x2 blocks (rank 7)
    + 1 trivial 2x2x1 block (rank 4). Returns integer numpy arrays U,V,W."""
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = [0]

    def add_strassen(k0, k1):
        c = col[0]
        # M1 = (a00+a11)(b00+b11)
        U[0, c] = 1; U[3, c] = 1
        V[0, c] = 1; V[1 * p + k1, c] = 1
        W[k0 * n + 0, c] = 1; W[k1 * n + 1, c] = 1
        col[0] += 1
        # M2 = (a10+a11)b00
        c = col[0]
        U[2, c] = 1; U[3, c] = 1
        V[0, c] = 1
        W[k0 * n + 1, c] = 1; W[k1 * n + 1, c] = -1
        col[0] += 1
        # M3 = a00(b10-b11)
        c = col[0]
        U[0, c] = 1
        V[1 * p + k0, c] = 1; V[1 * p + k1, c] = -1
        W[k0 * n + 0, c] = 1; W[k1 * n + 0, c] = -1
        col[0] += 1
        # M4 = a11(b01-b00)
        c = col[0]
        U[3, c] = 1
        V[1 * p + k1, c] = 1; V[0, c] = -1
        W[k1 * n + 1, c] = 1; W[k0 * n + 1, c] = -1
        col[0] += 1
        # M5 = (a00+a01)b11
        c = col[0]
        U[0, c] = 1; U[1, c] = 1
        V[1 * p + k1, c] = 1
        W[k1 * n + 0, c] = 1; W[k0 * n + 0, c] = -1
        col[0] += 1
        # M6 = (a10-a00)(b00+b10)
        c = col[0]
        U[2, c] = 1; U[0, c] = -1
        V[0, c] = 1; V[1 * p + k0, c] = 1
        W[k0 * n + 1, c] = 1; W[k1 * n + 1, c] = 1
        col[0] += 1
        # M7 = (a01-a11)(b01+b11)
        c = col[0]
        U[1, c] = 1; U[3, c] = -1
        V[1 * p + k1, c] = 1; V[1 * p + k0, c] = 1
        W[k1 * n + 0, c] = 1; W[k0 * n + 0, c] = 1
        col[0] += 1

    def add_trivial(k):
        for a in range(n):
            for b in range(m):
                c = col[0]
                U[a * m + b, c] = 1
                V[b * p + k, c] = 1
                W[k * n + a, c] = 1
                col[0] += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)
    return U, V, W


def _true_tensor(n, m, p):
    """The matrix-multiplication tensor T[i,j,k] = 1 when i=a*m+b, j=b*p+c, k=c*n+a."""
    T = np.zeros((n * m, m * p, p * n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                T[a * m + b, b * p + c, c * n + a] = 1.0
    return T


def _decomposed(U, V, W):
    return np.einsum('ir,jr,kr->ijk', U, V, W)


def _max_res(U, V, W, T):
    return np.max(np.abs(_decomposed(U, V, W) - T))


def _refine(U, V, W, T, time_budget):
    """Polish a rank-r decomposition with alternating least-squares (rank-preserving)."""
    start = time.time()
    U, V, W = U.copy(), V.copy(), W.copy()
    for _ in range(40):
        if time.time() - start > time_budget:
            break
        r = U.shape[1]
        # U from (V*W)
        A = np.einsum('jr,kr->jk r', V, W)
        b = np.einsum('ijk,ik->j r', T, W)
        U = np.linalg.lstsq(A, b, rcond=None)[0]
        # V from (U*W)
        A = np.einsum('ir,kr->jk r', U, W)
        b = np.einsum('ijk,jk->i r', T, W)
        V = np.linalg.lstsq(A, b, rcond=None)[0]
        # W from (U*V)
        A = np.einsum('ir,jr->ij r', U, V)
        b = np.einsum('ijk,ij->k r', T, U)
        W = np.linalg.lstsq(A, b, rcond=None)[0]
    return U, V, W


def _try_rank31(U, V, W, T, time_budget):
    """Try to drop one column of the rank-32 decomposition to get an exact rank-31.
    For each candidate column, set it to zero and solve the remaining 31 columns
    by least squares; accept if the residual is exactly zero."""
    start = time.time()
    r = U.shape[1]
    for drop in range(r):
        if time.time() - start > time_budget:
            break
        U31 = U[:, drop != np.arange(r)]
        V31 = V[:, drop != np.arange(r)]
        W31 = W[:, drop != np.arange(r)]
        U31, V31, W31 = _refine(U31, V31, W31, T, time_budget - (time.time() - start))
        res = _max_res(U31, V31, W31, T)
        if res < 1e-9:
            return U31, V31, W31
    return None


def _to_lists(U, V, W):
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _verify(U, V, W, n, m, p, tol=1e-8):
    T = _true_tensor(n, m, p)
    return _max_res(np.asarray(U, float), np.asarray(V, float), np.asarray(W, float), T) < tol


def construct():
    start_time = time.time()
    deadline = 120.0
    n, m, p = DIMENSIONS
    T = _true_tensor(n, m, p)

    # 1) Deterministic rank-32 baseline (fast, guaranteed valid).
    U32, V32, W32 = _build_rank32(n, m, p)
    res32 = _max_res(U32, V32, W32, T)
    if res32 > 1e-9:
        # safety: fall back to pure schoolbook rank-36
        rank = n * m * p
        U = np.zeros((n * m, rank)); V = np.zeros((m * p, rank)); W = np.zeros((p * n, rank))
        col = 0
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    U[a * m + b, col] = 1
                    V[b * p + c, col] = 1
                    W[c * n + a, col] = 1
                    col += 1
        return _to_lists(U, V, W)
    best = (32, U32, V32, W32)

    # 2) Attempt rank-31 via column elimination + least-squares refinement.
    if time.time() - start_time < deadline - 15:
        budget = max(1.0, deadline - 15 - (time.time() - start_time))
        out = _try_rank31(U32, V32, W32, T, budget)
        if out is not None:
            U31, V31, W31 = out
            if _verify(U31, V31, W31, n, m, p):
                best = (31, U31, V31, W31)

    r, U, V, W = best
    return _to_lists(U, V, W)
# EVOLVE-BLOCK-END
