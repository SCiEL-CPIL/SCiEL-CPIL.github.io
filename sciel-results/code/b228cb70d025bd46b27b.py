import time
import math
import random
import numpy as np

DIMENSIONS = [3, 3, 5]
DEADLINE = 150.0
START = time.time()
SAFETY = 12.0  # seconds to keep before the hard limit

# ---------------- tensor indexing helpers ----------------
def _build_indices(n, m, p):
    # A (n x m) flattened row-major -> index a = i*m + j
    # B (m x p) flattened row-major -> index b = j*p + k
    # C (p x n) flattened row-major -> index c = k*n + i
    A_idx, B_idx, C_idx = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A_idx.append(i * m + j)
                B_idx.append(j * p + k)
                C_idx.append(k * n + i)
    return (np.array(A_idx, dtype=np.int64),
            np.array(B_idx, dtype=np.int64),
            np.array(C_idx, dtype=np.int64))

def _tensor(U, V, W, A, B, C):
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # build the (n*m, m*p) matrix for each r, then sum W[c,r]*that
    r = U.shape[1]
    n, m, p = U.shape[0], V.shape[0], W.shape[0]
    out = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for rr in range(r):
        M = np.outer(U[:, rr], V[:, rr])  # (n*m, m*p)
        out[:, :, :] += M[:, :, None] * W[None, None, rr]
    return out[A, B, C]

def _residual(U, V, W, A, B, C):
    T = _tensor(U, V, W, A, B, C)
    return T - 1.0

def _softmin_residual(U, V, W, A, B, C, lam):
    # smooth surrogate for min(residual^2) via log-sum-exp
    r = _residual(U, V, W, A, B, C)
    s = r * r
    m = s.max()
    if not np.isfinite(m):
        return float('inf'), None
    e = np.exp(-(s - m) / lam)
    val = m + lam * math.log(e.sum())
    # gradient w.r.t. s: d/ds [ m + lam*log(sum exp(-(s-m)/lam)) ]
    #   = - (exp(-(s-m)/lam) / sum exp(...))
    w = e / e.sum()
    return val, w  # w is d val / d s

def _softmin_grads(U, V, W, A, B, C, lam):
    val, w = _softmin_residual(U, V, W, A, B, C, lam)
    if w is None:
        return val, None, None, None
    # d val / d r = 2 r * w
    r = _residual(U, V, W, A, B, C)
    dr = 2.0 * r * w  # (N,)
    # d val / d U[a,rr] = sum_{b,c: (a,b,c)} dr * V[b,rr] * W[c,rr]
    gU = np.zeros_like(U)
    gV = np.zeros_like(V)
    gW = np.zeros_like(W)
    for rr in range(U.shape[1]):
        # accumulate: for each (a,b,c) with weight dr, add V[b]*W[c] to U[a]
        # vectorize over the N triples
        gU[:, rr] = np.add.at(np.zeros(U.shape[0]), A, dr * (V[B, rr] * W[C, rr]))
        gV[:, rr] = np.add.at(np.zeros(V.shape[0]), B, dr * (U[A, rr] * W[C, rr]))
        gW[:, rr] = np.add.at(np.zeros(W.shape[0]), C, dr * (U[A, rr] * V[B, rr]))
    return val, gU, gV, gW

# ---------------- exact integer validation ----------------
def _is_half_int(x, tol=1e-6):
    return np.allclose(x, np.round(2 * x) / 2.0, atol=tol)

def _validate(U, V, W, A, B, C):
    # exact check: tensor entries must be exactly 0 or 1 (within tiny fp error)
    T = _tensor(U, V, W, A, B, C)
    # each entry should be 1.0
    if not np.allclose(T, 1.0, atol=1e-6, rtol=1e-6):
        return False
    # entries must be half-integers
    if not (_is_half_int(U) and _is_half_int(V) and _is_half_int(W)):
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
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _to_half_int(U, V, W, tol=1e-3):
    # snap to nearest half-integer and verify exactness
    Uh = np.round(2 * U) / 2.0
    Vh = np.round(2 * V) / 2.0
    Wh = np.round(2 * W) / 2.0
    if not (np.allclose(Uh, U, atol=tol) and np.allclose(Vh, V, atol=tol) and np.allclose(Wh, W, atol=tol)):
        return None
    return Uh, Vh, Wh

def _try_polish(U, V, W, A, B, C):
    # local high-precision polish: small Newton-ish steps on the exact residual,
    # then snap to half-integers and validate.
    best = None
    U, V, W = U.copy(), V.copy(), W.copy()
    r = U.shape[1]
    for _ in range(40):
        if time.time() - START > DEADLINE - SAFETY:
            break
        T = _tensor(U, V, W, A, B, C)
        res = T - 1.0
        if np.max(np.abs(res)) < 1e-9:
            break
        # gradient of sum res^2
        r2 = res * res
        # reuse softmin machinery with tiny lam for a smooth exact-ish gradient
        val, gU, gV, gW = _softmin_grads(U, V, W, A, B, C, lam=1e-6)
        if gU is None:
            break
        # adaptive step
        lr = 1e-3
        for _ in range(6):
            U2, V2, W2 = U - lr * gU, V - lr * gV, W - lr * gW
            T2 = _tensor(U2, V2, W2, A, B, C)
            err2 = np.sum((T2 - 1.0) ** 2)
            err = np.sum(r2)
            if err2 < err:
                U, V, W = U2, V2, W2
                break
            lr *= 0.5
        snap = _to_half_int(U, V, W, tol=1e-3)
        if snap is not None:
            Uh, Vh, Wh = snap
            if _validate(Uh, Vh, Wh, A, B, C):
                if best is None or Uh.shape[1] < best[0]:
                    best = (Uh.shape[1], Uh, Vh, Wh)
    return best

# ---------------- main search ----------------
def construct():
    global START
    START = time.time()
    n, m, p = DIMENSIONS
    A, B, C = _build_indices(n, m, p)

    # best known valid decomposition
    best = None
    U0, V0, W0 = _schoolbook(n, m, p)
    if _validate(U0, V0, W0, A, B, C):
        best = (U0.shape[1], U0.copy(), V0.copy(), W0.copy())

    def consider(r, U, V, W):
        nonlocal best
        snap = _to_half_int(U, V, W, tol=1e-3)
        if snap is None:
            return
        Uh, Vh, Wh = snap
        if _validate(Uh, Vh, Wh, A, B, C):
            if best is None or r < best[0]:
                best = (r, Uh.copy(), Vh.copy(), Wh.copy())

    rng = random.Random(12345)
    # Start from schoolbook, try to remove columns by zeroing and re-optimizing.
    rank = U0.shape[1]
    curU, curV, curW = U0.copy(), V0.copy(), W0.copy()

    # Phase 1: greedy column removal with smooth optimization to fill gaps
    for attempt in range(60):
        if time.time() - START > DEADLINE - SAFETY:
            break
        if best is not None and rank <= best[0]:
            # no room to improve
            break
        # pick a random column to drop
        col = rng.randrange(rank)
        keep = [c for c in range(rank) if c != col]
        U = curU[:, keep]
        V = curV[:, keep]
        W = curW[:, keep]
        newrank = len(keep)
        # optimize to reduce residual
        for it in range(600):
            if time.time() - START > DEADLINE - SAFETY:
                break
            lam = max(0.02, 0.5 * (0.97 ** it))
            val, gU, gV, gW = _softmin_grads(U, V, W, A, B, C, lam)
            if gU is None:
                break
            lr = 0.05
            for _ in range(8):
                U2, V2, W2 = U - lr * gU, V - lr * gV, W - lr * gW
                val2, _, _, _ = _softmin_grads(U2, V2, W2, A, B, C, lam)
                if val2 < val:
                    U, V, W = U2, V2, W2
                    val = val2
                    break
                lr *= 0.5
            if val < 1e-4:
                break
        consider(newrank, U, V, W)
        if best is not None and best[0] == newrank:
            curU, curV, curW = best[1], best[2], best[3]
            rank = newrank
        else:
            # revert to last best if we have one, else keep trying
            if best is not None:
                curU, curV, curW = best[1].copy(), best[2].copy(), best[3].copy()
                rank = best[0]
            else:
                curU, curV, curW = U0.copy(), V0.copy(), W0.copy()
                rank = U0.shape[1]

    # Phase 2: final polish on the best found
    if best is not None:
        r, Uh, Vh, Wh = best
        pol = _try_polish(Uh, Vh, Wh, A, B, C)
        if pol is not None and pol[0] < r:
            best = pol

    if best is None:
        # fallback: always return the schoolbook (valid)
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    r, Uh, Vh, Wh = best
    return {"U": Uh.tolist(), "V": Vh.tolist(), "W": Wh.tolist()}
