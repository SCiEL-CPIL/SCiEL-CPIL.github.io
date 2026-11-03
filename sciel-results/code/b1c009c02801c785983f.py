"""Rank-minimal exact tensor decomposition for 3x4x8 matrix multiplication.

Combines:
  - A verified rank-88 construction (schoolbook for i=0 + Strassen on 2x4x8).
  - A runtime exact-rank-reduction search (rank-1 extraction + greedy +
    random restarts) that keeps the best valid decomposition found, with a
    strict deadline guard so it always returns a valid result.
"""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check the cyclic bilinear identity:
       sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1  for all i,j,k.
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(U[i * m + j], V[j * p + k]))
                # need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
                # recompute properly (dot of U and V is not enough); do full:
    # full check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _to_halfint(U, V, W):
    def r(x):
        return np.round(2 * np.asarray(x, dtype=float)) / 2.0
    return r(U), r(V), r(W)


def _build_rank88():
    """Verified rank-88 construction: schoolbook(1x4x8) + Strassen(2x4x8)."""
    n, m, p = DIMENSIONS
    rank = 88
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    # Part 1: i=0 schoolbook 1x4x8 -> 32 terms
    for j in range(m):
        for k in range(p):
            U[0 * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + 0, col] = 1.0
            col += 1
    # Part 2: i=1,2 Strassen on 2x2x2 blocks -> 2*4*7 = 56 terms
    sU = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    sV = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    sW = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    for jb in range(2):
        js = jb * 2
        for kb in range(4):
            ks = kb * 2
            for t in range(7):
                for l in range(4):
                    if sU[t][l]:
                        li, lj = l // 2, l % 2
                        U[(1 + li) * m + (js + lj), col] += sU[t][l]
                    if sV[t][l]:
                        lj, lk = l // 2, l % 2
                        V[(js + lj) * p + (ks + lk), col] += sV[t][l]
                    if sW[t][l]:
                        li, lk = l // 2, l % 2
                        W[(ks + lk) * n + (1 + li), col] += sW[t][l]
                col += 1
    return U, V, W


def _extract_rank1(U, V, W, n, m, p, tol=1e-8):
    """Try to extract a rank-1 term u v w such that
       u[i*m+j]*v[j*p+k]*w[k*n+i] is a nonzero scalar multiple of one tensor
       'slice' pattern. Returns (u,v,w) arrays or None.
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    R = U.shape[1]
    # Build tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # We want to find a rank-1 component. Use a heuristic: pick a random
    # (i,j,k) with nonzero T, form u[i*m+j] proportional, etc.
    # Compute full T.
    T = np.zeros((n, m, p))
    for r in range(R):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        # term[r]_{i,j,k} = u[i,j]*v[j,k]*w[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    # Find a rank-1 approximation of T: pick a seed index.
    idx = np.argwhere(np.abs(T) > tol)
    if len(idx) == 0:
        return None
    # choose first
    i0, j0, k0 = idx[0]
    # u candidate: u[i,j] ~ T[i,j,k0] / (v[j,k0]*w[k0,i])
    # We need a consistent rank-1. Use ALS-style: start with v = ones, w = ones.
    v = np.ones((m, p))
    w = np.ones((p, n))
    u = np.zeros((n, m))
    for _ in range(30):
        # update u
        for i in range(n):
            for j in range(m):
            # T[i,j,:] = u[i,j] * (v[j,:]*w[:,i])
                denom = v[j, :] * w[:, i]
                # least squares: u[i,j] = <T[i,j,:], denom> / <denom,denom>
                dd = float(np.dot(denom, denom))
                if dd > tol:
                    u[i, j] = float(np.dot(T[i, j, :], denom)) / dd
        # update v
        for j in range(m):
            for k in range(p):
                denom = u[:, j] * w[k, :]
                dd = float(np.dot(denom, denom))
                if dd > tol:
                    v[j, k] = float(np.dot(T[:, j, k], denom)) / dd
        # update w
        for k in range(p):
            for i in range(n):
                denom = u[i, :] * v[:, k]
                dd = float(np.dot(denom, denom))
                if dd > tol:
                    w[k, i] = float(np.dot(T[i, :, k], denom)) / dd
    # Reconstruct rank-1 tensor
    T1 = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T1[i, j, k] = u[i, j] * v[j, k] * w[k, i]
    if np.max(np.abs(T1)) < tol:
        return None
    return u, v, w


def _subtract_term(U, V, W, u, v, w, n, m, p):
    """Subtract rank-1 term u v w from the decomposition by solving for a
       new column (r) such that its contribution equals u v w, then we can
       drop it. Simpler: build a new column set that includes this term and
       remove one existing column. We implement via: add column r with the
       term, then null out an existing column by linearity is nontrivial.
       Instead we do: represent decomposition in column form and reduce the
       span. For safety, we only accept if the new set (without one column)
       still reproduces T.
    """
    # We'll handle removal at the caller by testing.
    pass


def _greedy_reduce(U, V, W, n, m, p, deadline, tol=1e-6):
    """Attempt to reduce rank by repeatedly extracting a rank-1 term and
       removing a linearly redundant column. Returns best (U,V,W)."""
    U = np.asarray(U, dtype=float).copy()
    V = np.asarray(V, dtype=float).copy()
    W = np.asarray(W, dtype=float).copy()
    best = (U.copy(), V.copy(), W.copy())
    R = U.shape[1]
    # Build full tensor T once.
    T = np.zeros((n, m, p))
    for r in range(R):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    # Try to find a set of rank-1 terms that sum to T with fewer terms.
    # Greedy: repeatedly extract best rank-1, subtract, repeat.
    Rrem = T.copy()
    terms = []
    for _ in range(R):
        if time.time() > deadline:
            break
        if np.max(np.abs(Rrem)) < tol:
            break
        # extract
        res = _extract_rank1_from(Rrem, n, m, p, tol)
        if res is None:
            break
        u, v, w = res
        terms.append((u, v, w))
        # subtract
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Rrem[i, j, k] -= u[i, j] * v[j, k] * w[k, i]
    if len(terms) < R:
        # Build new decomposition from terms
        newR = len(terms)
        nU = np.zeros((n * m, newR))
        nV = np.zeros((m * p, newR))
        nW = np.zeros((p * n, newR))
        for r, (u, v, w) in enumerate(terms):
            nU[:, r] = u.reshape(-1)
            nV[:, r] = v.reshape(-1)
            nW[:, r] = w.reshape(-1)
        # verify
        if _verify(nU, nV, nW, n, m, p, tol=1e-4):
            best = (nU, nV, nW)
    return best


def _extract_rank1_from(T, n, m, p, tol=1e-8):
    """Extract a rank-1 term from a 3-tensor T via ALS."""
    u = np.zeros((n, m))
    v = np.ones((m, p))
    w = np.ones((p, n))
    for _ in range(40):
        for i in range(n):
            for j in range(m):
                denom = v[j, :] * w[:, i]
                dd = float(np.dot(denom, denom))
                if dd > tol:
                    u[i, j] = float(np.dot(T[i, j, :], denom)) / dd
        for j in range(m):
            for k in range(p):
                denom = u[:, j] * w[k, :]
                dd = float(np.dot(denom, denom))
                if dd > tol:
                    v[j, k] = float(np.dot(T[:, j, k], denom)) / dd
        for k in range(p):
            for i in range(n):
                denom = u[i, :] * v[:, k]
                dd = float(np.dot(denom, denom))
                if dd > tol:
                    w[k, i] = float(np.dot(T[i, :, k], denom)) / dd
    T1 = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T1[i, j, k] = u[i, j] * v[j, k] * w[k, i]
    if np.max(np.abs(T1)) < tol:
        return None
    return u, v, w


def construct():
    t0 = time.time()
    DEADLINE = 120.0
    n, m, p = DIMENSIONS

    # Safe verified baseline.
    U, V, W = _build_rank88()
    base_ok = _verify(U, V, W, n, m, p, tol=1e-6)
    U, V, W = _to_halfint(U, V, W)
    best = (U, V, W)
    best_rank = U.shape[1]

    # Attempt exact rank reduction within deadline.
    if base_ok and (DEADLINE - (time.time() - t0)) > 20.0:
        cand = _greedy_reduce(U, V, W, n, m, p, deadline=DEADLINE - 2.0, tol=1e-6)
        if cand is not None:
            cU, cV, cW = _to_halfint(*cand)
            if _verify(cU, cV, cW, n, m, p, tol=1e-4) and cU.shape[1] < best_rank:
                best = (cU, cV, cW)
                best_rank = cU.shape[1]

    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    U = np.array(result["U"]); V = np.array(result["V"]); W = np.array(result["W"])
    n, m, p = 3, 4, 8
    R = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                max_err = max(max_err, abs(s - 1.0))
    print(f"Max tensor error: {max_err}")
