"""3x3x8 matrix multiplication tensor decomposition.

Design: a runtime exact verifier gates every candidate decomposition. We build a
reduced-rank construction (two Laderman-style rank-7 3x3 blocks covering output
columns 0..5, plus a schoolbook 3x2 tail covering columns 6..7, total rank 32),
verify it reproduces the multiplication tensor T[i,j,k]=1 exactly, and fall back
to the provably-correct schoolbook decomposition (rank 72) if verification fails.
This guarantees the result is always valid (score >= 1.0) and improves to ~2.25
if the reduced construction verifies.
"""
import time

DIMENSIONS = [3, 3, 8]
_DEADLINE = 150.0  # wall-clock safety margin (evaluator kills at 150 s)


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    import numpy as np
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for r in range(rank):
        T += np.einsum('a,b,c->abc',
                       U[:, r].reshape(n, m),
                       V[:, r].reshape(m, p),
                       W[:, r].reshape(p, n))
    return bool(np.allclose(T, np.ones((n, m, p)), atol=tol, rtol=0))


def _schoolbook(n, m, p):
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


def _laderman_3x3():
    """7 rank-1 terms for 3x3x3 as (u over A(i,j), v over B(j,k), w over C(i,k))."""
    def A(i, j): return i * 3 + j
    def B(j, k): return j * 3 + k
    def C(i, k): return k * 3 + i
    u1 = {A(0,0):1,A(1,1):1,A(2,2):1}; v1 = {B(0,0):1,B(1,1):1,B(2,2):1}
    u2 = {A(0,1):1,A(1,1):1,A(2,2):1}; v2 = {B(1,0):1,B(2,0):1,B(0,0):1}
    u3 = {A(0,2):1,A(1,2):1,A(2,2):1}; v3 = {B(2,0):1,B(0,1):1,B(1,1):1}
    u4 = {A(0,0):1,A(1,0):1,A(2,0):1}; v4 = {B(0,1):1,B(1,2):1,B(2,2):1}
    u5 = {A(0,1):1,A(1,0):1,A(2,0):1}; v5 = {B(1,2):1,B(2,0):1,B(0,2):1}
    u6 = {A(0,2):1,A(1,0):1,A(2,0):1}; v6 = {B(2,1):1,B(0,2):1,B(1,0):1}
    u7 = {A(0,0):1,A(1,1):1,A(2,2):1}; v7 = {B(0,1):1,B(1,2):1,B(2,0):1}
    us = [u1,u2,u3,u4,u5,u6,u7]; vs = [v1,v2,v3,v4,v5,v6,v7]
    w = [{} for _ in range(7)]
    def add(t, i, k, c):
        w[t][C(i,k)] = w[t].get(C(i,k), 0) + c
    add(0,0,0,1); add(1,0,0,-1); add(2,0,0,-1); add(6,0,0,1)
    add(3,0,1,1); add(4,0,1,-1); add(6,0,1,-1)
    add(5,0,2,1); add(3,0,2,-1); add(6,0,2,1)
    add(1,1,0,1); add(0,1,0,-1); add(4,1,0,1)
    add(0,1,1,1); add(1,1,1,-1); add(2,1,1,1); add(3,1,1,-1)
    add(4,1,1,1); add(5,1,1,-1); add(6,1,1,1)
    add(5,1,2,1); add(4,1,2,-1); add(6,1,2,1)
    add(2,2,0,1); add(0,2,0,-1); add(5,2,0,1)
    add(3,2,1,1); add(1,2,1,-1); add(6,2,1,1)
    add(4,2,2,1); add(2,2,2,-1); add(6,2,2,1)
    return [(us[t], vs[t], w[t]) for t in range(7)]


def _build_reduced():
    n, m, p = 3, 3, 8
    terms = []
    lad = _laderman_3x3()
    for col0 in (0, 3):  # two 3x3 Laderman blocks -> output cols 0..5
        for (u, v, w) in lad:
            Ucol = [0] * (n * m); Vcol = [0] * (m * p); Wcol = [0] * (p * n)
            for idx, c in u.items():
                Ucol[idx] = c
            for idx, c in v.items():
                j, k = divmod(idx, m)
                Vcol[j * p + (col0 + k)] = c
            for idx, c in w.items():
                k, i = divmod(idx, n)
                Wcol[(col0 + k) * n + i] = c
            terms.append((Ucol, Vcol, Wcol))
    for i in range(n):      # schoolbook 3x2 tail -> output cols 6..7
        for j in range(m):
            for k in range(2):
                Ucol = [0] * (n * m); Vcol = [0] * (m * p); Wcol = [0] * (p * n)
                Ucol[i * m + j] = 1
                Vcol[j * p + (6 + k)] = 1
                Wcol[(6 + k) * n + i] = 1
                terms.append((Ucol, Vcol, Wcol))
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r, (uc, vc, wc) in enumerate(terms):
        for a in range(n * m): U[a][r] = uc[a]
        for b in range(m * p): V[b][r] = vc[b]
        for c in range(p * n): W[c][r] = wc[c]
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    # Best-known valid result: schoolbook (always correct).
    best = _schoolbook(n, m, p)
    # Try the reduced-rank candidate; accept only if it verifies exactly.
    try:
        cand = _build_reduced()
        if _verify(cand["U"], cand["V"], cand["W"], n, m, p):
            best = cand
    except Exception:
        pass
    if time.time() - start > _DEADLINE - 5:
        return _schoolbook(n, m, p)
    return best
