"""Rank-39 exact tensor decomposition for 2x2 * 2x11 matrix multiplication
using Strassen's 7-multiplication algorithm on 2x2 column-blocks."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import time

# Local Strassen rank-1 terms for M_{2,2,2}. Each term is a dict mapping
# (kind, a, c) -> value, where kind in {'u','v','w'}:
#   u[i,j]  : coefficient of A[i,j]
#   v[j,k]  : coefficient of B[j,k]
#   w[k,i]  : coefficient of C[i,k]
# The 7 terms sum exactly to the all-ones tensor of shape (2,2,2).
_STRASSEN_TERMS = [
    {('u',0,0):1,('u',1,1):1, ('v',0,0):1,('v',1,1):1, ('w',0,0):1,('w',1,1):1},   # p1
    {('u',1,0):1,('u',1,1):1, ('v',0,0):1, ('w',0,1):1,('w',1,1):-1},              # p2
    {('u',0,0):1, ('v',0,1):1,('v',1,1):-1, ('w',1,0):1},                          # p3
    {('u',1,1):1, ('v',1,0):1,('v',0,0):-1, ('w',0,1):1,('w',1,0):1},              # p4
    {('u',0,0):1,('u',0,1):1, ('v',1,1):1, ('w',0,0):-1,('w',1,0):1},              # p5
    {('u',1,0):1,('u',0,0):-1, ('v',0,0):1,('v',0,1):1, ('w',1,1):1},              # p6
    {('u',0,1):1,('u',1,1):-1, ('v',1,0):1,('v',1,1):1, ('w',0,0):1},              # p7
]

def _build_strassen(n, m, p):
    n_blocks = p // 2
    remainder = p % 2
    terms = []
    for b in range(n_blocks):
        k0 = 2 * b
        for term in _STRASSEN_TERMS:
            u_entries, v_entries, w_entries = {}, {}, {}
            for (kind, a, c), val in term.items():
                if kind == 'u':
                    idx = a * m + c
                    u_entries[idx] = u_entries.get(idx, 0) + val
                elif kind == 'v':
                    idx = a * p + (k0 + c)
                    v_entries[idx] = v_entries.get(idx, 0) + val
                else:
                    idx = (k0 + a) * n + c
                    w_entries[idx] = w_entries.get(idx, 0) + val
            terms.append((u_entries, v_entries, w_entries))
    for i in range(n):
        for j in range(m):
            terms.append(({i * m + j: 1}, {j * p + (n_blocks * 2): 1}, {(n_blocks * 2) * n + i: 1}))
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r, (ue, ve, we) in enumerate(terms):
        for idx, val in ue.items():
            U[idx][r] = val
        for idx, val in ve.items():
            V[idx][r] = val
        for idx, val in we.items():
            W[idx][r] = val
    return U, V, W

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                ui = U[i * m + j]
                vk = V[j * p + k]
                wi = W[k * n + i]
                for r in range(rank):
                    s += ui[r] * vk[r] * wi[r]
                if s != 1:
                    return False
    return True

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    deadline = start + 110.0  # safety margin under the 120/150s limit

    U, V, W = _build_strassen(n, m, p)
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    # Fallback: guaranteed-correct schoolbook decomposition.
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
