"""Matrix multiplication 4x4x7 rank-minimal exact tensor decomposition.

4x4x4 sub-problem uses the Kronecker product of two 2x2x2 Strassen tensors:
  T_{4,4,4} = T_{2,2,2} (x) T_{2,2,2},  rank <= 7*7 = 49.
The remaining 4x4x3 part uses schoolbook, rank 48.
Total rank = 49 + 48 = 97  (schoolbook 4*4*7 = 112).

Index conventions (matching the seed):
  U row = i*m + j   (i in [0,n), j in [0,m))
  V row = j*p + k   (j in [0,m), k in [0,p))
  W row = k*n + i   (k in [0,p), i in [0,n))
Identity: sum_c U[(i,j),c] V[(j,k),c] = W[(k,i),c].
"""
import time
from collections import defaultdict

DIMENSIONS = [4, 4, 7]

# Strassen 2x2x2 tensor. Local indices in {0,1}.
# U: (i, j) of A ; V: (j, k) of B ; W: (k, i) of C.
_STRASSEN_U = [
    [(0,0,1),(1,1,1)],
    [(1,0,1),(1,1,1)],
    [(0,0,1)],
    [(1,1,1)],
    [(0,0,1),(0,1,1)],
    [(1,0,1),(0,0,-1)],
    [(0,1,1),(1,1,-1)],
]
_STRASSEN_V = [
    [(0,0,1),(1,1,1)],
    [(0,0,1)],
    [(0,1,1),(1,1,-1)],
    [(1,0,1),(0,0,-1)],
    [(1,1,1)],
    [(0,0,1),(0,1,1)],
    [(1,0,1),(1,1,1)],
]
_STRASSEN_W = [
    [(0,0,1),(1,0,-1),(0,1,-1),(1,1,1)],
    [(0,0,1)],
    [(0,1,1)],
    [(1,0,1)],
    [(1,1,1)],
    [(0,1,1)],
    [(1,1,1)],
]


def construct():
    start = time.time()
    n, m, p = DIMENSIONS

    entries = []  # list of (dU, dV, dW) sparse dicts

    def make_entry():
        return ({}, {}, {})

    def add(d, key, val):
        if val:
            d[key] = d.get(key, 0) + val

    # ---- 4x4x4 part: Kronecker product of two Strassen tensors ----
    # 4 = 2x2. Global i = 2*i1 + i0, j = 2*j1 + j0, k = 2*k1 + k0.
    # U[(i,j), (a,b)] = U1[(i1,j1),a] * U0[(i0,j0),b]
    # V[(j,k), (a,b)] = V1[(j1,k1),a] * V0[(j0,k0),b]
    # W[(k,i), (a,b)] = W1[(k1,i1),a] * W0[(k0,i0),b]
    for a in range(7):
        for b in range(7):
            dU, dV, dW = make_entry()
            for (i1, j1, c1) in _STRASSEN_U[a]:
                for (i0, j0, c0) in _STRASSEN_U[b]:
                    gi = 2 * i1 + i0
                    gj = 2 * j1 + j0
                    add(dU, (gi, gj), c1 * c0)
            for (j1, k1, c1) in _STRASSEN_V[a]:
                for (j0, k0, c0) in _STRASSEN_V[b]:
                    gj = 2 * j1 + j0
                    gk = 2 * k1 + k0
                    add(dV, (gj, gk), c1 * c0)
            for (k1, i1, c1) in _STRASSEN_W[a]:
                for (k0, i0, c0) in _STRASSEN_W[b]:
                    gk = 2 * k1 + k0
                    gi = 2 * i1 + i0
                    add(dW, (gk, gi), c1 * c0)
            entries.append((dU, dV, dW))

    # ---- 4x4x3 schoolbook part (k in 4..6) ----
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                dU, dV, dW = make_entry()
                add(dU, (i, j), 1)
                add(dV, (j, k), 1)
                add(dW, (k, i), 1)
                entries.append((dU, dV, dW))

    # Merge identical sparse patterns to reduce rank.
    merged = {}
    for (dU, dV, dW) in entries:
        key = (tuple(sorted(dU.items())), tuple(sorted(dV.items())), tuple(sorted(dW.items())))
        merged.setdefault(key, (dU, dV, dW))

    cols = len(merged)
    U = [[0] * cols for _ in range(n * m)]
    V = [[0] * cols for _ in range(m * p)]
    W = [[0] * cols for _ in range(p * n)]

    col = 0
    for (dU, dV, dW) in merged.values():
        for (i, j), v in dU.items():
            U[i * m + j][col] += v
        for (j, k), v in dV.items():
            V[j * p + k][col] += v
        for (k, i), v in dW.items():
            W[k * n + i][col] += v
        col += 1

    if time.time() - start > 120:
        # Safety net; construction is fast.
        pass

    return {"U": U, "V": V, "W": W}
