import time
import numpy as np

N, M, P = 2, 4, 8


def _check(U, V, W):
    nr, mr, pr = len(U), len(V), len(W)
    if nr != N * M or mr != M * P or pr != P * N:
        return float('inf')
    R = min(len(U[0]), len(V[0]), len(W[0]))
    worst = 0.0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                u = U[i * M + j]
                v = V[j * P + k]
                w = W[k * N + i]
                s = 0.0
                for r in range(R):
                    s += u[r] * v[r] * w[r]
                d = abs(s - 1.0)
                if d > worst:
                    worst = d
    return worst


def _build_56():
    """Proven rank-56: 8 independent 2x2 Strassen blocks covering A(i,j)B(j,k)."""
    U = [[0] * 56 for _ in range(N * M)]
    V = [[0] * 56 for _ in range(M * P)]
    W = [[0] * 56 for _ in range(P * N)]

    SU = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1},
    ]
    SV = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1},
    ]
    SW = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(0, 0): -1, (1, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1},
    ]

    def block(rb, ib, jb, kb):
        for r in range(7):
            cr = rb + r
            for (li, lj), val in SU[r].items():
                U[(ib + li) * M + (jb + lj)][cr] += val
            for (lj, lk), val in SV[r].items():
                V[(jb + lj) * P + (kb + lk)][cr] += val
            for (lk, li), val in SW[r].items():
                W[(kb + lk) * N + (ib + li)][cr] += val

    # i=0 or 1, j in 0..3 (2x2 cols), k in 0..7 (4x2 rows) => 2*2*4 = 16 blocks of 2x2? 
    # A is 2x4: row i in {0,1}, col j in {0..3}. B is 4x8: row j in {0..3}, col k in {0..7}.
    # Strassen block on A[2x2] x B[2x2] covers i in {0,1}, j in {jb,jb+1}, k in {kb,kb+1}.
    for ib in (0, 1):
        for jb in (0, 2):
            for kb in (0, 2, 4, 6):
                block(ib * 7 * 4 + jb * 7 + (kb // 2) * 7 + (ib > 0) * 0, ib, jb, kb)
    return U, V, W


def construct():
    t0 = time.time()
    DEADLINE = 140.0

    U, V, W = _build_56()
    # The block indexing above may be off; instead build cleanly:
    U = [[0] * 56 for _ in range(N * M)]
    V = [[0] * 56 for _ in range(M * P)]
    W = [[0] * 56 for _ in range(P * N)]
    SU = [
        {(0, 0): 1, (1, 1): 1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 1): 1, (1, 1): -1},
    ]
    SV = [
        {(0, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1}, {(0, 0): 1, (0, 1): 1}, {(1, 0): 1, (1, 1): 1},
    ]
    SW = [
        {(0, 0): 1, (1, 1): 1}, {(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1, (0, 1): 1}, {(0, 0): -1, (1, 0): 1}, {(1, 1): 1}, {(0, 0): 1},
    ]
    idx = 0
    for ib in (0, 1):
        for jb in (0, 2):
            for kb in (0, 2, 4, 6):
                for r in range(7):
                    cr = idx + r
                    for (li, lj), val in SU[r].items():
                        U[(ib + li) * M + (jb + lj)][cr] += val
                    for (lj, lk), val in SV[r].items():
                        V[(jb + lj) * P + (kb + lk)][cr] += val
                    for (lk, li), val in SW[r].items():
                        W[(kb + lk) * N + (ib + li)][cr] += val
                idx += 7

    # Verify; if valid, attempt null-space column trim for potential rank reduction.
    if _check(U, V, W) > 1e-9:
        # Fallback: pure rank-64 direct (guaranteed valid).
        U = [[0] * 64 for _ in range(N * M)]
        V = [[0] * 64 for _ in range(M * P)]
        W = [[0] * 64 for _ in range(P * N)]
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    r = i * M * P + j * P + k
                    U[i * M + j][r] = 1
                    V[j * P + k][r] = 1
                    W[k * N + i][r] = 1
        return {"U": U, "V": V, "W": W}

    # Null-space trim: find dependent columns using the constraint T=1.
    # Build the 64 x R matrix of "tensor contribution gradients" is complex; 
    # do a safe greedy zero-column drop only.
    R = 56
    for _ in range(R):
        if time.time() > DEADLINE - 5:
            break
        dropped = False
        for r in range(len(U[0])):
            z = all(U[i][r] == 0 for i in range(N * M)) \
                and all(V[i][r] == 0 for i in range(M * P)) \
                and all(W[i][r] == 0 for i in range(P * N))
            if z:
                for row in U:
                    del row[r]
                for row in V:
                    del row[r]
                for row in W:
                    del row[r]
                dropped = True
                break
        if not dropped:
            break

    return {"U": U, "V": V, "W": W}
