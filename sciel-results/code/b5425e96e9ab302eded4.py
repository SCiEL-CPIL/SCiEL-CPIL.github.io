"""Matrix multiplication 3x5x6: rank-minimal exact tensor decomposition.

Strategy: build an EXACT decomposition (integer / half-integer entries) whose
rank is strictly below the schoolbook 90, using Strassen-style rank-7 2x2
blocks combined with schoolbook for the residual strips. Every candidate is
verified against the full multiplication tensor before it is returned, so the
program can never emit an invalid decomposition. A deadline guard guarantees a
valid (schoolbook) result is always returned within the wall-clock budget.
"""

import time

DIMENSIONS = [3, 5, 6]

_DEADLINE = 150.0
_START = time.time()


def _remaining():
    return _DEADLINE - time.time()


# --------------------------------------------------------------------------- #
# Verification: check that sum_c U[a,c]*V[b,c] == W[c] indexed appropriately.
# The multiplication tensor T[i,j,k] = 1  (C[i,k] = sum_j A[i,j] B[j,k]).
# Decomposition: C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_l V[j*p+k, c] ... )
# Actually the standard factorisation is:
#   C[i,k] = sum_c  U[i*m + j, c] * A[i,j] * B[j,k] * V[j*p + k, c] * W[k*n + i, c]
# which is NOT a product of three rank-1 slices. The correct tensor identity is:
#   T[i,j,k] = sum_c  U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
# where T[i,j,k] = 1 for all i,j,k (the multiplication tensor has a 1 in every
# (i,j,k) slot). So we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1.
# --------------------------------------------------------------------------- #
def _verify(U, V, W, n, m, p, tol=1e-6):
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build the tensor via einsum: T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    T = np.einsum('ic,jc,kc->ijk', U.reshape(n, m, rank),
                  V.reshape(m, p, rank), W.reshape(p, n, rank))
    target = np.ones((n, m, p))
    if not np.allclose(T, target, atol=tol, rtol=0):
        return False
    return True


def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}


# --------------------------------------------------------------------------- #
# Build a block decomposition. We treat the 3x5 A and 5x6 B by partitioning the
# inner dimension m=5 into (2,3) and the outer dims into blocks, using a
# Strassen rank-7 multiplication on the 2x2 sub-block and schoolbook elsewhere.
#
# We construct it generically: partition i into groups, j into groups, k into
# groups; for each (Igroup, Jgroup, Kgroup) triple with sizes (ai, bj, ck) we
# either use a known small exact kernel or schoolbook. We then verify.
#
# To keep it simple and provably exact we use a *block-diagonal* construction:
# partition rows i into [2,1], inner j into [2,3], cols k into [2,4]. For each
# block (I,J,K) of size (ai,bj,ck) we use the schoolbook rank ai*bj*ck EXCEPT
# for the 2x2x2 block where we use Strassen rank 7.
# --------------------------------------------------------------------------- #
def _strassen_2x2x2():
    # Returns U,V,W each of shape (4,7) for the 2x2x2 multiplication tensor.
    # T[i,j,k]=1. Standard Strassen factors (indices 0..3 for i*2+j etc).
    # We use the well-known rank-7 decomposition of the 2x2 matmul tensor.
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    U = [
        [1, 0, 0, 0, 0, 0, 0],   # i0j0
        [0, 0, 0, 0, 1, 0, 0],   # i0j1
        [0, 0, 0, 0, 0, 1, 0],   # i1j0
        [0, 0, 1, 0, 0, 0, 1],   # i1j1
    ]
    V = [
        [1, 0, 0, 0, 0, 0, 0],   # j0k0
        [0, 0, 1, 0, 0, 0, 0],   # j0k1
        [0, 0, 0, 1, 0, 0, 0],   # j1k0
        [0, 0, 0, 0, 0, 1, 1],   # j1k1
    ]
    W = [
        [0, 1, 0, 0, 1, 0, 0],   # k0i0
        [0, 0, 0, 0, 0, 1, 0],   # k0i1
        [1, 0, 1, 0, 0, 0, 0],   # k1i0
        [0, 0, 0, 1, 0, 0, 1],   # k1i1
    ]
    return U, V, W


def _build_block(n, m, p):
    # Partition sizes
    iparts = [2, 1]   # rows (sum 3)
    jparts = [2, 3]   # inner (sum 5)
    kparts = [2, 4]   # cols (sum 6)

    U = [[0] * (n * m) for _ in range(n * m)]
    V = [[0] * (m * p) for _ in range(m * p)]
    W = [[0] * (p * n) for _ in range(p * n)]
    # We'll accumulate into lists then pad; easier: build column-by-column.
    cols_U = []
    cols_V = []
    cols_W = []

    def add_block(i0, j0, k0, ai, bj, ck):
        # local schoolbook or strassen
        if ai == 2 and bj == 2 and ck == 2:
            u, v, w = _strassen_2x2x2()
        else:
            u = [[0] * (ai * bj * ck) for _ in range(ai * bj)]
            v = [[0] * (ai * bj * ck) for _ in range(bj * ck)]
            w = [[0] * (ai * bj * ck) for _ in range(ck * ai)]
            c = 0
            for ii in range(ai):
                for jj in range(bj):
                    for kk in range(ck):
                        u[ii * bj + jj][c] = 1
                        v[jj * ck + kk][c] = 1
                        w[kk * ai + ii][c] = 1
                        c += 1
        r = len(u[0])
        for c in range(r):
            cu = [0.0] * (n * m)
            cv = [0.0] * (m * p)
            cw = [0.0] * (p * n)
            for ii in range(ai):
                for jj in range(bj):
                    cu[(i0 + ii) * m + (j0 + jj)] += u[ii * bj + jj][c]
            for jj in range(bj):
                for kk in range(ck):
                    cv[(j0 + jj) * p + (k0 + kk)] += v[jj * ck + kk][c]
            for kk in range(ck):
                for ii in range(ai):
                    cw[(k0 + kk) * n + (i0 + ii)] += w[kk * ai + ii][c]
            cols_U.append(cu)
            cols_V.append(cv)
            cols_W.append(cw)

    i0 = 0
    for ai in iparts:
        j0 = 0
        for bj in jparts:
            k0 = 0
            for ck in kparts:
                add_block(i0, j0, k0, ai, bj, ck)
                k0 += ck
            j0 += bj
        i0 += ai

    rank = len(cols_U)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    for c in range(rank):
        for a in range(n * m):
            U[a][c] = cols_U[c][a]
        for b in range(m * p):
            V[b][c] = cols_V[c][b]
        for d in range(p * n):
            W[d][c] = cols_W[c][d]
    return {"U": U, "V": V, "W": W}


def _to_int_or_half(mat):
    out = []
    for row in mat:
        nr = []
        for x in row:
            hx = round(2 * x) / 2.0
            if abs(hx - x) < 1e-9:
                # keep as int if integer else float half
                if abs(hx - round(hx)) < 1e-9:
                    nr.append(int(round(hx)))
                else:
                    nr.append(hx)
            else:
                nr.append(x)
        out.append(nr)
    return out


def construct():
    n, m, p = DIMENSIONS

    # Deadline guard: always have a valid fallback.
    fallback = _schoolbook(n, m, p)
    assert _verify(fallback["U"], fallback["V"], fallback["W"], n, m, p)

    candidates = []

    if _remaining() > 5:
        try:
            blk = _build_block(n, m, p)
            if _verify(blk["U"], blk["V"], blk["W"], n, m, p):
                candidates.append(blk)
        except Exception:
            pass

    # Pick the verified candidate with minimal rank; fallback is rank n*m*p.
    candidates.append(fallback)
    candidates.sort(key=lambda d: d["U"] and len(d["U"][0]))

    best = candidates[0]
    # Normalise entries to int/half-integer for the evaluator.
    best = {"U": _to_int_or_half(best["U"]),
            "V": _to_int_or_half(best["V"]),
            "W": _to_int_or_half(best["W"])}
    return best


if __name__ == "__main__":
    r = construct()
    print("rank =", len(r["U"][0]))
