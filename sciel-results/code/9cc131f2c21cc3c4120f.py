"""Schoolbook base with bounded exact rank-reduction search."""
import time
import random

DIMENSIONS = [3, 4, 5]


def _verify(U, V, W, n, m, p, rank):
    """Exact verification: sum_r U[a,r]V[b,r]W[c,r] == 1 iff i_a==i_c and j_a==j_b."""
    for a in range(n * m):
        i_a = a // m
        j_a = a % m
        row_u = U[a]
        for b in range(m * p):
            j_b = b // p
            k_b = b % p
            row_v = V[b]
            for c in range(p * n):
                k_c = c // n
                i_c = c % n
                expected = 1 if (i_a == i_c and j_a == j_b) else 0
                total = 0
                for r in range(rank):
                    total += row_u[r] * row_v[r] * W[c][r]
                if total != expected:
                    return False
    return True


def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    return U, V, W, rank


def construct():
    deadline = time.time() + 110.0
    n, m, p = 3, 4, 5

    U, V, W, rank = _build_schoolbook(n, m, p)
    if not _verify(U, V, W, n, m, p, rank):
        # Should never happen; schoolbook is correct by construction.
        raise RuntimeError("schoolbook failed verification")
    best = (U, V, W, rank)

    # Bounded exact search for rank reductions.
    # Strategy: try to merge two terms r1, r2 into a single term when their
    # outer-product supports allow a linear combination that reproduces both.
    # A sufficient (and easy) condition: the two terms are identical up to a
    # constant scaling that preserves integer/half-integer entries, i.e. one
    # is a scalar multiple of the other with scalar in {1, -1, 1/2, 2, ...}.
    # We attempt merges, verify, and keep the best valid result.

    def try_merge(Uc, Vc, Wc, rc, r1, r2):
        # Look for scalars a,b (in {1,-1,1/2,2,1/4,4}) such that
        #   a*col(r1) + b*col(r2) == 0  in each of U,V,W (i.e. they are dependent)
        # and the merged term equals a*col(r1) (== -b*col(r2)).
        # Simpler: only merge if col(r1) == col(r2) exactly (a=1,b=-1), then
        # the merged column is col(r1) and we drop one rank.
        for mat in (Uc, Vc, Wc):
            c1 = [mat[x][r1] for x in range(len(mat))]
            c2 = [mat[x][r2] for x in range(len(mat))]
            if c1 != c2:
                return None
        # Identical columns: replace both by their common value in r1, zero r2.
        Uc2 = [row[:] for row in Uc]
        Vc2 = [row[:] for row in Vc]
        Wc2 = [row[:] for row in Wc]
        for x in range(len(Uc2)):
            Uc2[x][r2] = 0
        for x in range(len(Vc2)):
            Vc2[x][r2] = 0
        for x in range(len(Wc2)):
            Wc2[x][r2] = 0
        return Uc2, Vc2, Wc2, rc - 1

    # Remove any all-zero columns (none in schoolbook, but safe).
    def drop_zero_cols(Uc, Vc, Wc, rc):
        keep = []
        for r in range(rc):
            z = all(Uc[x][r] == 0 for x in range(len(Uc))) and \
                all(Vc[x][r] == 0 for x in range(len(Vc))) and \
                all(Wc[x][r] == 0 for x in range(len(Wc)))
            if not z:
                keep.append(r)
        if len(keep) == rc:
            return Uc, Vc, Wc, rc
        Uc2 = [[Uc[x][r] for r in keep] for x in range(len(Uc))]
        Vc2 = [[Vc[x][r] for r in keep] for x in range(len(Vc))]
        Wc2 = [[Wc[x][r] for r in keep] for x in range(len(Wc))]
        return Uc2, Vc2, Wc2, len(keep)

    U, V, W, rank = drop_zero_cols(U, V, W, rank)

    # Attempt merges in a fixed order with a hard time guard.
    if time.time() < deadline - 5.0:
        merged = False
        for r1 in range(rank):
            if time.time() > deadline - 5.0:
                break
            for r2 in range(r1 + 1, rank):
                if time.time() > deadline - 5.0:
                    break
                res = try_merge(U, V, W, rank, r1, r2)
                if res is None:
                    continue
                U2, V2, W2, r2k = res
                if _verify(U2, V2, W2, n, m, p, r2k):
                    U, V, W, rank = drop_zero_cols(U2, V2, W2, r2k)
                    merged = True
                    break
            if merged:
                break

    # Keep the best (lowest rank) valid result found; schoolbook is guaranteed valid.
    if rank < best[3] and _verify(U, V, W, n, m, p, rank):
        best = (U, V, W, rank)

    U, V, W, rank = best
    return {"U": U, "V": V, "W": W}
