"""Fresh design: bounded exact search for a low-rank decomposition of the
2x5 by 5x5 matrix multiplication tensor, with an exact schoolbook fallback."""
import time

DIMENSIONS = [2, 5, 5]

_DEADLINE = 140.0  # safety margin below the 150s wall-clock kill


def _remaining():
    return _DEADLINE - time.time()


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


def _matmul(A, B):
    # A: r x n, B: n x c  -> r x c
    r = len(A)
    n = len(A[0])
    c = len(B[0])
    return [[sum(A[i][t] * B[t][j] for t in range(n)) for j in range(c)] for i in range(r)]


def _verify(n, m, p, U, V, W):
    # Build T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]; must equal 1.
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True


def _rank_of_matrix(M):
    """Exact integer rank via fraction-free Gaussian elimination."""
    if not M:
        return 0
    r = len(M)
    c = len(M[0])
    A = [row[:] for row in M]
    rank = 0
    for col in range(c):
        if rank >= r:
            break
        # find pivot
        piv = None
        for i in range(rank, r):
            if A[i][col] != 0:
                piv = i
                break
        if piv is None:
            continue
        A[rank], A[piv] = A[piv], A[rank]
        pv = A[rank][col]
        for i in range(rank + 1, r):
            f = A[i][col]
            if f == 0:
                continue
            for j in range(col, c):
                A[i][j] = A[i][j] * pv - f * A[rank][j]
        rank += 1
    return rank


def _construct():
    n, m, p = DIMENSIONS
    # Fallback first (always valid).
    best = _schoolbook(n, m, p)

    # Try to find a strictly lower-rank exact decomposition by a bounded,
    # deterministic search over small integer coefficient sets. The search is
    # cheap per candidate because we only accept rank < current best.
    import itertools

    target_best_rank = len(best["U"][0])
    # Candidate ranks to attempt (must be < current best to improve).
    # Keep the search tiny and fully bounded so it never approaches the deadline.
    max_rank_to_try = target_best_rank - 1
    # A rank-1 term is a product of three vectors; we attempt a small pool of
    # structured terms. This is a heuristic that will not generally succeed for
    # this tensor, but it is exact and safe; failure simply keeps the fallback.
    # We cap total candidates so runtime is negligible.
    limit = 0
    max_candidates = 200
    for cand_rank in range(max_rank_to_try, 0, -1):
        if limit >= max_candidates:
            break
        # Attempt: greedily pick terms that cover uncovered (i,j,k) entries.
        # Each term c has (u_c, v_c, w_c) with u_c[i*m+j], v_c[j*p+k], w_c[k*n+i].
        # For a valid term, the set of (i,j,k) it touches must have consistent
        # product == 1. We use unit terms only (u,v,w in {0,1}) to stay exact.
        # A unit term covers (i,j,k) iff u[i*m+j]=v[j*p+k]=w[k*n+i]=1, and the
        # sum over terms of that product must equal 1 for every (i,j,k).
        # Greedy set cover over unit terms:
        total = n * m * p
        covered = [0] * total
        used = []
        idx = 0
        while idx < total and len(used) < cand_rank and limit < max_candidates:
            i, j, k = divmod(idx, 1) if False else (idx // (m * p), (idx // p) % m, idx % p)
            if covered[idx] != 0:
                idx += 1
                continue
            # Try a unit term that covers exactly (i,j,k) and no other uncovered
            # entry to avoid double counting.
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            u[i * m + j] = 1
            v[j * p + k] = 1
            w[k * n + i] = 1
            # Count how many entries this unit term covers.
            cnt = 0
            for ii in range(n):
                for jj in range(m):
                    if u[ii * m + jj] == 0:
                        continue
                    for kk in range(p):
                        if v[jj * p + kk] == 0 or w[kk * n + ii] == 0:
                            continue
                        cnt += 1
            if cnt == 1:
                covered[idx] = 1
                used.append((u, v, w))
            idx += 1
        limit += 1
        if len(used) == total:
            # Build matrices
            r = len(used)
            U = [[0] * r for _ in range(n * m)]
            V = [[0] * r for _ in range(m * p)]
            W = [[0] * r for _ in range(p * n)]
            for c, (u, v, w) in enumerate(used):
                for a in range(n * m):
                    U[a][c] = u[a]
                for a in range(m * p):
                    V[a][c] = v[a]
                for a in range(p * n):
                    W[a][c] = w[a]
            if _verify(n, m, p, U, V, W) and r < len(best["U"][0]):
                best = {"U": U, "V": V, "W": W}
                return best
    return best


# EVOLVE-BLOCK-START
def construct():
    # Deterministic, bounded, exact search with a guaranteed valid fallback.
    try:
        return _construct()
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
