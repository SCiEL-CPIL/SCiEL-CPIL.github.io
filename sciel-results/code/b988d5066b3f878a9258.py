"""Matrix multiplication 3,4,5 — rank-minimal exact tensor decomposition.

We search for an exact decomposition of the matrix-multiplication tensor
T[i,j,k] = 1 (for all i in [0,n), j in [0,m), k in [0,p)) as
    T = sum_{c=1}^{rank} U[:,c] * V[:,c] * W[:,c]
with U of shape (n*m, rank), V of shape (m*p, rank), W of shape (p*n, rank),
using integer or half-integer (multiples of 0.5) entries.

Strategy: start from the schoolbook rank-60 decomposition, then attempt to
find a strictly lower-rank exact decomposition via randomized sparse-tensor
search in the doubled-integer space (so half-integers become integers).
Every candidate is verified exactly using Python integers (no floating point).
A deadline guard returns the best valid decomposition found before the limit.
"""

import time
import random
import itertools

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Exact verification that sum_c U[i*m+j, c]*V[j*p+k, c]*W[k*n+i, c] == 1
    for all (i,j,k). Returns True iff exact.
    Entries are assumed to be integers in the *doubled* space, i.e. the real
    value is entry/2. So the product of three real values is (a*b*c)/8, and
    we need sum_c (a*b*c)/8 == 1  <=>  sum_c a*b*c == 8.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for c in range(rank):
                    s += U[ui][c] * V[vj][c] * W[wk][c]
                if s != 8:
                    return False
    return True


def _to_half(U, V, W):
    """Convert doubled-integer matrices back to half-integer (float) form.
    The evaluator accepts integer/half-integer entries; we return floats that
    are exact multiples of 0.5.
    """
    def conv(M):
        return [[x / 2.0 for x in row] for row in M]
    return {"U": conv(U), "V": conv(V), "W": conv(W)}


def _schoolbook(n, m, p):
    """Schoolbook rank n*m*p decomposition in doubled-integer space.
    Real entries are 0.5, product 0.125 = 1/8, sum over 1 term = 1/8? No.
    We need each (i,j,k) to sum to 1. With one term per (i,j,k) and real
    value 1 on each of U,V,W, product = 1, sum = 1. In doubled space each
    entry is 2, product 8, sum 8. Good.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 2
                V[j * p + k][col] = 2
                W[k * n + i][col] = 2
                col += 1
    return U, V, W


def _random_sparse_term(n, m, p, rng, sparsity):
    """Generate a random sparse rank-1 term in doubled-integer space.
    Returns (U_col, V_col, W_col) as lists, each with mostly-zero small-integer
    entries. We bias toward small magnitudes (0, ±1, ±2) to keep products small.
    """
    nm, mp, pn = n * m, m * p, p * n
    vals = [0, 0, 0, 1, -1, 2, -2]

    def make(L, sp):
        row = [0] * L
        n_nonzero = max(1, int(L * sp))
        idx = rng.sample(range(L), min(n_nonzero, L))
        for ix in idx:
            row[ix] = rng.choice(vals)
        return row

    U_col = make(nm, sparsity)
    V_col = make(mp, sparsity)
    W_col = make(pn, sparsity)
    return U_col, V_col, W_col


def _try_rank(r, n, m, p, rng, deadline, iters):
    """Attempt to find an exact rank-r decomposition. Returns (U,V,W) in
    doubled-integer space or None. Uses randomized construction + local repair.
    """
    nm, mp, pn = n * m, m * p, p * n
    # We need sum_c U_i V_j W_k == 8 for all (i,j,k).
    # Parametrize: build r random sparse terms, then solve a *linear* correction.
    # Approach: fix U and V randomly (sparse), then W must satisfy
    #   sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 8  for all (i,j,k).
    # This is linear in W! For fixed U,V, the equation for each (i,j,k) is:
    #   sum_c A_{(i,j,k),c} * W_{(k*n+i),c} == 8
    # where A_{(i,j,k),c} = U[i*m+j,c]*V[j*p+k,c].
    # Unknowns are the W entries W[x,c] for x in [0,pn), c in [0,r).
    # Number of equations = n*m*p = 60, unknowns = p*n*r.
    # If p*n*r >= 60 we have a (possibly underdetermined) linear system.
    # We solve it over the rationals and check for a half-integer (doubled-integer)
    # solution.

    # Build the linear system in doubled-integer space.
    # Equation index e = (i*m+j)*p + k  (i in [0,n), j in [0,m), k in [0,p))
    # Unknown index u = x*r + c  where x = k*n + i in [0, p*n), c in [0,r)
    neq = n * m * p
    nunk = pn * r

    # We'll solve via Gaussian elimination over rationals using fractions,
    # but that's slow; instead we solve over reals with numpy and then check
    # if a nearby doubled-integer solution exists, then verify exactly.
    import numpy as np

    for it in range(iters):
        if time.time() > deadline:
            return None
        # Random sparse U, V
        U = [[0] * r for _ in range(nm)]
        V = [[0] * r for _ in range(mp)]
        sp = 0.25
        for x in range(nm):
            for c in range(r):
                if rng.random() < sp:
                    U[x][c] = rng.choice([1, -1, 2, -2])
        for x in range(mp):
            for c in range(r):
                if rng.random() < sp:
                    V[x][c] = rng.choice([1, -1, 2, -2])

        # Build matrix A (neq x nunk) and rhs (8 each)
        # A[e, u] where e=(i*m+j)*p+k, u=(k*n+i)*r + c
        A = np.zeros((neq, nunk), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    e = (i * m + j) * p + k
                    ui = i * m + j
                    vj = j * p + k
                    x = k * n + i
                    for c in range(r):
                        val = U[ui][c] * V[vj][c]
                        if val != 0:
                            A[e, x * r + c] = val
        b = np.full(neq, 8.0)

        # Solve least squares
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            continue
        # Round to nearest integer (doubled space)
        Wflat = np.round(sol).astype(int)
        # Check residual exactly
        W = [[0] * r for _ in range(pn)]
        for x in range(pn):
            for c in range(r):
                W[x][c] = int(Wflat[x * r + c])
        # Exact verify
        if _verify(U, V, W, n, m, p):
            return U, V, W
    return None


def _try_rank_fixed_UV(n, m, p, r, rng, deadline, iters):
    """Same as _try_rank but also tries a few fixed structured U,V patterns
    (e.g. identity-like) to improve the chance of a clean integer W.
    """
    nm, mp, pn = n * m, m * p, p * n
    neq = n * m * p
    nunk = pn * r
    import numpy as np
    b = np.full(neq, 8.0)

    patterns = []
    # Pattern 0: random (handled in loop below via flag)
    for it in range(iters):
        if time.time() > deadline:
            return None
        U = [[0] * r for _ in range(nm)]
        V = [[0] * r for _ in range(mp)]
        mode = rng.random()
        if mode < 0.5:
            sp = 0.2
            for x in range(nm):
                for c in range(r):
                    if rng.random() < sp:
                        U[x][c] = rng.choice([1, -1, 2, -2])
            for x in range(mp):
                for c in range(r):
                    if rng.random() < sp:
                        V[x][c] = rng.choice([1, -1, 2, -2])
        else:
            # Structured: make U and V mostly identity-like so A is well-conditioned
            for c in range(r):
                if c < nm:
                    U[c][c] = 2
                if c < mp:
                    V[c][c] = 2
                # add a little noise
                if rng.random() < 0.1:
                    U[rng.randrange(nm)][c] += rng.choice([1, -1])
                    V[rng.randrange(mp)][c] += rng.choice([1, -1])

        A = np.zeros((neq, nunk), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    e = (i * m + j) * p + k
                    ui = i * m + j
                    vj = j * p + k
                    x = k * n + i
                    for c in range(r):
                        val = U[ui][c] * V[vj][c]
                        if val != 0:
                            A[e, x * r + c] = val
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            continue
        Wflat = np.round(sol).astype(int)
        W = [[0] * r for _ in range(pn)]
        for x in range(pn):
            for c in range(r):
                W[x][c] = int(Wflat[x * r + c])
        if _verify(U, V, W, n, m, p):
            return U, V, W
    return None


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    # Best known valid decomposition (doubled-integer space)
    U0, V0, W0 = _schoolbook(n, m, p)
    best = (U0, V0, W0)
    best_rank = n * m * p

    rng = random.Random(12345)

    # Try to find lower-rank decompositions, descending from best_rank-1.
    # For each rank, give a time budget; stop early if we find one.
    # We focus effort on ranks just below 60 first (most likely to exist).
    for r in range(best_rank - 1, 15, -1):
        if time.time() > deadline - 20.0:
            break
        sub_deadline = min(deadline - 5.0, time.time() + 18.0)
        # Try fixed-UV structured approach
        res = _try_rank_fixed_UV(n, m, p, r, rng, sub_deadline, iters=2000)
        if res is not None:
            best = res
            best_rank = r
            continue
        # Try random approach
        sub_deadline2 = min(deadline - 5.0, time.time() + 12.0)
        res = _try_rank(n, m, p, r, rng, sub_deadline2, iters=1000)
        if res is not None:
            best = res
            best_rank = r
            continue

    U, V, W = best
    return _to_half(U, V, W)

# EVOLVE-BLOCK-END
