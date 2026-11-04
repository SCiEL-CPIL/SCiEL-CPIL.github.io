"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k.
    Accepts lists or numpy arrays.
    """
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    rank = Ua.shape[1]
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr, optimize=True)
    return bool(np.allclose(T, 1.0, atol=tol))

def _schoolbook(n, m, p):
    """
    Schoolbook decomposition of rank n*m*p, returned as plain Python lists
    (integer entries). U[i*m+j,c], V[j*p+k,c], W[k*n+i,c].
    """
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
    return U, V, W

def _merge_candidate(U, V, W, n, m, p, r, seed, max_steps=20000):
    """
    Start from the schoolbook (rank n*m*p), then repeatedly try to merge two
    columns into one using the linear-dependency (rank-revealing) relation:
    a term (u1,v1,w1) is redundant when a linear combination of the remaining
    terms reproduces it. We search for a merge that keeps the tensor exact.

    A merge replaces columns a,b with a new column c such that the total
    contribution to every (i,j,k) is unchanged. We solve for the new column by
    requiring the removed pair's contribution to be reproduced by a scalar
    combination. Concretely we look for alpha,beta and a single column that
    equals alpha*col_a + beta*col_b on the U side while the product is preserved;
    this is only exact in special cases, so we verify after each candidate.
    """
    Ur = np.array(U, dtype=float)
    Vr = np.array(V, dtype=float)
    Wr = np.array(W, dtype=float)
    cur_rank = Ur.shape[1]
    if cur_rank <= r:
        return None

    def compute_T(Ur, Vr, Wr):
        rr = Ur.shape[1]
        return np.einsum('ijc,jkc,kic->ijk',
                         Ur.reshape(n, m, rr), Vr.reshape(m, p, rr),
                         Wr.reshape(p, n, rr), optimize=True)

    T_target = compute_T(Ur, Vr, Wr)  # should be all ones

    def exact(Ur, Vr, Wr, tol=1e-9):
        return np.allclose(compute_T(Ur, Vr, Wr), T_target, atol=tol)

    # Try to reduce one rank at a time by finding a redundant column.
    # A column c is redundant if its contribution can be expressed via others.
    # We approximate by: pick a pair (a,b), form a merged column as a
    # linear combination and check exactness.
    rng = random.Random(seed)
    step = 0
    while Ur.shape[1] > r and step < max_steps:
        step += 1
        rr = Ur.shape[1]
        a = rng.randrange(rr)
        b = rng.randrange(rr - 1)
        if b == a:
            b = (b + 1) % rr
        # candidate merged column: alpha*col_a + beta*col_b on each factor
        for (alpha, beta) in [(1.0, 0.0), (0.0, 1.0), (0.5, 0.5), (1.0, 1.0)]:
            Ur2 = Ur.copy()
            Vr2 = Vr.copy()
            Wr2 = Wr.copy()
            # replace column a with alpha*a+beta*b, then drop column b
            Ur2[:, a] = alpha * Ur2[:, a] + beta * Ur2[:, b]
            Vr2[:, a] = alpha * Vr2[:, a] + beta * Vr2[:, b]
            Wr2[:, a] = alpha * Wr2[:, a] + beta * Wr2[:, b]
            if a < b:
                Ur2 = np.delete(Ur2, b, axis=1)
                Vr2 = np.delete(Vr2, b, axis=1)
                Wr2 = np.delete(Wr2, b, axis=1)
            else:
                Ur2 = np.delete(Ur2, a, axis=1)
                Vr2 = np.delete(Vr2, a, axis=1)
                Wr2 = np.delete(Wr2, a, axis=1)
            if exact(Ur2, Vr2, Wr2):
                Ur, Vr, Wr = Ur2, Vr2, Wr2
                break
    if Ur.shape[1] <= r and exact(Ur, Vr, Wr):
        return Ur, Vr, Wr
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before the 150s kill

    # Guaranteed fallback: schoolbook of rank n*m*p (returned as lists).
    U, V, W = _schoolbook(n, m, p)
    best_result = {"U": U, "V": V, "W": W}

    # Fast self-check of the fallback.
    if not _verify(U, V, W, n, m, p):
        # Should never happen; keep the result regardless.
        pass

    # Focused search for a strictly lower-rank exact decomposition.
    # Rank-revealing / linear-dependency merge over the 125 standard terms.
    for target_rank in [n * m * p - 1]:
        if time.time() > deadline - 15:
            break
        for seed in range(4):
            if time.time() > deadline - 15:
                break
            res = _merge_candidate(U, V, W, n, m, p, target_rank, seed,
                                   max_steps=8000)
            if res is not None:
                Ur, Vr, Wr = res
                # Convert to plain lists with integer/half-integer entries.
                def to_lists(A):
                    return [[float(x) for x in row] for row in A]
                cand = {"U": to_lists(Ur), "V": to_lists(Vr), "W": to_lists(Wr)}
                if _verify(Ur, Vr, Wr, n, m, p):
                    best_result = cand
                    break
        if best_result["U"] is not U:
            break

    return best_result
# EVOLVE-BLOCK-END
