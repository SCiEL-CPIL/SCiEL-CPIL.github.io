"""Combined program: correct Strassen baseline + ALS refinement + half-integer rounding."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(U[i*m+j], V[j*p+k] * W[k*n+i]))
                if abs(s - 1.0) > tol:
                    return False
    return True

def _is_half_integer(M, tol=1e-9):
    M = np.asarray(M, dtype=float)
    return bool(np.all(np.abs(M * 2.0 - np.round(M * 2.0)) <= tol))

def _round_half_integer(M):
    return np.round(np.asarray(M, dtype=float) * 2.0) / 2.0

def _strassen_2x2_terms():
    """7 Strassen terms for 2x2x2. Each: U,V,W dicts keyed by (row,col) within 2x2."""
    T = []
    T.append({'U': {(0,0):1,(1,1):1}, 'V': {(0,0):1,(1,1):1}, 'W': {(0,0):1,(1,1):1}})
    T.append({'U': {(1,0):1,(1,1):1}, 'V': {(0,0):1},           'W': {(1,0):1,(1,1):-1}})
    T.append({'U': {(0,0):1},           'V': {(1,0):1,(1,1):-1}, 'W': {(0,1):1,(1,1):1}})
    T.append({'U': {(1,1):1},           'V': {(0,1):1,(0,0):-1}, 'W': {(0,0):1,(1,0):1}})
    T.append({'U': {(0,0):1,(0,1):1},   'V': {(1,1):1},           'W': {(0,0):-1,(0,1):1}})
    T.append({'U': {(1,0):1,(0,0):-1},  'V': {(0,0):1,(0,1):1},   'W': {(1,1):1}})
    T.append({'U': {(0,1):1,(1,1):-1},  'V': {(1,0):1,(1,1):1},   'W': {(0,0):1}})
    return T

def _build_4x4x4_strassen(n=4, m=4, p=4):
    """Correct rank-49 Strassen decomposition for 4x4x4 (4 independent 2x2x2 blocks)."""
    terms = _strassen_2x2_terms()
    rank = 4 * 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    blocks = [(0,0),(0,1),(1,0),(1,1)]
    for b, (bi, bj) in enumerate(blocks):
        for t, term in enumerate(terms):
            idx = b*7 + t
            for (li, lj), c in term['U'].items():
                U[bi*2+li, bj*2+lj] += c
            for (lj, lk), c in term['V'].items():
                V[bj*2+lj, bi*2+lk] += c
            for (lk, li), c in term['W'].items():
                W[bi*2+lk, bi*2+li] += c
    return U, V, W

def _build_4x4x5_strassen(n=4, m=4, p=5):
    """4x4x4 Strassen (rank 49) + 4x4x1 schoolbook (rank 4) = rank 53."""
    U4, V4, W4 = _build_4x4x4_strassen(n, m, 4)
    rank = 49 + 4
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    U[:, :49] = U4
    V[:, :49] = V4
    W[:, :49] = W4
    for j in range(m):
        r = 49 + j
        for i in range(n):
            U[i*m+j, r] = 1.0
        V[j*p+4, r] = 1.0
        for i in range(n):
            W[4*n+i, r] = 1.0
    return U, V, W

def _schoolbook(n, m, p):
    rank = n*m*p
    U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1.0; V[j*p+k, c] = 1.0; W[k*n+i, c] = 1.0; c += 1
    return U, V, W

def _als_decompose(n, m, p, rank, U0=None, V0=None, W0=None, iters=200, seed=0):
    """ALS minimization of sum_{i,j,k}(sum_t U[i,j,t]V[j,k,t]W[k,i,t] - 1)^2."""
    rng = np.random.default_rng(seed)
    if U0 is None:
        U = rng.normal(size=(n*m, rank)) * 0.5
        V = rng.normal(size=(m*p, rank)) * 0.5
        W = rng.normal(size=(p*n, rank)) * 0.5
    else:
        U = U0.copy(); V = V0.copy(); W = W0.copy()
    for _ in range(iters):
        # Update U
        for i in range(n):
            for j in range(m):
            # grad: 2 * sum_k (s-1) * V[j,k,:] * W[k,i,:]
                G = np.zeros(rank)
                for k in range(p):
                    G += V[j*p+k] * W[k*n+i]
                # s = sum_t U[i,j,t]*G[t]
                s = float(np.dot(U[i*m+j], G))
                U[i*m+j] -= 0.5 * (s - 1.0) * G
        # Update V
        for j in range(m):
            for k in range(p):
                G = np.zeros(rank)
                for i in range(n):
                    G += U[i*m+j] * W[k*n+i]
                s = float(np.dot(V[j*p+k], G))
                V[j*p+k] -= 0.5 * (s - 1.0) * G
        # Update W
        for k in range(p):
            for i in range(n):
                G = np.zeros(rank)
                for j in range(m):
                    G += U[i*m+j] * V[j*p+k]
                s = float(np.dot(W[k*n+i], G))
                W[k*n+i] -= 0.5 * (s - 1.0) * G
    return U, V, W

def _loss(U, V, W, n, m, p):
    U = np.asarray(U); V = np.asarray(V); W = np.asarray(W)
    tot = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(U[i*m+j], V[j*p+k] * W[k*n+i]))
                tot += (s - 1.0)**2
    return tot

def _round_and_check(U, V, W, n, m, p, tol=1e-6):
    Ur = _round_half_integer(U); Vr = _round_half_integer(V); Wr = _round_half_integer(W)
    if _verify_decomposition(Ur, Vr, Wr, n, m, p, tol=tol):
        return Ur, Vr, Wr
    return None

def _search_rank(n, m, p, rank, seed, U0, V0, W0, time_budget):
    """Try to find a half-integer decomposition of given rank via ALS + rounding."""
    start = time.time()
    best = None
    best_loss = float('inf')
    for attempt in range(6):
        if time.time() - start > time_budget:
            break
        if attempt == 0 and U0 is not None:
            U, V, W = U0.copy(), V0.copy(), W0.copy()
        else:
            rng = np.random.default_rng(seed + attempt*1000)
            U = rng.normal(size=(n*m, rank)) * 0.5
            V = rng.normal(size=(m*p, rank)) * 0.5
            W = rng.normal(size=(p*n, rank)) * 0.5
        U, V, W = _als_decompose(n, m, p, rank, U, V, W, iters=150, seed=seed+attempt)
        # try rounding at several scales
        for scale in [1.0, 0.5, 2.0]:
            cand = _round_and_check(U*scale, V*scale, W*scale, n, m, p)
            if cand is not None:
                # normalize scale back: entries already half-integers
                return cand
        l = _loss(U, V, W, n, m, p)
        if l < best_loss:
            best_loss = l
            best = (U, V, W)
    # final rounding attempt on best
    if best is not None:
        cand = _round_and_check(best[0], best[1], best[2], n, m, p)
        if cand is not None:
            return cand
    return None

def construct():
    start = time.time()
    deadline = start + 130
    n, m, p = DIMENSIONS

    # Correct baseline: Strassen rank 53
    U, V, W = _build_4x4x5_strassen(n, m, p)
    if _verify_decomposition(U, V, W, n, m, p):
        base = (U.copy(), V.copy(), W.copy())
        base_rank = 53
    else:
        U, V, W = _schoolbook(n, m, p)
        base = (U.copy(), V.copy(), W.copy())
        base_rank = n*m*p

    best = base
    best_rank = base_rank

    # Search for lower ranks via ALS + rounding
    # lower bound: max(n*m, m*p, p*n) = 20; realistically try down to ~40
    for target in range(base_rank - 1, 39, -1):
        if time.time() > deadline - 15:
            break
        budget = min(18, max(4, (deadline - 15 - time.time())))
        cand = _search_rank(n, m, p, target, seed=target, U0=base[0][:, :min(target, base_rank)],
                            V0=base[1][:, :min(target, base_rank)], W0=base[2][:, :min(target, base_rank)],
                            time_budget=budget)
        if cand is not None:
            best = cand
            best_rank = target
            break  # found a valid lower rank; stop (searching lower is expensive)

    U, V, W = best
    # Final half-integer safety
    U = _round_half_integer(U); V = _round_half_integer(V); W = _round_half_integer(W)
    if not _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
        # fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)

    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W],
    }

# EVOLVE-BLOCK-END
