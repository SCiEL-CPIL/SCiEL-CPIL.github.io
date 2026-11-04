"""Stronger QAP solver: rank-matching start + Sinkhorn entropic relaxation + 2-opt polish."""
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def _cost(a, b, p):
    # exact integer cost for permutation p (facility i -> location p[i])
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _rank_match(a, b):
    # heavy flows -> short distances
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(len(a), dtype=np.int64)
    p[heavy] = central
    return p


def _sinkhorn_init(a, b, n, iters, t0, t1, seed=0):
    # Entropic relaxation: find doubly-stochastic X minimizing sum X_ij * C_ij
    # where C_ij = sum_k a[i,k]*b[j,k]  (facility i to location j contribution)
    C = a @ b  # C[i,j] = sum_k a[i,k] b[j,k]
    rng = np.random.default_rng(seed)
    # temperature annealing
    t = t0
    X = np.full((n, n), 1.0 / n)
    # log-sum-exp based Sinkhorn for numerical stability
    def sinkhorn(C, t, X0, iters):
        # minimize <X,C> s.t. X doubly stochastic, with entropic reg 1/t
        # use log-domain: K = -C/t
        K = -C / t
        # initialize with log of X0
        r = np.full(n, 0.0)
        c = np.full(n, 0.0)
        for _ in range(iters):
            # row update
            row = K + c[None, :]
            logsumrow = _logsumexp_axis(row, axis=1)  # shape (n,)
            r = -logsumrow
            # col update
            col = K + r[:, None]
            logsumcol = _logsumexp_axis(col, axis=0)  # shape (n,)
            c = -logsumcol
        X = np.exp(K + r[:, None] + c[None, :])
        return X

    for step in range(iters):
        frac = step / max(1, iters - 1)
        t = t0 + (t1 - t0) * frac
        X = sinkhorn(C, t, X, 40)
    return X


def _logsumexp_axis(x, axis):
    # numerically stable logsumexp along axis
    m = np.max(x, axis=axis, keepdims=True)
    out = np.log(np.sum(np.exp(x - m), axis=axis))
    out = out + np.squeeze(m, axis=axis)
    return out


def _round(X):
    # Hungarian rounding of the soft assignment matrix
    row, col = linear_sum_assignment(-X)
    p = np.empty(len(X), dtype=np.int64)
    p[row] = col
    return p


def _swap_delta(a, b, p, i, k):
    # delta in cost when swapping facilities i and k (locations p[i], p[k])
    li = p[i]
    lk = p[k]
    # cost terms involving i or k:
    # before: sum_j a[i,j] b[li, p[j]] + sum_j a[k,j] b[lk, p[j]]
    #        + sum_j a[j,i] b[p[j], li] + sum_j a[j,k] b[p[j], lk]
    #        + a[i,k] b[li, lk] + a[k,i] b[lk, li]
    # after:  same but i<->k locations swapped
    # Use precomputed row/col contributions.
    # Let R_i(l) = sum_j a[i,j] b[l, p[j]]  (row i placed at l)
    # Let Col_j(l) = sum_j a[j,i] b[p[j], l]
    # We'll compute delta directly.
    # Precompute for current p:
    #   A_i = sum_j a[i,j] b[li, p[j]]  (i's out-row at li)
    #   B_i = sum_j a[j,i] b[p[j], li]  (i's in-col at li)
    #   A_k = sum_j a[k,j] b[lk, p[j]]
    #   B_k = sum_j a[j,k] b[p[j], lk]
    #   cross_ik = a[i,k] b[li,lk] + a[k,i] b[lk,li]
    # after swap:
    #   i goes to lk, k goes to li
    #   A_i' = sum_j a[i,j] b[lk, p'[j]]  where p' differs only at i,k
    # This gets messy; compute via full row/col vectors.
    # Efficient: precompute for each facility f: outrow_f = a[f,:] (shape n)
    # and for each location l, the vector b[l, p[.]] etc.
    # We'll just compute the two full sums O(n) each.
    # Before contribution of {i,k}:
    before = 0
    # i out to all j at li
    before += np.dot(a[i, :], b[li, p])
    # k out to all j at lk
    before += np.dot(a[k, :], b[lk, p])
    # all j in to i at li
    before += np.dot(a[:, i], b[p, li])
    # all j in to k at lk
    before += np.dot(a[:, k], b[p, lk])
    # After: swap p[i]<->p[k]
    p2 = p.copy()
    p2[i], p2[k] = lk, li
    after = 0
    after += np.dot(a[i, :], b[lk, p2])
    after += np.dot(a[k, :], b[li, p2])
    after += np.dot(a[:, i], b[p2, lk])
    after += np.dot(a[:, k], b[p2, li])
    return after - before


def _two_opt(a, b, p, budget):
    n = len(p)
    current = _cost(a, b, p)
    start = time.time()
    improved = True
    while improved and (time.time() - start) < budget:
        improved = False
        # precompute helper vectors to make swap eval O(1)-ish
        # For each facility f and location l: out(f,l) = dot(a[f,:], b[l,p])
        # For each facility f and location l: inn(f,l) = dot(a[:,f], b[p,l])
        # out matrix: (n, n) ; inn matrix: (n, n)
        # out[f, l] = a[f,:] @ b[l, p]  = (a @ b) ... no, b[l,p] is permuted
        # Build Bp = b[:, p] (n x n): Bp[l, j] = b[l, p[j]]
        Bp = b[:, p]
        out = a @ Bp          # out[f, l] = sum_j a[f,j] b[l,p[j]]
        inn = a.T @ Bp.T      # inn[f, l] = sum_j a[j,f] b[p[j], l]
        # cross term between i,k at locations li,lk:
        # cross(i,k,li,lk) = a[i,k] b[li,lk] + a[k,i] b[lk,li]
        # cost of placing i at li and k at lk (with rest fixed) includes:
        #   out[i,li]+out[k,lk]+inn[i,li]+inn[k,lk] + cross(i,k,li,lk)
        # But out[i,li] already includes j=k term a[i,k] b[li,lk]; inn likewise.
        # To avoid double counting cross, define:
        #   base(i,li) = out[i,li]+inn[i,li]  (includes cross with all j incl. partner)
        # Then pair cost = base(i,li)+base(k,lk) - cross_double(i,k,li,lk)
        # where cross_double = a[i,k] b[li,lk] + a[k,i] b[lk,li] (counted twice in base)
        # So pair(i,k,li,lk) = base[i,li]+base[k,lk] - cross(i,k,li,lk)
        base = out + inn
        # cross matrix between facilities at given locations:
        # cross(i,k,li,lk) = a[i,k]*b[li,lk] + a[k,i]*b[lk,li]
        # For fixed li,lk define M[i,k] = a[i,k]*b[li,lk] + a[k,i]*b[lk,li]
        # We'll compute per candidate pair on the fly O(1).
        # Delta for swapping i,k:
        #   before_pair = base[i,li]+base[k,lk] - cross(i,k,li,lk)
        #   after_pair  = base[i,lk]+base[k,li] - cross(i,k,lk,li)
        #   delta = after_pair - before_pair
        for i in range(n):
            li = p[i]
            bi = base[i, li]
            for k in range(i + 1, n):
                lk = p[k]
                bk = base[k, lk]
                c_before = a[i, k] * b[li, lk] + a[k, i] * b[lk, li]
                c_after = a[i, k] * b[lk, li] + a[k, i] * b[li, lk]
                delta = (base[i, lk] + base[k, li] - c_after) - (bi + bk - c_before)
                if delta < 0:
                    p[i], p[k] = lk, li
                    current += delta
                    improved = True
                    # recompute base because p changed
                    Bp = b[:, p]
                    out = a @ Bp
                    inn = a.T @ Bp.T
                    base = out + inn
                    break
            if improved:
                continue
        # (simple first-improvement; recompute each pass)
    return p, current


def solve(a, b):
    n = len(a)
    deadline = 150.0
    t_start = time.time()
    safety = 10.0  # return this far before kill

    # 1) rank-matching start
    p = _rank_match(a, b)
    best_p = p.copy()
    best_cost = _cost(a, b, p)

    def remaining():
        return deadline - safety - (time.time() - t_start)

    # 2) Sinkhorn entropic relaxation -> soft assignment -> Hungarian round
    if remaining() > 5.0:
        try:
            X = _sinkhorn_init(a, b, n, iters=12, t0=1.0, t1=0.05, seed=1)
            p2 = _round(X)
            c2 = _cost(a, b, p2)
            if c2 < best_cost:
                best_p = p2.copy()
                best_cost = c2
        except Exception:
            pass

    # 3) 2-opt local search polish on the best so far
    if remaining() > 3.0:
        p = best_p.copy()
        budget = max(1.0, remaining() - 2.0)
        p, c = _two_opt(a, b, p, budget)
        if c < best_cost:
            best_p = p.copy()
            best_cost = c

    # final safety
    if time.time() - t_start > deadline - safety:
        return best_p.tolist()

    return best_p.tolist()
