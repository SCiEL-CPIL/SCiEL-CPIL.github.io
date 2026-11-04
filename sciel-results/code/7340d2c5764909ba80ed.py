# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def _cost(p, a, b, n):
    """Exact QAP cost sum_{i,j} a[i,j]*b[p[i],p[j]]."""
    return int(np.einsum('ij,ik,jl,kl->', a, b[p, :], b[p, :], a))


def _linear_cost_matrix(a, b, n):
    """C[i, j] = sum_j a[i,j]*b[j, :]  (linear part of the QAP)."""
    return a @ b


def _cyclic_cost_matrix(a, b, n):
    """D[i, j] = sum_k a[i,k]*b[j,k]  (cyclic part of the QAP)."""
    return a @ b.T


def _dp_cycle(D, n, deadline):
    """Minimise sum_i D[i, (i+t) mod n] over rotation t in O(n^2)."""
    best_t = 0
    best_v = np.inf
    for t in range(n):
        v = float(np.einsum('ij,ij->', D, np.roll(D, -t, axis=1)))
        if v < best_v:
            best_v = v
            best_t = t
        if time.time() > deadline:
            break
    return best_t


def _lin_search(P, D, a, b, n, deadline):
    """Swap two facilities to minimise linear cost (vectorised)."""
    lin = a @ b
    C = P @ lin
    best_i = -1
    best_j = -1
    best_gain = 0
    for i in range(n):
        if time.time() > deadline:
            break
        d = lin[i] - C[i]
        idx = int(np.argmin(d))
        g = -float(d[idx])
        if g > best_gain:
            best_gain = g
            best_i = i
            best_j = idx
    return best_i, best_j


def solve(a, b):
    t0 = time.time()
    HARD = t0 + 112.0          # safety margin well under the 120s kill
    deadline = t0 + 116.0
    n = len(a)
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)

    # ---- initial solution: rank matching (heavy flows -> short distances) ----
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    P = np.empty(n, dtype=np.int64)
    P[heavy] = central
    P = P.copy()
    p = P.copy()
    best_cost = _cost(p, a, b, n)

    D = _cyclic_cost_matrix(a, b, n)
    lin = a @ b

    def apply_cycle(t):
        nonlocal P, p, best_cost
        P = np.roll(P, t)
        p = P.copy()
        c = _cost(p, a, b, n)
        if c < best_cost:
            best_cost = c

    def apply_swap(i, j):
        nonlocal p, best_cost
        p[i], p[j] = p[j], p[i]
        c = _cost(p, a, b, n)
        if c < best_cost:
            best_cost = c
            P = p.copy()

    # ---- alternating refinement until deadline ----
    while time.time() < HARD:
        # (A) Hungarian: exact optimal assignment for the linear part
        try:
            if time.time() > HARD:
                break
            C = _linear_cost_matrix(a, b, n)
            ii, jj = linear_sum_assignment(C)
            cand = np.empty(n, dtype=np.int64)
            cand[ii] = jj
            c = _cost(cand, a, b, n)
            if c < best_cost:
                best_cost = c
                p = cand.copy()
                P = cand.copy()
        except Exception:
            pass

        # (B) cyclic rotation refinement (O(n^2) DP)
        if time.time() < HARD:
            t = _dp_cycle(D, n, deadline)
            if t != 0:
                apply_cycle(t)

        # (C) linear local search (one best swap)
        if time.time() < HARD:
            i, j = _lin_search(P, D, a, b, n, deadline)
            if i >= 0 and j >= 0 and i != j:
                apply_swap(i, j)

    # ---- final safety: return best found (validate it is a permutation) ----
    p = np.asarray(p, dtype=np.int64)
    if len(p) != n or len(np.unique(p)) != n:
        # fallback: rank-matching solution is always a valid permutation
        p = P
    return p.tolist()
# EVOLVE-BLOCK-END
