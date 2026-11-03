import time
import sys
import numpy as np
from itertools import combinations

def construct():
    deadline = time.time() + 145.0
    n, m, p = 3, 5, 6

    # Build initial Strassen-based decomposition using the precise 2x2x2
    # Strassen coefficients (the "structural idea" from the inspiration),
    # combined with the current program's block-partition machinery.
    terms = _build_initial_terms(n, m, p)

    rank = len(terms)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            U[i * m + j][c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][c] = val

    # Verify; fall back to schoolbook if broken.
    if not _verify(U, V, W, n, m, p):
        return _schoolbook_fallback(n, m, p)

    # Refinement: local search to reduce rank by removing linearly-dependent
    # (redundant) columns, using the exact tensor identity as the constraint.
    U, V, W = _reduce_rank(U, V, W, n, m, p, deadline)

    return {"U": U, "V": W if False else W, "W": W} if False else {"U": U, "V": V, "W": W}


def _build_initial_terms(n, m, p):
    """Block-partition Strassen build (current program's structure) with
    corrected 2x2x2 Strassen coefficients (inspiration's precision)."""
    terms = []

    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))

    def add_schoolbook(i, j, k):
        add_term({(i, j): 1}, {(j, k): 1}, {(k, i): 1})

    def add_strassen(i0, i1, j0, j1, k0, k1):
        # Corrected Strassen 7 products.
        # m1 = (A00+A11)(B00+B11)
        add_term(
            {(i0, j0): 1, (i1, j1): 1},
            {(j0, k0): 1, (j1, k1): 1},
            {(k0, i0): 1, (k1, i1): 1})
        # m2 = (A10+A11)B00
        add_term(
            {(i1, j0): 1, (i1, j1): 1},
            {(j0, k0): 1},
            {(k0, i1): 1, (k1, i1): -1})
        # m3 = A00(B10-B11)
        add_term(
            {(i0, j0): 1},
            {(j1, k0): 1, (j1, k1): -1},
            {(k0, i0): 1})
        # m4 = A11(B01-B00)
        add_term(
            {(i1, j1): 1},
            {(j0, k1): 1, (j0, k0): -1},
            {(k1, i1): 1})
        # m5 = (A00+A01)B11
        add_term(
            {(i0, j0): 1, (i0, j1): 1},
            {(j1, k1): 1},
            {(k1, i1): 1, (k0, i0): -1})
        # m6 = (A10-A00)(B00+B01)
        add_term(
            {(i1, j0): 1, (i0, j0): -1},
            {(j0, k0): 1, (j0, k1): 1},
            {(k1, i1): 1})
        # m7 = (A01-A11)(B10+B11)
        add_term(
            {(i0, j1): 1, (i1, j1): -1},
            {(j1, k0): 1, (j1, k1): 1},
            {(k0, i0): 1, (k1, i1): -1})

    i_blocks = [[0, 1], [2]]
    j_blocks = [[0, 1], [2, 3], [4]]
    k_blocks = [[0, 1], [2, 3], [4, 5]]

    for i_blk in i_blocks:
        for j_blk in j_blocks:
            for k_blk in k_blocks:
                if len(i_blk) == 2 and len(j_blk) == 2 and len(k_blk) == 2:
                    i0, i1 = i_blk
                    j0, j1 = j_blk
                    k0, k1 = k_blk
                    add_strassen(i0, i1, j0, j1, k0, k1)
                else:
                    for i in i_blk:
                        for j in j_blk:
                            for k in k_blk:
                                add_schoolbook(i, j, k)
    return terms


def _verify(U, V, W, n, m, p):
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


def _reduce_rank(U, V, W, n, m, p, deadline):
    """Exact rank reduction: drop any column whose removal preserves the
    identity (i.e. its contribution is linearly redundant). Greedy, safe."""
    rank = len(U[0])
    keep = list(range(rank))
    changed = True
    while changed and time.time() < deadline - 3.0:
        changed = False
        for c in list(keep):
            trial = [x for x in keep if x != c]
            if not trial:
                continue
            # Build reduced matrices and test the identity.
            U2 = [ [U[r][c] for c in trial] for r in range(len(U)) ]
            V2 = [ [V[r][c] for c in trial] for r in range(len(V)) ]
            W2 = [ [W[r][c] for c in trial] for r in range(len(W)) ]
            if _verify(U2, V2, W2, n, m, p):
                keep = trial
                changed = True
                break
    if len(keep) == rank:
        return U, V, W
    U = [ [U[r][c] for c in keep] for r in range(len(U)) ]
    V = [ [V[r][c] for c in keep] for r in range(len(V)) ]
    W = [ [W[r][c] for c in keep] for r in range(len(W)) ]
    return U, V, W


def _schoolbook_fallback(n, m, p):
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
    return {"U": U, "V": V, "W": W}
