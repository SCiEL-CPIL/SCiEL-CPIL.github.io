"""Kissing configuration in R^11 via large sparse candidate pool + greedy selection."""
import time
import itertools
import numpy as np

D = 11


def construct():
    deadline = time.time() + 140.0  # safety margin vs 150s kill

    # ---- Build a large candidate pool of sparse integer vectors ----
    # Vectors with support s (1..6) and coefficients in {-2,-1,1,2}.
    # A generous pool lets the greedy selector find a much larger valid set
    # than the 220-vector seed (support-2, coeff ±1) could.
    coeffs = (-2, -1, 1, 2)
    pool = []
    for s in range(1, 7):
        for idx in itertools.combinations(range(D), s):
            for vals in itertools.product(coeffs, repeat=s):
                v = np.zeros(D, dtype=np.int64)
                v[list(idx)] = vals
                pool.append(v)
    pool = np.asarray(pool, dtype=np.int64)  # shape (N, 11)

    # Precompute per-vector squared norms.
    norms = (pool * pool).sum(axis=1)  # shape (N,)

    # ---- Greedy selection ----
    # Invariant: for every pair (i,j) in selected,
    #   ||v_i - v_j||^2 >= max(norm_i, norm_j).
    # Expanding this to: ||v_i - v_j||^2 >= max over selected norms
    # only tightens the constraint, so validity is preserved.
    selected = []
    max_norm = 0
    # Sort by descending norm so early (high-norm) vectors are placed first;
    # this tends to lock in a large max-norm early, giving the greedy a
    # well-defined growing threshold.
    order = np.argsort(-norms)

    for idx in order:
        if time.time() >= deadline:
            break
        if not selected:
            selected.append(int(idx))
            max_norm = int(norms[idx])
            continue
        v = pool[idx]
        n = int(norms[idx])
        # Pairwise squared distances to all selected vectors (vectorized).
        diffs = pool[selected] - v
        dists = (diffs * diffs).sum(axis=1)
        if dists.min() >= max(max_norm, n):
            selected.append(int(idx))
            if n > max_norm:
                max_norm = n

    if not selected:
        # Fallback: a single nonzero vector is always a valid configuration.
        selected = [int(order[0])]

    result = np.asarray(pool[selected], dtype=np.int64)

    # ---- Final safety validation (cheap) ----
    k = len(result)
    if k > 1:
        ns = (result * result).sum(axis=1)
        mm = int(ns.max())
        ok = True
        for a in range(k):
            d2 = ((result[a] - result) ** 2).sum(axis=1)
            if d2.min() < mm:
                ok = False
                break
        if not ok:
            # Should never trigger; fall back to a trivially valid single vector.
            result = result[:1]

    return result
