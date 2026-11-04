"""QAPLIB wil100 solver: rank-matching start + 2-opt local search with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _total_cost(a, b, p):
    """Exact integer QAP cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j):
    """Cost change of swapping facilities i and j (delta = new - old).

    Only rows/cols i and j of the assignment matrix change, so the change
    is confined to a 2x2 block:
      old = a[i,i]b[pi,pi] + a[i,j]b[pi,pj] + a[j,i]b[pj,pi] + a[j,j]b[pj,pj]
      new = a[i,i]b[pj,pj] + a[i,j]b[pj,pi] + a[j,i]b[pi,pj] + a[j,j]b[pi,pi]
    """
    pi = p[i]
    pj = p[j]
    aii = a[i, i]
    aij = a[i, j]
    aji = a[j, i]
    ajj = a[j, j]
    bipi = b[pi, pi]
    bipj = b[pi, pj]
    bjpi = b[pj, pi]
    bjbj = b[pj, pj]
    old = aii * bipi + aij * bipj + aji * bjpi + ajj * bjbj
    new = aii * bjbj + aij * bjpi + aji * bipj + ajj * bipi
    return new - old


def solve(a, b):
    start = time.perf_counter()
    deadline = start + 140.0  # safety margin under the 150s kill

    n = a.shape[0]
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)

    # --- Initial assignment: rank-matching with variance tie-break ---
    # Heavy-flow facilities (large row sum) go to central locations (small row sum).
    flow = a.sum(axis=1)
    dist = b.sum(axis=1)
    # secondary key: flow variance (more spread -> prefer more central)
    flow_var = (a ** 2).sum(axis=1)
    dist_var = (b ** 2).sum(axis=1)

    heavy = np.lexsort((-flow_var, -flow))      # facilities: heaviest flow first
    central = np.lexsort((dist_var, dist))       # locations: most central first
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central

    cur = _total_cost(a, b, p)
    best = cur
    best_p = p.copy()

    # --- 2-opt first-improvement local search with deadline guard ---
    improved = True
    while improved and (time.perf_counter() < deadline):
        improved = False
        for i in range(n - 1):
            if time.perf_counter() > deadline:
                break
            pi = p[i]
            for j in range(i + 1, n):
                pj = p[j]
                d = _swap_delta(a, b, p, i, j)
                if d < 0:
                    # accept swap
                    p[i], p[j] = pj, pi
                    cur += d
                    improved = True
                    if cur < best:
                        best = cur
                        best_p = p.copy()
                    # after a swap, pi changed; recompute for next j
                    pi = p[i]

    # Final safety: return the best assignment found.
    if best < cur:
        return best_p.tolist()
    return p.tolist()
# EVOLVE-BLOCK-END
