"""Linearized greedy construction + 2-opt local search for QAPLIB tho150."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _cost(p, a, b, ap):
    # exact integer cost sum a[i,j] * b[p[i], p[j]]
    return int(np.sum(ap * b[np.ix_(p, p)]))


def _greedy_construct(a, b, n, sym):
    # Linearized greedy: place facilities one by one using a dynamic marginal-cost matrix.
    C = np.zeros((n, n), dtype=np.int64)
    p = np.empty(n, dtype=np.int64)
    placed = np.zeros(n, dtype=bool)
    # Precompute row sums of b for the diagonal-free term if needed.
    if sym:
        # marginal(i,j) = 2 * sum_{k placed} a[i,k] * b[j,p[k]]
        for step in range(n):
            # pick facility i with best (min) remaining marginal cost to an unplaced location
            # restrict C to unplaced facilities (rows) and unplaced locations (cols)
            sub = C.copy()
            sub[placed, :] = np.iinfo(np.int64).max
            sub[:, placed] = np.iinfo(np.int64).max
            flat = sub.ravel()
            idx = int(np.argmin(flat))
            i = idx // n
            j = idx % n
            p[i] = j
            placed[i] = True
            placed[j] = True
            # update C for remaining facilities: C[i2, j2] += 2*a[i2, i]*b[j2, j]
            if step + 1 < n:
                C += 2 * np.outer(a[:, i], b[:, j])
    else:
        # general: marginal(i,j) = sum_k a[i,k]*b[j,p[k]] + sum_k a[k,i]*b[p[k],j]
        for step in range(n):
            sub = C.copy()
            sub[placed, :] = np.iinfo(np.int64).max
            sub[:, placed] = np.iinfo(np.int64).max
            flat = sub.ravel()
            idx = int(np.argmin(flat))
            i = idx // n
            j = idx % n
            p[i] = j
            placed[i] = True
            placed[j] = True
            if step + 1 < n:
                C += np.outer(a[:, i], b[:, j]) + np.outer(a[i, :], b[j, :])
    return p


def _two_opt(p, a, b, n, deadline):
    # 2-opt: swap positions of two facilities; accept if cost decreases.
    # Precompute helper arrays for fast delta evaluation.
    # cost = sum a[i,j]*b[p[i],p[j]]. Swapping facilities x and y (locations px, py):
    # delta can be computed from the rows/cols. We recompute affected terms.
    best = _cost(p, a, b, None)
    improved = True
    # Use a best-improvement first pass, then first-improvement sweeps.
    order = np.random.default_rng(0).permutation(n)
    while improved and time.time() < deadline:
        improved = False
        for x in range(n):
            if time.time() >= deadline:
                break
            for y in range(x + 1, n):
                if time.time() >= deadline:
                    break
                # swap p[x] and p[y]
                px = p[x]; py = p[y]
                if px == py:
                    continue
                # compute cost delta
                # affected pairs: (x, all), (all, x), (y, all), (all, y)
                # but simplest correct: recompute contributions involving x or y
                # cost = sum_{i,j} a[i,j]*b[p[i],p[j]]
                # terms with i==x or j==x or i==y or j==y change
                # delta = new - old for those terms
                # old contribution:
                #   for i in {x,y}, j in all: a[i,j]*b[p[i],p[j]]  (rows x,y)
                #   for j in {x,y}, i in all (i not in {x,y}): a[i,j]*b[p[i],p[j]] (cols x,y)
                # new: same but with p[x],p[y] swapped
                old = (a[x, :] * b[px, p]).sum() + (a[y, :] * b[py, p]).sum() \
                    + (a[:, x] * b[p, px]).sum() + (a[:, y] * b[p, py]).sum()
                # subtract double counted (x,x),(x,y),(y,x),(y,y) counted twice
                old -= a[x, x] * b[px, px] + a[x, y] * b[px, py] \
                    + a[y, x] * b[py, px] + a[y, y] * b[py, py]
                p[x], p[y] = py, px
                new = (a[x, :] * b[px, p]).sum() + (a[y, :] * b[py, p]).sum() \
                    + (a[:, x] * b[p, px]).sum() + (a[:, y] * b[p, py]).sum()
                new -= a[x, x] * b[px, px] + a[x, y] * b[px, py] \
                    + a[y, x] * b[py, px] + a[y, y] * b[py, py]
                if new < old:
                    best += int(new - old)
                    improved = True
                else:
                    p[x], p[y] = px, py
    return best


def solve(a, b):
    n = len(a)
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    # Symmetry check (QAPLIB instances are symmetric)
    sym = bool(np.array_equal(a, a.T)) and bool(np.array_equal(b, b.T))

    # 1) Linearized greedy construction
    p = _greedy_construct(a, b, n, sym)

    # 2) 2-opt local search with deadline guard
    _two_opt(p, a, b, n, deadline)

    return p.tolist()
# EVOLVE-BLOCK-END
