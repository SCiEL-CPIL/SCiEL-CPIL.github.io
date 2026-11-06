"""Minimum maximal no-isosceles set via vectorized local search + coverage-weighted greedy."""
import time
import random
import numpy as np

N = 16
DEADLINE = 320.0  # safety margin under 330s kill

# Precompute all grid points and their squared-distance structure
_ALL = np.array([(x, y) for y in range(N) for x in range(N)], dtype=np.int32)  # (256,2)
# D2[i,j] = squared distance between point i and point j
_DX = _ALL[:, 0:1] - _ALL[:, 0:1].T
_DY = _ALL[:, 1:2] - _ALL[:, 1:2].T
D2 = _DX * _DX + _DY * _DY  # (256,256)


def _is_no_isosceles(idx):
    """idx: 1-D array of point indices. Return True if no three form isosceles."""
    k = len(idx)
    if k < 3:
        return True
    D = D2[np.ix_(idx, idx)]  # (k,k)
    # For each pair (i,j), check if any k exists with D[i,k]==D[j,k] or D[i,k]==D[i,j] or D[j,k]==D[i,j]
    # Vectorized: for each pair (i,j), the set of k that form isosceles:
    #   D[i,k] == D[j,k]  (k equidistant from i and j)
    #   D[i,k] == D[i,j]  (k equidistant from i as j is)
    #   D[j,k] == D[i,j]  (k equidistant from j as i is)
    # We need: for ALL pairs (i,j), there is NO k (k != i, k != j) satisfying any of these.
    # Build a (k,k) boolean matrix "bad[i,j]" = True if pair (i,j) is "witnessed" by some k.
    # bad[i,j] = exists k != i,j : D[i,k]==D[j,k] OR D[i,k]==D[i,j] OR D[j,k]==D[i,j]
    # We can compute for each pair (i,j):
    #   eq_ij = (D[i,:] == D[j,:])  -> k equidistant
    #   eq_i  = (D[i,:] == D[i,j])
    #   eq_j  = (D[j,:] == D[i,j])
    #   bad_k = eq_ij | eq_i | eq_j ; exclude k==i and k==j
    # Let's do it with broadcasting over pairs. k is up to ~30, so k^2 pairs, each O(k) -> O(k^3) ~ 27000, fine.
    # But we want speed. Let's do: for each i, for each j>i, check.
    # Use a loop over pairs but vectorize over k. k is small so this is fast enough.
    for i in range(k):
        for j in range(i + 1, k):
            dij = D[i, j]
            # k equidistant from i and j
            eq_ij = (D[i, :] == D[j, :])
            # k such that D[i,k]==dij
            eq_i = (D[i, :] == dij)
            eq_j = (D[j, :] == dij)
            bad = eq_ij | eq_i | eq_j
            bad[i] = False
            bad[j] = False
            if bad.any():
                return False
    return True


def _can_add_vec(pidx, idx):
    """Return True if adding point pidx to set idx keeps no-isosceles."""
    newidx = np.concatenate([idx, [pidx]])
    return _is_no_isosceles(newidx)


def _coverage(idx):
    """For each grid point not in idx, count how many pairs in idx would be 'witnessed' if added.
    Higher coverage = more 'blocked' = more valuable to add. Returns array of 256 floats."""
    k = len(idx)
    if k < 2:
        return np.zeros(256, dtype=np.float32)
    D = D2[np.ix_(idx, idx)]
    cov = np.zeros(256, dtype=np.float32)
    # For each candidate point c not in idx, adding c creates isosceles with pair (i,j) if
    # D2[c,i]==D2[c,j] or D2[c,i]==D2[i,j] or D2[c,j]==D2[i,j]
    # We want to count, for each c, the number of pairs (i,j) in idx that c would form isosceles with.
    # Vectorize: for each c, compute distances to all idx points.
    for c in range(256):
        if c in idx:
            continue
        dc = D2[c, idx]  # (k,)
        # For each pair (i,j): isosceles if dc[i]==dc[j] or dc[i]==D[i,j] or dc[j]==D[i,j]
        # Build (k,k) matrix of "would be isosceles"
        eq_dc = (dc[:, None] == dc[None, :])
        eq_i = (dc[:, None] == D)
        eq_j = (dc[None, :] == D)
        bad = eq_dc | eq_i | eq_j
        np.fill_diagonal(bad, False)
        cov[c] = bad.sum()
    return cov


def _greedy_coverage(idx, order):
    """Starting from idx, greedily add points in 'order' if possible. Returns new idx array."""
    current = list(idx)
    in_set = set(current)
    for p in order:
        if p in in_set:
            continue
        if _can_add_vec(p, np.array(current, dtype=np.int64)):
            current.append(p)
            in_set.add(p)
    return np.array(current, dtype=np.int64)


def _reduce_minimal(idx):
    """Try to remove each point and refill; keep if smaller."""
    current = list(idx)
    improved = True
    while improved:
        improved = False
        for i in range(len(current)):
            without = current[:i] + current[i + 1:]
            # Refill greedily from all points not in without, in a fixed order
            in_set = set(without)
            order = [p for p in range(256) if p not in in_set]
            newset = list(without)
            for p in order:
                if _can_add_vec(p, np.array(newset, dtype=np.int64)):
                    newset.append(p)
            if len(newset) < len(current):
                current = newset
                improved = True
                break
    return np.array(current, dtype=np.int64)


def construct():
    t0 = time.time()
    rng = random.Random(12345)
    best = None
    best_size = None

    # Phase 1: many greedy restarts with coverage-weighted ordering
    for restart in range(2000):
        if time.time() - t0 > DEADLINE:
            break
        # Random starting seed: a few random points that form no-isosceles
        seed_pts = []
        for _ in range(rng.randint(2, 6)):
            p = rng.randint(0, 255)
            if p not in seed_pts and _can_add_vec(p, np.array(seed_pts, dtype=np.int64) if seed_pts else np.array([], dtype=np.int64)):
                seed_pts.append(p)
        # Build coverage-weighted order
        order = list(range(256))
        rng.shuffle(order)
        # Compute coverage for ordering bias
        cov = _coverage(np.array(seed_pts, dtype=np.int64))
        # Sort by coverage descending, but add noise
        order.sort(key=lambda c: -(cov[c] + rng.random() * 0.5))
        # Greedy fill
        cand = _greedy_coverage(seed_pts, order)
        # Reduce to minimal
        cand = _reduce_minimal(cand)
        if best is None or len(cand) < best_size:
            best = cand
            best_size = len(cand)

    # Phase 2: local search on best: random removal + refill
    if best is not None:
        current = list(best)
        for step in range(5000):
            if time.time() - t0 > DEADLINE:
                break
            i = rng.randint(0, len(current) - 1)
            without = current[:i] + current[i + 1:]
            in_set = set(without)
            order = [p for p in range(256) if p not in in_set]
            rng.shuffle(order)
            newset = list(without)
            for p in order:
                if _can_add_vec(p, np.array(newset, dtype=np.int64)):
                    newset.append(p)
            if len(newset) < len(current):
                current = newset
                if best is None or len(current) < best_size:
                    best = np.array(current, dtype=np.int64)
                    best_size = len(current)

    if best is None:
        return [(0, 0)]
    return [(int(_ALL[i, 0]), int(_ALL[i, 1])) for i in best]


# Safety wrapper
_orig_construct = construct


def construct():
    try:
        res = _orig_construct()
        if res is None:
            return [(0, 0)]
        return res
    except Exception:
        return [(0, 0)]
