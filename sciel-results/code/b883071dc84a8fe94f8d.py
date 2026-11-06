import time
import random
import numpy as np
from collections import Counter

N = 30
ALL_COORDS = np.array([(x, y) for x in range(-30, 31) for y in range(-30, 31)], dtype=np.int64)
MAX_TIME = 88.0  # safety margin before 90s kill

def _score_full(pts):
    """pts: (30,2) int array. Return max multiplicity of a positive doubled area."""
    # pairwise diffs
    p = pts.astype(np.int64)
    # d[i,j] = p[j]-p[i]
    d = p[None, :, :] - p[:, None, :]  # (30,30,2)
    # cross for triples (i,j,k): cross(d[i,j], d[i,k])
    # For fixed i: cross of d[i,j] and d[i,k] for all j,k
    # build matrix M_i[j,k] = d[i,j,0]*d[i,k,1] - d[i,j,1]*d[i,k,0]
    total = Counter()
    for i in range(N):
        di = d[i]  # (30,2)
        # cross matrix
        cx = di[:, 0:1] * di[None, :, 1] - di[:, 1:2] * di[None, :, 0]  # (30,30)
        a = np.abs(cx)
        # exclude zero and exclude degenerate (i in triple means j,k != i but cross with j=i is 0 anyway)
        a = a[a > 0]
        if a.size:
            vals, cnts = np.unique(a, return_counts=True)
            for v, c in zip(vals.tolist(), cnts.tolist()):
                total[v] += c
    if not total:
        return 0
    return max(total.values())

def _score_move(pts, idx, new_xy):
    """
    pts: (30,2) int array, idx: point to move, new_xy: (2,) int.
    Returns (delta_best, new_best) relative to current best multiplicity computed
    by caller? Simpler: recompute only affected triples.
    Affected triples: those containing idx.
    """
    p = pts
    old = p[idx]
    # triples containing idx: (idx, a, b) for a<b, a!=idx, b!=idx, plus (a,b,idx) same set
    # For each pair (a,b) with a!=idx, b!=idx, a<b: triangle (idx,a,b)
    # Also triangles (a,b,c) not containing idx are unchanged.
    # We need counts for areas before and after for affected triples only,
    # but the global max may come from unaffected triples.
    # So compute: unaffected_max = max multiplicity among areas from triples NOT containing idx.
    # affected: recompute.
    # This requires knowing per-area counts. Expensive to maintain. 
    # Simpler: full recompute but vectorized is fast enough.
    p2 = p.copy()
    p2[idx] = new_xy
    return _score_full(p2)

def _local_search(pts, time_limit):
    start = time.time()
    p = pts.astype(np.int64).copy()
    best = p.copy()
    best_score = _score_full(best)
    cur_score = best_score
    stuck = 0
    max_stuck = 400
    occupied = set(map(tuple, p.tolist()))

    while time.time() - start < time_limit:
        idx = random.randrange(N)
        old_xy = p[idx]
        # pick random empty
        while True:
            cand = tuple(ALL_COORDS[random.randrange(len(ALL_COORDS))])
            if cand not in occupied:
                break
        p[idx] = cand
        occupied.discard(tuple(old_xy))
        occupied.add(cand)
        ns = _score_full(p)
        if ns > cur_score:
            cur_score = ns
            stuck = 0
            if ns > best_score:
                best_score = ns
                best = p.copy()
        else:
            p[idx] = old_xy
            occupied.discard(cand)
            occupied.add(tuple(old_xy))
            stuck += 1
            if stuck > max_stuck:
                p = best.copy()
                occupied = set(map(tuple, p.tolist()))
                cur_score = best_score
                stuck = 0
    return best, best_score

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)

    candidates = []
    # grids
    candidates.append(np.array([(x, y) for x in range(5) for y in range(6)], dtype=np.int64))
    candidates.append(np.array([(x, y) for x in range(6) for y in range(5)], dtype=np.int64))
    candidates.append(np.array([(x, y) for x in range(3) for y in range(10)], dtype=np.int64))
    candidates.append(np.array([(x, y) for x in range(2) for y in range(15)], dtype=np.int64))
    # random
    candidates.append(ALL_COORDS[np.random.choice(len(ALL_COORDS), 30, replace=False)])

    best_pts = None
    best_sc = -1
    for c in candidates:
        s = _score_full(c)
        if s > best_sc:
            best_sc = s
            best_pts = c.copy()

    remaining = MAX_TIME - (time.time() - t0)
    if remaining > 5:
        # split time among a few restarts from best
        final_pts, final_sc = _local_search(best_pts, remaining - 2)
        if final_sc > best_sc:
            best_sc = final_sc
            best_pts = final_pts

    return [tuple(map(int, pt)) for pt in best_pts.tolist()]

# EVOLVE-BLOCK-END
