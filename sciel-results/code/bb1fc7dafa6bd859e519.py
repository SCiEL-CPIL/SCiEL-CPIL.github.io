"""Covering design C(v,k,t): minimize number of k-blocks covering all t-subsets.

Strategy: randomized greedy (bitmask coverage + vectorized gain), many seeds,
post-hoc exact redundancy removal, time-bounded with a hard deadline guard.
"""
import time
import random
import itertools
import numpy as np


def _tmask(v, t):
    """Map each t-subset of {0..v-1} to a unique bit index (0..M-1)."""
    idx = np.empty(1 << v, dtype=np.int64)
    idx[:] = -1
    m = 0
    for combo in itertools.combinations(range(v), t):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        idx[mask] = m
        m += 1
    return idx, m


def _randomized_greedy(v, k, t, idx, M, rng, time_budget):
    """One randomized-greedy pass. Returns list of blocks (tuples)."""
    all_points = np.arange(v)
    # Precompute, for every k-subset, the M-bit coverage mask (which t-sets it covers).
    # Build by OR-ing the t-subset masks within the k-subset.
    # Precompute mask_of_ksub via combinations.
    ksub_masks = []
    for combo in itertools.combinations(range(v), k):
        cmask = 0
        for sub in itertools.combinations(combo, t):
            smask = 0
            for x in sub:
                smask |= (1 << x)
            cmask |= (1 << idx[smask])
        ksub_masks.append(cmask)
    ksub_masks = np.array(ksub_masks, dtype=object)  # python ints (big)
    n_ksub = len(ksub_masks)

    # Precompute per-point coverage: for each point x, the set of t-subset indices it can cover.
    # point_tsets[x] = list of t-subset indices containing x.
    point_tsets = [[] for _ in range(v)]
    for combo in itertools.combinations(range(v), t):
        smask = 0
        for x in combo:
            smask |= (1 << x)
        i = idx[smask]
        for x in combo:
            point_tsets[x].append(i)
    point_tsets_arr = [np.array(pt, dtype=np.int64) for pt in point_tsets]

    deadline = time.time() + time_budget
    uncovered = [1] * M  # 1 = uncovered
    uncovered_arr = np.array(uncovered, dtype=np.int8)
    blocks = []
    used_ksub = set()

    while True:
        if time.time() > deadline:
            break
        # find an uncovered t-subset to open a block (random-ish: first uncovered)
        unc_idx = np.nonzero(uncovered_arr)[0]
        if len(unc_idx) == 0:
            break
        start_i = int(unc_idx[rng.randrange(len(unc_idx))])
        # which t-subset is this?
        # find combo for index start_i: we stored idx by mask; reverse map
        # We'll just use the t-subset containing start point = smallest point in that tset.
        # Determine the t-subset points for start_i:
        # Build reverse: tset_of_index
        # Precompute tset points per index.
        # (computed below lazily)
        # We need the set of points of the t-subset with index start_i.
        # Use a precomputed list.
        tset_pts = _TSET_POINTS[start_i]
        block = list(tset_pts)
        block_set = set(block)

        # Grow block to size k by best gain (with random tie-breaks).
        while len(block) < k:
            if time.time() > deadline:
                break
            # For each candidate point not in block, gain = number of uncovered t-sets
            # that contain that point AND are otherwise covered by (block).
            # A t-set {a..} is covered by adding x iff (t-set \ {x}) subset of block.
            # gain(x) = count of uncovered t-sets S with x in S and (S - {x}) subset of block.
            # = count over (t-1)-subsets R of block such that R ∪ {x} is an uncovered t-set.
            gains = np.zeros(v, dtype=np.int64)
            for R in itertools.combinations(block, t - 1):
                Rset = set(R)
                for x in range(v):
                    if x in Rset or x in block_set:
                        continue
                    # is R ∪ {x} an uncovered t-set?
                    # build mask
                    m = 0
                    for y in R:
                        m |= (1 << y)
                    m |= (1 << x)
                    ti = idx[m]
                    if ti >= 0 and uncovered[ti]:
                        gains[x] += 1
            # choose best gain with random tie-break
            candidates = [x for x in range(v) if x not in block_set]
            if not candidates:
                break
            maxg = max(gains[x] for x in candidates)
            best = [x for x in candidates if gains[x] == maxg]
            if not best:
                # fallback: random
                best = candidates
            x = best[rng.randrange(len(best))]
            block.append(x)
            block_set.add(x)

        # finalize block
        block.sort()
        blocks.append(tuple(block))
        used_ksub.add(frozenset(block))
        # update uncovered
        cmask = 0
        for sub in itertools.combinations(block, t):
            sm = 0
            for x in sub:
                sm |= (1 << x)
            cmask |= (1 << idx[sm])
        # clear bits
        for sub in itertools.combinations(block, t):
            sm = 0
            for x in sub:
                sm |= (1 << x)
            ti = idx[sm]
            if uncovered[ti]:
                uncovered[ti] = 0
                uncovered_arr[ti] = 0

    return blocks


# reverse map index -> tuple of points (filled in construct)
_TSET_POINTS = []


def _remove_redundant(blocks, v, k, t, idx, M):
    """Remove blocks that don't cover any t-set uniquely. Greedy exact sweep."""
    if not blocks:
        return blocks
    # For each t-set, count how many blocks cover it.
    # For each block, list of t-sets it covers.
    block_tsets = []
    for b in blocks:
        tsets = []
        for sub in itertools.combinations(b, t):
            sm = 0
            for x in sub:
                sm |= (1 << x)
            tsets.append(idx[sm])
        block_tsets.append(tsets)
    # count coverage
    count = np.zeros(M, dtype=np.int32)
    for tsets in block_tsets:
        for ti in tsets:
            count[ti] += 1
    keep = []
    for bi, tsets in enumerate(block_tsets):
        # block is redundant if every t-set it covers has count >= 2
        redundant = True
        for ti in tsets:
            if count[ti] == 1:
                redundant = False
                break
        if not redundant:
            keep.append(blocks[bi])
    return keep


def construct(v, k, t):
    global _TSET_POINTS
    start_time = time.time()
    HARD_DEADLINE = start_time + 110.0  # safety margin under 120s (evaluator kills at 150s)

    idx, M = _tmask(v, t)
    _TSET_POINTS = []
    for combo in itertools.combinations(range(v), t):
        smask = 0
        for x in combo:
            smask |= (1 << x)
        _TSET_POINTS.append(tuple(combo))

    best_blocks = None
    best_len = float('inf')

    # Multi-seed randomized greedy, time-bounded.
    seeds = list(range(0, 200000))
    rng = random.Random(12345)
    rng.shuffle(seeds)

    # Estimate per-pass budget; start generous, shrink as time runs out.
    pass_i = 0
    for seed in seeds:
        if time.time() > HARD_DEADLINE - 5.0:
            break
        # adaptive budget: give more time early
        remaining = HARD_DEADLINE - time.time()
        if pass_i < 5:
            budget = min(8.0, remaining)
        elif pass_i < 20:
            budget = min(4.0, remaining)
        else:
            budget = min(2.0, remaining)
        if budget <= 0.1:
            break
        rr = random.Random(seed)
        blocks = _randomized_greedy(v, k, t, idx, M, rr, budget)
        if blocks:
            # redundancy removal
            blocks = _remove_redundant(blocks, v, k, t, idx, M)
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
        pass_i += 1

    if best_blocks is None:
        # Fallback: pure deterministic greedy (shouldn't happen)
        best_blocks = []

    return [list(b) for b in best_blocks]
