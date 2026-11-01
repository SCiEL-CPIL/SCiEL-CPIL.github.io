"""Randomized adaptive-weight greedy covering + local-search polish."""
import time
import random
import itertools
from collections import defaultdict


def _all_t_subsets(v, t):
    return [set(c) for c in itertools.combinations(range(v), t)]


def _build_greedy(v, k, t, deadline, rng, use_adaptive=True, randomize=True):
    all_subs = _all_t_subsets(v, t)
    n = len(all_subs)
    uncovered = bytearray(b'\x01') * n
    # map subset -> index for fast lookup
    idx = {}
    for i, s in enumerate(all_subs):
        idx[frozenset(s)] = i

    # weights per point
    weights = [1.0] * v
    blocks = []

    # Precompute, for each point, the list of subset-indices that contain it
    # (for efficient uncovered-count). We'll maintain uncovered_count per point.
    point_subsets = [[] for _ in range(v)]
    for i, s in enumerate(all_subs):
        for x in s:
            point_subsets[x].append(i)
    uncovered_count = [len(ps) for ps in point_subsets]

    def refresh_weights():
        if not use_adaptive:
            return
        # scale weights toward uncovered_count (smooth)
        for x in range(v):
            weights[x] = 1.0 + 0.5 * uncovered_count[x]

    while True:
        if time.time() > deadline:
            break
        # find any uncovered subset to seed the block
        seed_idx = None
        for i in range(n):
            if uncovered[i]:
                seed_idx = i
                break
        if seed_idx is None:
            break
        seed = list(all_subs[seed_idx])
        block = list(seed)
        block_set = set(block)

        # pick additional points up to k
        while len(block) < k:
            if time.time() > deadline:
                break
            best_gain = -1
            best_cand = None
            # candidates: all points not in block
            # compute gain = number of uncovered t-subsets that would become covered
            # A t-subset is covered by adding x if it contains x and the other t-1 are in block
            # and the subset is currently uncovered.
            # For efficiency, iterate over candidates and compute gain.
            if randomize:
                cand_list = [x for x in range(v) if x not in block_set]
                rng.shuffle(cand_list)
            else:
                cand_list = [x for x in range(v) if x not in block_set]

            for x in cand_list:
                # gain: count uncovered subsets S where x in S and S\{x} subset of block
                gain = 0
                # S has t elements, one is x, the rest (t-1) must be in block
                for rest in itertools.combinations(block, t - 1):
                    s = frozenset(list(rest) + [x])
                    i = idx.get(s)
                    if i is not None and uncovered[i]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_cand = x
            if best_cand is None:
                break
            block.append(best_cand)
            block_set.add(best_cand)

        # add the block if it covers anything new
        if len(block) == k:
            covered_any = False
            for s in itertools.combinations(block, t):
                i = idx[frozenset(s)]
                if uncovered[i]:
                    uncovered[i] = 0
                    covered_any = True
                    # decrement uncovered_count for each point in s
                    for x in s:
                        uncovered_count[x] -= 1
            if covered_any:
                blocks.append(sorted(block))
                refresh_weights()

    return blocks, all_subs, idx, uncovered


def _local_search_polish(v, k, t, blocks, all_subs, idx, uncovered, deadline, rng):
    # uncovered is a bytearray of length n; 1 means uncovered.
    # But after greedy, uncovered should be all 0. We'll recompute coverage counts
    # to allow removal.
    # coverage count: how many blocks cover each subset
    n = len(all_subs)
    cover_count = [0] * n
    for b in blocks:
        for s in itertools.combinations(b, t):
            cover_count[idx[frozenset(s)]] += 1

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # try to remove a redundant block
        for bi in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline:
                break
            b = blocks[bi]
            redundant = True
            for s in itertools.combinations(b, t):
                i = idx[frozenset(s)]
                if cover_count[i] == 1:
                    redundant = False
                    break
            if redundant:
                for s in itertools.combinations(b, t):
                    i = idx[frozenset(s)]
                    cover_count[i] -= 1
                blocks.pop(bi)
                improved = True
                break

        # try swap: replace one block with a different one that covers more uncovered
        # (only useful if we have uncovered, but after removal all are covered)
        # Instead, try to reduce by finding a block that can be replaced by a smaller-impact
        # block. Skip for simplicity; removal is the main win.

    return blocks


def construct(v, k, t):
    start = time.time()
    # hard deadline with safety margin
    deadline = start + 140.0

    rng = random.Random(12345)

    # Run greedy with adaptive weights + randomization
    blocks, all_subs, idx, uncovered = _build_greedy(
        v, k, t, deadline, rng, use_adaptive=True, randomize=True)

    # If we still have time, polish with local search
    if time.time() < deadline - 2.0:
        blocks = _local_search_polish(v, k, t, blocks, all_subs, idx,
                                      uncovered, deadline, rng)

    # Ensure validity: if something went wrong (shouldn't), fallback to a
    # simple deterministic greedy to guarantee a cover.
    # Quick check: recompute uncovered
    still_uncovered = [0] * len(all_subs)
    for b in blocks:
        for s in itertools.combinations(b, t):
            still_uncovered[idx[frozenset(s)]] = 1
    if any(1 - x for x in still_uncovered):
        # fallback: naive greedy to guarantee coverage
        uncovered2 = [1 - x for x in still_uncovered]
        while True:
            seed_idx = None
            for i in range(len(all_subs)):
                if uncovered2[i]:
                    seed_idx = i
                    break
            if seed_idx is None:
                break
            block = list(all_subs[seed_idx])
            block_set = set(block)
            while len(block) < k:
                best_gain = -1
                best_cand = None
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        s = frozenset(list(rest) + [x])
                        i = idx.get(s)
                        if i is not None and uncovered2[i]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_cand = x
                if best_cand is None:
                    break
                block.append(best_cand)
                block_set.add(best_cand)
            if len(block) == k:
                blocks.append(sorted(block))
                for s in itertools.combinations(block, t):
                    uncovered2[idx[frozenset(s)]] = 0

    return blocks
