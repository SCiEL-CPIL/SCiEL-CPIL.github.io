"""Covering design C(v,k,t) via fast bitset greedy + local search.

EVOLVE-BLOCK-START
"""
import time
from itertools import combinations
from math import comb
import numpy as np

# deadline guard: leave a safety margin before the 150 s wall-clock kill
_DEADLINE = time.time() + 120.0


def _build_masks(v, k, t):
    """Map every t-subset to an index and precompute the mask of each k-block."""
    tsubs = list(combinations(range(v), t))
    index = {}
    for i, c in enumerate(tsubs):
        index[c] = i
    nt = len(tsubs)
    # coverage mask for every k-subset
    masks = np.zeros(comb(v, k), dtype=np.uint64)
    for j, b in enumerate(combinations(range(v), k)):
        m = 0
        for tsub in combinations(b, t):
            m |= 1 << index[tsub]
        masks[j] = m
    return nt, masks, tsubs, index


def _greedy(nt, masks, tsubs, index, k, v):
    """Greedy covering using the least-covered uncovered t-subset as seed."""
    covered = np.zeros(nt, dtype=np.uint64)  # bit i = covered by block i
    block_masks = []
    for _ in range(nt):  # at most one block per t-subset
        if time.time() >= _DEADLINE - 5:
            break
        # uncovered t-subset with fewest covering blocks
        cov_count = (covered != 0).sum(axis=1)
        uncovered = np.where(cov_count == 0)[0]
        if uncovered.size == 0:
            break
        # pick the least-covered among uncovered
        pick = uncovered[np.argmin(cov_count[uncovered])]
        seed = set(tsubs[pick])
        block = sorted(seed)
        # grow to size k
        while len(block) < k:
            best_x = None
            best_gain = -1
            bl = block
            for x in range(v):
                if x in bl:
                    continue
                gain = 0
                for rest in combinations(bl, t - 1):
                    tt = rest + (x,)
                    if tuple(sorted(tt)) in index:
                        i = index[tuple(sorted(tt))]
                        if covered[i] == 0:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)
        block.sort()
        bm = 0
        for tsub in combinations(block, t):
            bm |= 1 << index[tsub]
        block_masks.append(bm)
        covered[len(block_masks) - 1] = bm
    return block_masks


def _evaluate(blocks, nt, masks):
    """Return (num_blocks, uncovered_count)."""
    if not blocks:
        return 0, nt
    covered = np.array(blocks, dtype=np.uint64)
    cov_count = (covered != 0).sum(axis=1)
    return len(blocks), int((cov_count == 0).sum())


def _local_search(blocks, nt, masks, tsubs, index, k, v, budget):
    """1-swap / 1-insert / 1-delete / 2-opt moves until budget runs out."""
    end = time.time() + budget
    best = list(blocks)
    best_uncov = _evaluate(best, nt, masks)[1]
    while time.time() < end:
        improved = False
        covered = np.array(best, dtype=np.uint64) if best else np.zeros(0, dtype=np.uint64)
        cov_count = (covered != 0).sum(axis=1) if best else np.zeros(0)

        # --- 1-swap: replace block i with a better k-subset ---
        for i, b in enumerate(best):
            if time.time() >= end:
                break
            bset = set(b)
            bm = masks[list(combinations(range(v), k)).index(tuple(b))] if False else None
            # recompute current block mask
            cur_mask = 0
            for tsub in combinations(b, t):
                cur_mask |= 1 << index[tsub]
            # count how many t-subsets ONLY this block covers
            only = 0
            for tsub in combinations(b, t):
                idx = index[tsub]
                if cov_count[idx] == 1:
                    only += 1
            # try swapping one point out for one in
            best_gain = 0
            best_pair = None
            for out in b:
                rem = [x for x in b if x != out]
                for inp in range(v):
                    if inp in bset:
                        continue
                    nb = rem + [inp]
                    nb.sort()
                    nmask = 0
                    for tsub in combinations(nb, t):
                        nmask |= 1 << index[tsub]
                    # new blocks covering t-subsets that were only covered by old block
                    new_gain = 0
                    for tsub in combinations(nb, t):
                        idx = index[tsub]
                        if cov_count[idx] == 1:
                            new_gain += 1
                    # gain = new_gain - only (only counted once per block)
                    g = new_gain - only
                    if g > best_gain:
                        best_gain = g
                        best_pair = (out, inp)
            if best_pair is not None:
                out, inp = best_pair
                nb = [x for x in b if x != out]
                nb.append(inp)
                nb.sort()
                new_mask = 0
                for tsub in combinations(nb, t):
                    new_mask |= 1 << index[tsub]
                best[i] = nb
                improved = True
                break
        if improved:
            continue

        # --- 1-delete: remove a redundant block ---
        if len(best) > 1:
            for i in range(len(best)):
                if time.time() >= end:
                    break
                bm = 0
                for tsub in combinations(best[i], t):
                    bm |= 1 << index[tsub]
                # is every t-subset in this block covered by another block?
                redundant = True
                for tsub in combinations(best[i], t):
                    idx = index[tsub]
                    if cov_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    best.pop(i)
                    improved = True
                    break
        if improved:
            continue

        # --- 1-insert: add a block covering the most uncovered t-subsets ---
        if len(best) < nt:
            # find the uncovered t-subset with fewest covers (should be 0)
            uncov = np.where(cov_count == 0)[0]
            if uncov.size > 0:
                # build a block around the least-covered uncovered t-subset
                pick = uncov[0]
                seed = tsubs[pick]
                block = list(seed)
                while len(block) < k:
                    best_x = None
                    best_gain = -1
                    bl = block
                    for x in range(v):
                        if x in bl:
                            continue
                        gain = 0
                        for rest in combinations(bl, t - 1):
                            tt = tuple(sorted(rest + (x,)))
                            if tt in index:
                                ii = index[tt]
                                if cov_count[ii] == 0:
                                    gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    block.append(best_x)
                block.sort()
                # only add if it covers at least one uncovered t-subset
                adds = 0
                for tsub in combinations(block, t):
                    if cov_count[index[tsub]] == 0:
                        adds += 1
                if adds > 0:
                    best.append(block)
                    improved = True
        if improved:
            continue

        # no improvement this pass
        break

    return best


def construct(v, k, t):
    global _DEADLINE
    _DEADLINE = time.time() + 120.0
    nt, masks, tsubs, index = _build_masks(v, k, t)

    # greedy initialization
    blocks = _greedy(nt, masks, tsubs, index, k, v)

    # local search with remaining budget
    remaining = _DEADLINE - time.time() - 10
    if remaining > 5:
        blocks = _local_search(blocks, nt, masks, tsubs, index, k, v, remaining)

    # ensure validity: if something went wrong, fall back to a simple valid cover
    cov_count = None
    if blocks:
        covered = np.array(blocks, dtype=np.uint64)
        cov_count = (covered != 0).sum(axis=1)
    if cov_count is None or (cov_count == 0).any():
        # emergency: brute-force add blocks for any uncovered t-subset
        if cov_count is None:
            covered = np.zeros(0, dtype=np.uint64)
            cov_count = np.zeros(nt)
        for idx in np.where(cov_count == 0)[0]:
            ts = tsubs[idx]
            # build a block containing this t-subset
            block = list(ts)
            for x in range(v):
                if x not in block and len(block) < k:
                    block.append(x)
            block.sort()
            bm = 0
            for tsub in combinations(block, t):
                bm |= 1 << index[tsub]
            blocks.append(block)
            covered = np.vstack([covered, bm.reshape(1)]) if covered.size else bm.reshape(1)
            cov_count = (covered != 0).sum(axis=1)

    # return as lists
    return [list(b) for b in blocks]


# EVOLVE-BLOCK-END
