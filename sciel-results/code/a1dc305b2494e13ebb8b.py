"""Greedy + precomputed-index 1-exchange local search covering for C(v,k,t)."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import sys
import math
from collections import defaultdict


def _precompute(v, k, t):
    """Build global index structures for all k-subsets of v and their t-subsets."""
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}

    block_covered = [None] * num_blocks
    block_covered_set = [None] * num_blocks
    t_to_blocks = [[] for _ in range(num_t)]
    for bi, b in enumerate(all_blocks):
        idxs = []
        for ts in itertools.combinations(b, t):
            idxs.append(t_to_idx[ts])
        block_covered[bi] = idxs
        block_covered_set[bi] = set(idxs)
        for idx in idxs:
            t_to_blocks[idx].append(bi)
    return all_blocks, num_blocks, num_t, block_covered, block_covered_set, t_to_blocks


def _remove_redundant(blocks, block_covered, num_t):
    """Remove blocks whose t-subsets are all covered by other selected blocks."""
    if not blocks:
        return blocks
    cover_count = [0] * num_t
    for bi in blocks:
        for idx in block_covered[bi]:
            cover_count[idx] += 1
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            bi = blocks[i]
            redundant = True
            for idx in block_covered[bi]:
                if cover_count[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for idx in block_covered[bi]:
                    cover_count[idx] -= 1
                blocks.pop(i)
                changed = True
                break
    return blocks


def _greedy_construct(num_t, block_covered, block_covered_set, deadline, rng):
    """Pick blocks by maximum uncovered t-subset coverage (exact, precomputed)."""
    uncovered = set(range(num_t))
    blocks = []
    while uncovered and time.time() < deadline:
        best_bi = -1
        best_gain = 0
        for bi in range(len(block_covered)):
            gain = 0
            for idx in block_covered[bi]:
                if idx in uncovered:
                    gain += 1
            if gain > best_gain or (gain == best_gain and best_bi == -1):
                best_gain = gain
                best_bi = bi
        if best_bi == -1 or best_gain == 0:
            break
        blocks.append(best_bi)
        uncovered.difference_update(block_covered_set[best_bi])
    return blocks, uncovered


def _fill(blocks, num_t, block_covered, block_covered_set, deadline, rng):
    """Add blocks to cover any remaining uncovered t-subsets (greedy)."""
    covered = set()
    for bi in blocks:
        covered.update(block_covered[bi])
    uncovered = set(range(num_t)) - covered
    blocks = list(blocks)
    while uncovered and time.time() < deadline:
        best_bi = -1
        best_gain = 0
        for bi in range(len(block_covered)):
            if bi in set(blocks):
                continue
            gain = 0
            for idx in block_covered[bi]:
                if idx in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_bi = bi
        if best_bi == -1 or best_gain == 0:
            break
        blocks.append(best_bi)
        uncovered.difference_update(block_covered_set[best_bi])
    return blocks


def _local_search(blocks, block_covered, block_covered_set, t_to_blocks, num_t, deadline, rng, max_iter=5000):
    """1-exchange local search using precomputed indices."""
    if not blocks:
        return blocks
    cover_count = [0] * num_t
    for bi in blocks:
        for idx in block_covered[bi]:
            cover_count[idx] += 1
    selected_set = set(blocks)

    improved = True
    iters = 0
    while improved and iters < max_iter and time.time() < deadline:
        improved = False
        iters += 1

        # Pass A: remove any redundant block
        removed = False
        for i in range(len(blocks)):
            bi = blocks[i]
            redundant = True
            for idx in block_covered[bi]:
                if cover_count[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for idx in block_covered[bi]:
                    cover_count[idx] -= 1
                blocks.pop(i)
                selected_set.discard(bi)
                improved = True
                removed = True
                break
        if removed:
            continue

        # Pass B: try a 1-exchange that reduces count (swap a block for one covering its essentials)
        for i in range(len(blocks)):
            bi = blocks[i]
            essential = [idx for idx in block_covered[bi] if cover_count[idx] == 1]
            if not essential:
                for idx in block_covered[bi]:
                    cover_count[idx] -= 1
                blocks.pop(i)
                selected_set.discard(bi)
                improved = True
                break

            # Candidates: unselected blocks covering essential[0]
            cand_pool = [c for c in t_to_blocks[essential[0]] if c not in selected_set]
            if not cand_pool:
                continue
            if len(cand_pool) > 2000:
                cand_pool = rng.sample(cand_pool, 2000)

            old_set = block_covered_set[bi]
            best_cand = -1
            best_metric = float('-inf')
            for cand in cand_pool:
                cand_set = block_covered_set[cand]
                # must cover all essential subsets
                if any(e not in cand_set for e in essential):
                    continue
                # metric: new coverage gained minus unique loss not covered by cand
                new_cover = 0
                for idx in block_covered[cand]:
                    if cover_count[idx] == 0:
                        new_cover += 1
                unique_lost = 0
                for idx in block_covered[bi]:
                    if cover_count[idx] == 1 and idx not in cand_set:
                        unique_lost += 1
                metric = new_cover - unique_lost * 2.0
                if metric > best_metric:
                    best_metric = metric
                    best_cand = cand

            if best_cand != -1 and best_metric > 0:
                for idx in block_covered[bi]:
                    cover_count[idx] -= 1
                for idx in block_covered[best_cand]:
                    cover_count[idx] += 1
                blocks[i] = best_cand
                selected_set.discard(bi)
                selected_set.add(best_cand)
                improved = True
                break
    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    all_blocks, num_blocks, num_t, block_covered, block_covered_set, t_to_blocks = _precompute(v, k, t)

    best_blocks = None
    best_count = float('inf')

    # Multi-restart greedy + local search
    rng = random.Random(42)
    restarts = 4
    for r in range(restarts):
        if time.time() > deadline - 20.0:
            break
        rng_r = random.Random(42 + r * 7919)
        blocks, uncovered = _greedy_construct(num_t, block_covered, block_covered_set,
                                              min(deadline - 15.0, start_time + 12.0 + r), rng_r)
        if uncovered:
            blocks = _fill(blocks, num_t, block_covered, block_covered_set, deadline - 10.0, rng_r)
        blocks = _remove_redundant(blocks, block_covered, num_t)
        blocks = _local_search(blocks, block_covered, block_covered_set, t_to_blocks, num_t,
                               deadline - 8.0, rng_r, max_iter=3000)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]

    if best_blocks is None:
        blocks, uncovered = _greedy_construct(num_t, block_covered, block_covered_set, deadline - 5.0, rng)
        blocks = _fill(blocks, num_t, block_covered, block_covered_set, deadline - 3.0, rng)
        best_blocks = blocks
    else:
        blocks = best_blocks

    # Final fill to guarantee validity, then cleanup
    blocks = _fill(blocks, num_t, block_covered, block_covered_set, deadline - 1.5, rng)
    blocks = _remove_redundant(blocks, block_covered, num_t)

    # Convert to distinct sorted point-lists
    seen = set()
    out = []
    for bi in blocks:
        b = sorted(all_blocks[bi])
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            out.append(b)
    return out
# EVOLVE-BLOCK-END
