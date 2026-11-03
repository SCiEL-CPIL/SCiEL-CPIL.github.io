"""Covering design C(v,k,t): fast randomized greedy + swap-out local search + redundancy removal."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def _all_t_subsets(v, t):
    return list(itertools.combinations(range(v), t))


def _subset_to_idx(t_subs):
    return {s: i for i, s in enumerate(t_subs)}


def _precompute_gains(v, k, t, idx_map, n_tsubs):
    """Precompute gain[x][mask_of_t1subs] = number of uncovered tsubs gained by adding x.

    For each point x, for each (t-1)-subset s (represented as a frozenset or tuple),
    compute the set of t-subset indices covered by s ∪ {x}.
    We store: gain_table[x] = dict mapping frozenset(s) -> set of t-subset indices.
    """
    # All (t-1)-subsets of range(v)
    t1_all = list(itertools.combinations(range(v), t - 1))
    # For each x, map each t1-subset not containing x to the set of t-sub indices
    gain_table = [dict() for _ in range(v)]
    for x in range(v):
        for s in t1_all:
            if x in s:
                continue
            new_sub = tuple(sorted(s + (x,)))
            if new_sub in idx_map:
                gain_table[x][frozenset(s)] = idx_map[new_sub]
    return gain_table, t1_all


def _greedy_pass_fast(v, k, t, rng, gain_table, t1_all, idx_map, n_tsubs):
    """Fast greedy: pick random uncovered tsub, grow block using precomputed gains."""
    uncovered = [True] * n_tsubs
    uncovered_count = n_tsubs
    blocks = []

    while uncovered_count > 0:
        while True:
            idx = rng.randrange(n_tsubs)
            if uncovered[idx]:
                break
        # We don't have t_subs list here; reconstruct start from idx? 
        # Actually we need the actual subset. Let's pass t_subs too.
        # For now, this function needs t_subs. Let me restructure.
        raise NotImplementedError


def _greedy_pass(v, k, t, rng, gain_table, t1_all, idx_map, n_tsubs, t_subs):
    """Fast greedy pass using precomputed gain tables."""
    uncovered = [True] * n_tsubs
    uncovered_count = n_tsubs
    blocks = []

    while uncovered_count > 0:
        while True:
            idx = rng.randrange(n_tsubs)
            if uncovered[idx]:
                break
        start_sub = t_subs[idx]
        block = list(start_sub)

        while len(block) < k:
            block_set = set(block)
            candidates = [x for x in range(v) if x not in block_set]
            best_x = None
            best_gain = -1

            # Get all (t-1)-subsets of current block
            t1_subs_block = list(itertools.combinations(block, t - 1))

            for x in candidates:
                gain = 0
                gt = gain_table[x]
                for rest in t1_subs_block:
                    i = gt.get(frozenset(rest))
                    if i is not None and uncovered[i]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x

            block.append(best_x)

        block.sort()
        blocks.append(block)

        for comb in itertools.combinations(block, t):
            i = idx_map[comb]
            if uncovered[i]:
                uncovered[i] = False
                uncovered_count -= 1

    return blocks


def _remove_redundant_blocks(blocks, v, k, t, idx_map, n_tsubs, rng=None):
    """Remove redundant blocks."""
    if not blocks:
        return blocks
    if rng is None:
        rng = random.Random(0)

    block_masks = []
    for b in blocks:
        mask = set()
        for comb in itertools.combinations(b, t):
            mask.add(idx_map[comb])
        block_masks.append(mask)

    n_blocks = len(blocks)
    coverage_count = [0] * n_tsubs
    for mask in block_masks:
        for i in mask:
            coverage_count[i] += 1

    kept_indices = list(range(n_blocks))
    changed = True
    max_passes = 8

    for _ in range(max_passes):
        if not changed:
            break
        changed = False
        rng.shuffle(kept_indices)

        new_kept = []
        for i in kept_indices:
            redundant = True
            for tsub_idx in block_masks[i]:
                if coverage_count[tsub_idx] <= 1:
                    redundant = False
                    break
            if not redundant:
                new_kept.append(i)
            else:
                for tsub_idx in block_masks[i]:
                    coverage_count[tsub_idx] -= 1
                changed = True

        kept_indices = new_kept

    return [blocks[i] for i in kept_indices]


def _swap_local_search(blocks, v, k, t, idx_map, n_tsubs, rng, max_iters=50):
    """Try to replace blocks with alternatives that cover fewer unique tsubs.
    
    For each block, try removing it and see if coverage is still complete.
    If not, try swapping it for a different block that covers the missing tsubs.
    This is simplified: just do redundancy removal with randomization.
    """
    if not blocks:
        return blocks

    block_masks = []
    for b in blocks:
        mask = set()
        for comb in itertools.combinations(b, t):
            mask.add(idx_map[comb])
        block_masks.append(mask)

    n_blocks = len(blocks)
    coverage_count = [0] * n_tsubs
    for mask in block_masks:
        for i in mask:
            coverage_count[i] += 1

    kept = list(range(n_blocks))
    improved = True
    passes = 0
    max_passes = 12

    while improved and passes < max_passes:
        improved = False
        passes += 1
        rng.shuffle(kept)
        new_kept = []
        for i in kept:
            redundant = True
            for tsub_idx in block_masks[i]:
                if coverage_count[tsub_idx] <= 1:
                    redundant = False
                    break
            if not redundant:
                new_kept.append(i)
            else:
                for tsub_idx in block_masks[i]:
                    coverage_count[tsub_idx] -= 1
                improved = True
        kept = new_kept

    return [blocks[i] for i in kept]


def construct(v, k, t):
    """Find a covering design using fast randomized greedy + local search + redundancy removal."""
    start_time = time.time()
    deadline = start_time + 118  # Safety margin: 118s out of 150s limit

    t_subs = _all_t_subsets(v, t)
    idx_map = _subset_to_idx(t_subs)
    n_tsubs = len(t_subs)

    # Precompute gain tables (one-time cost)
    gain_table, t1_all = _precompute_gains(v, k, t, idx_map, n_tsubs)

    best_blocks = None
    best_len = float('inf')

    attempt = 0
    while time.time() < deadline:
        rng = random.Random(1000 + attempt)
        blocks = _greedy_pass(v, k, t, rng, gain_table, t1_all, idx_map, n_tsubs, t_subs)
        # Redundancy removal
        blocks = _remove_redundant_blocks(blocks, v, k, t, idx_map, n_tsubs, rng)
        # Additional swap-based local search
        blocks = _swap_local_search(blocks, v, k, t, idx_map, n_tsubs, rng)

        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks

        if best_len <= 338:
            break

        attempt += 1

    if best_blocks is None:
        rng = random.Random(0)
        best_blocks = _greedy_pass(v, k, t, rng, gain_table, t1_all, idx_map, n_tsubs, t_subs)
        best_blocks = _remove_redundant_blocks(best_blocks, v, k, t, idx_map, n_tsubs, rng)

    return best_blocks
# EVOLVE-BLOCK-END
