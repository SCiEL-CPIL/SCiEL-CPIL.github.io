# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def _build_lookup(v, k, t):
    """Precompute all structures needed for fast greedy and local search."""
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # For each (t-1)-subset S, map point x (not in S) -> t-subset index of S ∪ {x}
    # stored as a dict of dicts: gain_map[S][x] = idx  (only for x not in S)
    # For speed, store as a flat structure:
    #   For each S (tuple), a dict {x: idx}
    t_minus_1 = t - 1
    gain_map = {}
    for S in itertools.combinations(range(v), t_minus_1):
        d = {}
        for x in range(v):
            if x not in S:
                full = tuple(sorted(S + (x,)))
                d[x] = tsub_to_idx[full]
        gain_map[S] = d

    return all_tsubs, tsub_to_idx, n_tsubs, gain_map, t_minus_1


def _greedy_build(v, k, t, rng, deadline, all_tsubs, tsub_to_idx, n_tsubs, gain_map, t_minus_1):
    """Fast greedy: repeatedly pick the block covering the most uncovered t-subsets."""
    uncovered = np.ones(n_tsubs, dtype=bool)
    blocks = []

    while np.any(uncovered):
        if time.time() > deadline - 1.0:
            break

        # Pick a random uncovered t-subset to seed
        unc_idx = np.where(uncovered)[0]
        start_idx = rng.choice(unc_idx)
        start = all_tsubs[start_idx]
        block = list(start)
        block_set = set(block)

        while len(block) < k:
            if time.time() > deadline - 1.0:
                break

            # Compute gain for each candidate x not in block
            # gain[x] = number of uncovered t-subsets ts such that x in ts and ts\{x} ⊆ block
            # = number of (t-1)-subsets S of block with x not in S and S∪{x} uncovered
            best_x = -1
            best_gain = -1

            # Get all (t-1)-subsets of current block
            block_tsubs = list(itertools.combinations(block, t_minus_1))

            # Accumulate gains
            gains = [0] * v
            for S in block_tsubs:
                d = gain_map[S]
                for x, idx in d.items():
                    if x not in block_set and uncovered[idx]:
                        gains[x] += 1

            for x in range(v):
                if x not in block_set and gains[x] > best_gain:
                    best_gain = gains[x]
                    best_x = x

            if best_x == -1 or best_gain <= 0:
                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break
                best_x = rng.choice(candidates)

            block.append(best_x)
            block_set.add(best_x)

        block.sort()
        blocks.append(block)

        # Update uncovered
        for ts in itertools.combinations(block, t):
            uncovered[tsub_to_idx[ts]] = False

    return blocks


def _local_search(v, k, t, blocks, deadline, rng, all_tsubs, tsub_to_idx, n_tsubs, gain_map, t_minus_1):
    """
    Local search:
    1. Remove redundant blocks.
    2. Try to replace a block with a smaller-equivalent one that still covers its unique t-subsets.
    3. Try to merge: if two blocks' union of unique t-subsets can be covered by one new block, replace both.
    """
    # Build coverage count
    cover_count = np.zeros(n_tsubs, dtype=np.int32)
    for block in blocks:
        for ts in itertools.combinations(block, t):
            cover_count[tsub_to_idx[ts]] += 1

    n_blocks = len(blocks)
    active = [True] * n_blocks
    improved = True

    while improved and time.time() < deadline - 3.0:
        improved = False
        order = list(range(n_blocks))
        rng.shuffle(order)

        for bi in order:
            if not active[bi] or time.time() > deadline - 3.0:
                continue

            block = blocks[bi]

            # Find unique t-subsets (covered only by this block)
            unique_subs = []
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                if cover_count[idx] == 1:
                    unique_subs.append(ts)

            if not unique_subs:
                # Block is fully redundant
                for ts in itertools.combinations(block, t):
                    cover_count[tsub_to_idx[ts]] -= 1
                active[bi] = False
                improved = True
                continue

            # The new block must contain all points in the union of unique_subs
            required = set()
            for ts in unique_subs:
                required.update(ts)

            if len(required) > k:
                continue

            need_more = k - len(required)
            remaining = [x for x in range(v) if x not in required]

            # Try several random completions; keep the one that maximizes additional coverage
            best_new_block = None
            best_extra = -1

            n_trials = 10 if need_more <= 3 else 5
            for _ in range(n_trials):
                if need_more == 0:
                    new_block = sorted(required)
                else:
                    extra = rng.sample(remaining, need_more)
                    new_block = sorted(required | set(extra))

                # Count how many currently-covered t-subsets the new block would add
                # (i.e., t-subsets of new_block that are currently covered by ≥1 other block)
                extra_cover = 0
                for ts in itertools.combinations(new_block, t):
                    idx = tsub_to_idx[ts]
                    if cover_count[idx] >= 1:  # already covered by someone (including bi)
                        # if we replace bi, this is still covered if cover_count[idx] > 1
                        if cover_count[idx] > 1:
                            extra_cover += 1
                # We want to maximize extra_cover (blocks that remain covered after removal)
                if extra_cover > best_extra:
                    best_extra = extra_cover
                    best_new_block = new_block

            if best_new_block is None:
                continue

            # Check: does best_new_block cover all unique_subs? (it should by construction)
            # Verify: every unique_sub ⊆ best_new_block
            nb_set = set(best_new_block)
            valid = all(all(x in nb_set for x in ts) for ts in unique_subs)
            if not valid:
                continue

            # Replace block bi with best_new_block
            # Update cover_count: remove old block's contributions, add new block's
            for ts in itertools.combinations(block, t):
                cover_count[tsub_to_idx[ts]] -= 1
            for ts in itertools.combinations(best_new_block, t):
                cover_count[tsub_to_idx[ts]] += 1

            blocks[bi] = best_new_block
            improved = True

    result = [blocks[i] for i in range(n_blocks) if active[i]]
    return result


def construct(v, k, t):
    deadline = time.time() + 145.0  # safety margin before 150s kill

    # Build lookup once
    all_tsubs, tsub_to_idx, n_tsubs, gain_map, t_minus_1 = _build_lookup(v, k, t)

    best = None
    best_len = float('inf')

    # Phase 1: Multiple greedy restarts
    seed = 42
    while time.time() < deadline - 20.0:
        rng = random.Random(seed)
        candidate = _greedy_build(v, k, t, rng, deadline, all_tsubs, tsub_to_idx, n_tsubs, gain_map, t_minus_1)
        if len(candidate) < best_len:
            best = candidate
            best_len = len(candidate)

        # Local search on the best candidate found so far
        if time.time() < deadline - 10.0:
            rng_ls = random.Random(seed + 1000)
            refined = _local_search(v, k, t, candidate, deadline, rng_ls, all_tsubs, tsub_to_idx, n_tsubs, gain_map, t_minus_1)
            if len(refined) < best_len:
                best = refined
                best_len = len(refined)

        seed += 1

    # Phase 2: If we have time, try more aggressive local search on the best
    if best is not None and time.time() < deadline - 5.0:
        rng_final = random.Random(9999)
        refined = _local_search(v, k, t, best, deadline, rng_final, all_tsubs, tsub_to_idx, n_tsubs, gain_map, t_minus_1)
        if len(refined) < best_len:
            best = refined
            best_len = len(refined)

    if best is None:
        # Fallback: simple greedy
        rng = random.Random(0)
        best = _greedy_build(v, k, t, rng, deadline, all_tsubs, tsub_to_idx, n_tsubs, gain_map, t_minus_1)

    return best
# EVOLVE-BLOCK-END
