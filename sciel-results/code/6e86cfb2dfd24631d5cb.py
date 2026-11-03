"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def _build_index(v, t):
    """Precompute t-subset index mapping and per-point subset lists."""
    sub_to_idx = {}
    idx_to_sub = []
    idx = 0
    for sub in itertools.combinations(range(v), t):
        sub_to_idx[sub] = idx
        idx_to_sub.append(sub)
        idx += 1
    n_subsets = idx
    # For each point, list of t-subset indices containing it
    point_subs = [[] for _ in range(v)]
    for sub, i in sub_to_idx.items():
        for x in sub:
            point_subs[x].append(i)
    return sub_to_idx, idx_to_sub, n_subsets, point_subs


def _greedy_bitmask(v, k, t, rng, deadline, sub_to_idx, idx_to_sub, n_subsets, point_subs):
    """Fast greedy using bitmasks for uncovered t-subsets."""
    # uncovered: list of bool, True if not yet covered
    uncovered = [True] * n_subsets
    uncovered_count = n_subsets
    blocks = []

    # Helper: mark all t-subsets of a block as covered, return newly covered count
    def mark_block(block):
        nonlocal uncovered_count
        newly = 0
        for sub in itertools.combinations(block, t):
            i = sub_to_idx[sub]
            if uncovered[i]:
                uncovered[i] = False
                newly += 1
        return newly

    # Precompute for each point x: sorted list of t-subset indices
    # For gain computation when block has size b, we need:
    #   gain(x) = #{uncovered t-subsets S : x in S, S\{x} ⊆ block}
    # We iterate over (t-1)-subsets of block and check if (sub + x) is uncovered.

    while uncovered_count > 0 and time.time() < deadline:
        # Pick a random uncovered t-subset to seed the block
        # Find an uncovered index
        # For speed, maintain a list of uncovered indices or just scan
        # Since n_subsets = 11628, scanning is fine
        unc_idx = -1
        for i in range(n_subsets):
            if uncovered[i]:
                unc_idx = i
                break
        if unc_idx < 0:
            break
        seed_sub = idx_to_sub[unc_idx]
        block = list(seed_sub)

        # Greedily add points to maximize gain
        while len(block) < k:
            block_set = frozenset(block)
            best_gain = -1
            best_x = -1
            # Precompute (t-1)-subsets of block for gain computation
            if len(block) >= t - 1:
                rest_list = list(itertools.combinations(block, t - 1))
            else:
                rest_list = []

            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in rest_list:
                    # Form t-subset: rest + (x,)
                    # Need to sort and look up
                    # rest is a tuple of t-1 elements, add x
                    # Create sorted tuple
                    arr = list(rest) + [x]
                    arr.sort()
                    sub = tuple(arr)
                    i = sub_to_idx.get(sub, -1)
                    if i >= 0 and uncovered[i]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x < 0 or best_gain <= 0:
                # No gain; add random point to fill block
                remaining = [x for x in range(v) if x not in block_set]
                if remaining:
                    best_x = rng.choice(remaining)
                else:
                    break
            block.append(best_x)

        block.sort()
        # Avoid duplicate blocks
        if block not in blocks:
            blocks.append(block)
            newly = mark_block(block)
            uncovered_count -= newly

    return blocks, uncovered_count


def _local_improve(blocks, v, k, t, rng, deadline, sub_to_idx, idx_to_sub, n_subsets, point_subs):
    """Try to replace blocks with better ones to reduce count or improve coverage."""
    if not blocks:
        return blocks

    # Build coverage: for each t-subset, count how many blocks cover it
    cover_count = [0] * n_subsets
    for block in blocks:
        for sub in itertools.combinations(block, t):
            i = sub_to_idx[sub]
            cover_count[i] += 1

    blocks = list(blocks)

    # Try to remove redundant blocks (those that don't uniquely cover any t-subset)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Recompute cover counts
        cover_count = [0] * n_subsets
        for block in blocks:
            for sub in itertools.combinations(block, t):
                i = sub_to_idx[sub]
                cover_count[i] += 1

        # Find redundant blocks
        for i in range(len(blocks) - 1, -1, -1):
            block = blocks[i]
            unique = False
            for sub in itertools.combinations(block, t):
                if cover_count[sub_to_idx[sub]] == 1:
                    unique = True
                    break
            if not unique:
                blocks.pop(i)
                changed = True

    # Try block replacement: for each block, try replacing it with a different block
    # that covers the same unique t-subsets plus more
    if time.time() < deadline:
        cover_count = [0] * n_subsets
        for block in blocks:
            for sub in itertools.combinations(block, t):
                i = sub_to_idx[sub]
                cover_count[i] += 1

        # For each block, find t-subsets it uniquely covers
        # Try to replace with a block that covers those and more
        n_blocks = len(blocks)
        for i in range(n_blocks):
            if time.time() > deadline:
                break
            block_i = blocks[i]
            # Find t-subsets uniquely covered by block_i
            unique_subs = []
            for sub in itertools.combinations(block_i, t):
                idx = sub_to_idx[sub]
                if cover_count[idx] == 1:
                    unique_subs.append(idx)
            if not unique_subs:
                continue
            # Try random replacements
            for attempt in range(20):
                if time.time() > deadline:
                    break
                # Generate a random block containing all unique_subs' elements
                # The union of all unique t-subsets must be ⊆ new block
                union = set()
                for idx in unique_subs:
                    union.update(idx_to_sub[idx])
                if len(union) > k:
                    break  # Can't fit in one block
                # Fill remaining slots randomly
                remaining_pool = [x for x in range(v) if x not in union]
                need = k - len(union)
                if need < 0:
                    break
                fill = rng.sample(remaining_pool, need) if need > 0 else []
                new_block = sorted(list(union) + fill)
                if new_block == block_i:
                    continue
                # Check if new_block covers all unique_subs
                new_block_set = set(new_block)
                ok = True
                for idx in unique_subs:
                    if not all(x in new_block_set for x in idx_to_sub[idx]):
                        ok = False
                        break
                if not ok:
                    continue
                # Check validity: after replacement, all t-subsets still covered
                # New block covers its own t-subsets
                # Remove old block's coverage, add new block's coverage
                # Check if any t-subset drops to 0
                valid = True
                # First, check t-subsets of old block that are NOT in new block
                old_set = set(block_i)
                new_set = set(new_block)
                for sub in itertools.combinations(block_i, t):
                    idx = sub_to_idx[sub]
                    if cover_count[idx] == 1:
                        # Only covered by old block; must be in new block
                        if not all(x in new_set for x in sub):
                            valid = False
                            break
                if not valid:
                    continue
                # Accept replacement
                blocks[i] = new_block
                # Update cover counts
                for sub in itertools.combinations(block_i, t):
                    cover_count[sub_to_idx[sub]] -= 1
                for sub in itertools.combinations(new_block, t):
                    cover_count[sub_to_idx[sub]] += 1
                # Check no zero
                zero_found = False
                for idx in range(n_subsets):
                    if cover_count[idx] < 0:
                        # Shouldn't happen, but restore
                        zero_found = True
                        break
                if zero_found:
                    # Restore
                    blocks[i] = block_i
                    for sub in itertools.combinations(new_block, t):
                        cover_count[sub_to_idx[sub]] -= 1
                    for sub in itertools.combinations(block_i, t):
                        cover_count[sub_to_idx[sub]] += 1

    return blocks


def construct(v, k, t):
    """Construct a covering design C(v,k,t) using randomized greedy with restarts."""
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin before 150s limit

    # Precompute index structures
    sub_to_idx, idx_to_sub, n_subsets, point_subs = _build_index(v, t)

    best_blocks = None
    best_count = float('inf')

    # Multiple restarts with different seeds
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break

        rng = random.Random(42 + restart * 1000)

        blocks, unc = _greedy_bitmask(v, k, t, rng, deadline - 20,
                                       sub_to_idx, idx_to_sub, n_subsets, point_subs)

        if not blocks:
            continue

        if unc == 0:
            # Local improvement
            blocks = _local_improve(blocks, v, k, t, rng, deadline - 10,
                                     sub_to_idx, idx_to_sub, n_subsets, point_subs)

            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
                if best_count <= 300:
                    break

    if best_blocks is None:
        # Fallback: single deterministic greedy
        rng = random.Random(0)
        blocks, unc = _greedy_bitmask(v, k, t, rng, deadline,
                                       sub_to_idx, idx_to_sub, n_subsets, point_subs)
        best_blocks = blocks

    return best_blocks
# EVOLVE-BLOCK-END
