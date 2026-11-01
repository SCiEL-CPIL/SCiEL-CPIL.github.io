"""Approach: greedy seed + randomized local search (block replacement) using NumPy."""
import itertools
import random
import numpy as np


def _all_tsubs(v, t):
    return list(itertools.combinations(range(v), t))


def _cover_mask(block, tsub_index):
    """Return a set of t-subsets (as frozensets) covered by block."""
    return set(itertools.combinations(block, t))


def construct(v, k, t):
    rng = random.Random(12345)

    all_tsubs = _all_tsubs(v, t)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)

    # ---- Phase 1: Greedy seed (improved: pick block maximizing uncovered coverage) ----
    uncovered = np.ones(n_tsubs, dtype=bool)
    blocks = []

    # Precompute for each point-set of size < k the potential; but simpler:
    # For greedy seed, build block by picking the t-subset with max "synergy" then fill.
    while uncovered.any():
        # Pick the uncovered t-subset that, when extended to a k-block, covers most uncovered.
        # Heuristic: start from a random uncovered t-subset, then hill-climb the block.
        best_block = None
        best_covered = -1

        # Try a few random starting t-subsets
        unc_idx = np.where(uncovered)[0]
        tries = min(6, len(unc_idx))
        for _ in range(tries):
            start_ts = tuple(all_tsubs[rng.choice(unc_idx)])
            block = set(start_ts)
            # Hill-climb: add points to maximize newly covered uncovered t-subsets
            for _ in range(k - t):
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    # gain = number of uncovered t-subsets in block U {x}
                    new_block = block | {x}
                    gain = 0
                    for ts in itertools.combinations(new_block, t):
                        if uncovered[tsub_to_idx[ts]]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.add(best_x)

            # Count total uncovered covered by this full block
            gain_total = 0
            for ts in itertools.combinations(block, t):
                if uncovered[tsub_to_idx[ts]]:
                    gain_total += 1
            if gain_total > best_covered:
                best_covered = gain_total
                best_block = block

        if best_block is None or best_covered <= 0:
            # Fallback: just take the min uncovered t-subset and fill arbitrarily
            i = int(np.where(uncovered)[0][0])
            block = set(all_tsubs[i])
            for x in range(v):
                if len(block) >= k:
                    break
                if x not in block:
                    block.add(x)
            best_block = block

        blocks.append(sorted(best_block))
        for ts in itertools.combinations(best_block, t):
            uncovered[tsub_to_idx[ts]] = False

    # ---- Phase 2: Local search — try to replace blocks to reduce count ----
    # Recompute uncovered set for current blocks
    def compute_uncovered(blk_list):
        unc = np.ones(n_tsubs, dtype=bool)
        for b in blk_list:
            for ts in itertools.combinations(b, t):
                unc[tsub_to_idx[ts]] = False
        return unc

    uncovered = compute_uncovered(blocks)

    # Local search: for a fixed number of iterations, try to improve by
    # (a) removing a redundant block, or
    # (b) replacing a block with a better one.
    max_iters = 3000
    for it in range(max_iters):
        # Try removing each block if it doesn't leave any t-subset uncovered
        improved = False
        # Try replacement of a random block
        if len(blocks) > 1:
            idx = rng.randrange(len(blocks))
            old_block = blocks[idx]
            # Temporarily remove it
            tmp_unc = compute_uncovered(blocks[:idx] + blocks[idx+1:])
            # Now try to find a single block that covers all currently-uncovered-by-old t-subsets
            # that are uncovered after removal. Actually we want a block that, when added,
            # covers at least as many as old_block did, ideally more.
            # Simpler: try to find a block that covers all t-subsets that ONLY old_block covered.
            # Compute t-subsets that become uncovered if we remove old_block:
            # Those are t-subsets covered by old_block but by no other block.
            # That's expensive; instead, do a greedy rebuild of one block to maximize coverage
            # of the t-subsets that are uncovered after removing old_block, plus keep it valid.

            # Candidate: greedily build a block to cover as many of tmp_unc as possible.
            best_new = None
            best_gain = -1
            for _ in range(3):
                if not tmp_unc.any():
                    break
                # start from a random uncovered t-subset
                unc_idx2 = np.where(tmp_unc)[0]
                start_ts = tuple(all_tsubs[rng.choice(unc_idx2)])
                blk = set(start_ts)
                for _ in range(k - t):
                    best_x = None
                    best_g = -1
                    for x in range(v):
                        if x in blk:
                            continue
                        nb = blk | {x}
                        g = 0
                        for ts in itertools.combinations(nb, t):
                            if tmp_unc[tsub_to_idx[ts]]:
                                g += 1
                        if g > best_g:
                            best_g = g
                            best_x = x
                    if best_x is None:
                        break
                    blk.add(best_x)
                # gain = number of tmp_unc it covers
                g_total = 0
                for ts in itertools.combinations(blk, t):
                    if tmp_unc[tsub_to_idx[ts]]:
                        g_total += 1
                if g_total > best_gain:
                    best_gain = g_total
                    best_new = blk

            # old_block's coverage of tmp_unc:
            old_gain = 0
            for ts in itertools.combinations(old_block, t):
                if tmp_unc[tsub_to_idx[ts]]:
                    old_gain += 1

            # If new block covers strictly more of the "at-risk" t-subsets, replace
            if best_new is not None and best_gain > old_gain:
                blocks[idx] = sorted(best_new)
                improved = True

        # Try removing a redundant block
        if not improved and len(blocks) > 1:
            # Find a block whose removal leaves everything covered
            # Compute per-block unique coverage
            # For speed, only try a random block
            idx = rng.randrange(len(blocks))
            tmp_unc = compute_uncovered(blocks[:idx] + blocks[idx+1:])
            if not tmp_unc.any():
                blocks.pop(idx)
                improved = True

        # Recompute uncovered for next iteration's bookkeeping
        uncovered = compute_uncovered(blocks)
        if not uncovered.any():
            break

    # Final validity check and return
    # Ensure every t-subset is covered
    final_unc = compute_uncovered(blocks)
    if final_unc.any():
        # Fallback: add blocks for remaining uncovered t-subsets (should not happen)
        for ts in all_tsubs:
            if final_unc[tsub_to_idx[ts]]:
                blk = list(ts)
                for x in range(v):
                    if len(blk) >= k:
                        break
                    if x not in blk:
                        blk.append(x)
                blocks.append(sorted(blk))
                final_unc[tsub_to_idx[ts]] = False

    return blocks
