"""
Strategy: Portfolio of Randomized Weighted Greedy Set Covers + Two-Phase Local Search

Key structural differences from the seed:
1. Multiple randomized weighted greedy runs (different random seeds perturb triple weights)
   to generate diverse initial coverings.
2. A dedicated pruning phase that repeatedly removes any block whose triples are all
   covered by other blocks (before attempting exchanges).
3. Exchange-based local search: try replacing one selected block with one from the pool
   that covers all its critical triples while adding new coverage.
4. Use numpy bitmasks for fast overlap computation.
"""

import itertools
import random
import time
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Precompute all t-subsets
    all_triples = list(itertools.combinations(range(v), t))
    triple_count = len(all_triples)
    triple_index = {tr: i for i, tr in enumerate(all_triples)}

    # Bitmask representation
    num_words = (triple_count + 63) // 64
    FULL_MASK = np.full(num_words, (1 << 64) - 1, dtype=np.uint64)

    def block_to_mask(block):
        mask = np.zeros(num_words, dtype=np.uint64)
        for tr in itertools.combinations(block, t):
            idx = triple_index[tr]
            mask[idx // 64] |= (1 << (idx % 64))
        return mask

    def mask_popcount(mask):
        return sum(bin(w).count('1') for w in mask)

    def mask_and_not(a, b):
        return a & (~b) & FULL_MASK

    def mask_or(a, b):
        return a | b

    def masks_equal(a, b):
        return np.array_equal(a, b)

    def mask_subset(a, b):
        """Check if a is a subset of b (all bits in a are in b)."""
        return np.all((a & ~b) == 0)

    # --- Helper: weighted greedy set cover ---
    def weighted_greedy(pool, pool_masks, weights):
        """
        Greedy set cover where at each step we pick the block maximizing
        sum of weights of newly covered triples.
        """
        covered = np.zeros(num_words, dtype=np.uint64)
        selected = []

        while not masks_equal(covered, FULL_MASK):
            if time.time() > deadline - 15:
                break
            best_idx = -1
            best_score = -1.0

            for idx in range(len(pool)):
                # Gain: weighted count of newly covered triples
                new_cover = mask_and_not(pool_masks[idx], covered)
                # Compute weighted popcount
                score = 0.0
                for wi in range(num_words):
                    w = new_cover[wi]
                    if w == 0:
                        continue
                    base = wi * 64
                    ww = w
                    while ww:
                        bit = (ww & -ww).bit_length() - 1
                        ti = base + bit
                        if ti < triple_count:
                            score += weights[ti]
                        ww &= ww - 1
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx == -1 or best_score <= 0:
                break

            selected.append(best_idx)
            covered = mask_or(covered, pool_masks[best_idx])

        return selected, covered

    # --- Helper: pruning pass ---
    def prune_blocks(selected, pool_masks):
        """Remove any block whose triples are all covered by other selected blocks."""
        changed = True
        while changed:
            changed = False
            n = len(selected)
            for i in range(n):
                if time.time() > deadline - 10:
                    return selected
                # Check if block i is redundant
                mask_i = pool_masks[selected[i]]
                # Compute coverage of all triples by blocks other than i
                other_cover = np.zeros(num_words, dtype=np.uint64)
                for j in range(n):
                    if j != i:
                        other_cover = mask_or(other_cover, pool_masks[selected[j]])
                # If mask_i is subset of other_cover, block i is redundant
                if mask_subset(mask_i, other_cover):
                    selected.pop(i)
                    changed = True
                    break
        return selected

    # --- Helper: exchange local search ---
    def exchange_search(selected, pool, pool_masks, max_iters=50):
        """Try to reduce size by exchanging blocks."""
        best_size = len(selected)
        best_selected = selected[:]

        for it in range(max_iters):
            if time.time() > deadline - 8:
                break

            improved = False
            n = len(selected)

            for i in range(n):
                if time.time() > deadline - 8:
                    return best_selected, best_size
                idx = selected[i]
                mask_i = pool_masks[idx]

                # Find triples only covered by block i
                other_cover = np.zeros(num_words, dtype=np.uint64)
                for j in range(n):
                    if j != i:
                        other_cover = mask_or(other_cover, pool_masks[selected[j]])

                critical = mask_and_not(mask_i, other_cover)
                if mask_popcount(critical) == 0:
                    # Redundant, remove
                    selected.pop(i)
                    best_size = len(selected)
                    best_selected = selected[:]
                    improved = True
                    break

                # Try to find a pool block that covers all critical triples
                # and adds the most new coverage
                best_replacement = -1
                best_new_gain = -1
                for j in range(len(pool)):
                    if j in selected:
                        continue
                    mask_j = pool_masks[j]
                    # Check if j covers all critical triples
                    if mask_subset(critical, mask_j):
                        # Gain: new triples covered by j not in current cover
                        new_gain = mask_popcount(mask_and_not(mask_j, other_cover))
                        if new_gain > best_new_gain:
                            best_new_gain = new_gain
                            best_replacement = j

                if best_replacement != -1:
                    selected[i] = best_replacement
                    improved = True
                    break

            if not improved:
                break

        return best_selected, best_size

    # --- Phase 1: Build candidate pool ---
    # Start with a quick deterministic greedy to get a baseline
    uncovered_set = set(all_triples)
    baseline_blocks = []
    while uncovered_set and time.time() < deadline - 60:
        first_triple = next(iter(uncovered_set))
        block = list(first_triple)
        while len(block) < k:
            best_point = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for subset in itertools.combinations(block, t - 1):
                    new_triple = tuple(sorted(subset + (x,)))
                    if new_triple in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point is None:
                break
            block.append(best_point)
        block.sort()
        baseline_blocks.append(tuple(block))
        for tr in itertools.combinations(block, t):
            uncovered_set.discard(tr)

    # Build pool from baseline + random blocks
    rng = random.Random(123)
    pool_set = set(baseline_blocks)
    pool = list(pool_set)

    # Add random blocks to expand pool
    target_pool = min(400, max(len(pool) + 150, 250))
    attempts = 0
    while len(pool) < target_pool and attempts < 2000:
        attempts += 1
        r_triple = tuple(sorted(rng.sample(range(v), t)))
        block = list(r_triple)
        while len(block) < k:
            remaining = [x for x in range(v) if x not in block]
            if not remaining:
                break
            block.append(rng.choice(remaining))
        block.sort()
        b = tuple(block)
        if b not in pool_set:
            pool_set.add(b)
            pool.append(b)

    # Precompute masks for pool
    pool_masks = [block_to_mask(b) for b in pool]

    # --- Phase 2: Portfolio of randomized weighted greedy runs ---
    best_selected = None
    best_size = float('inf')

    num_runs = 8
    for run_id in range(num_runs):
        if time.time() > deadline - 40:
            break

        # Randomize weights: each triple gets a random weight in [0.5, 2.0]
        # This diversifies which blocks get picked
        seed = 42 + run_id * 137
        run_rng = random.Random(seed)
        weights = np.array([run_rng.uniform(0.5, 2.0) for _ in range(triple_count)], dtype=np.float64)

        selected, covered = weighted_greedy(pool, pool_masks, weights)

        if not masks_equal(covered, FULL_MASK):
            continue

        # Prune
        selected = prune_blocks(selected, pool_masks)

        # Verify still valid after pruning
        cov = np.zeros(num_words, dtype=np.uint64)
        for idx in selected:
            cov = mask_or(cov, pool_masks[idx])
        if not masks_equal(cov, FULL_MASK):
            continue

        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = selected[:]

    # Fallback if no valid solution from portfolio
    if best_selected is None:
        # Use baseline directly
        best_selected = list(range(len(baseline_blocks)))
        best_size = len(best_selected)
        # Verify
        cov = np.zeros(num_words, dtype=np.uint64)
        for idx in best_selected:
            cov = mask_or(cov, pool_masks[idx])
        if not masks_equal(cov, FULL_MASK):
            # Last resort: return baseline
            return [list(b) for b in baseline_blocks]

    # --- Phase 3: Exchange local search on best solution ---
    best_selected, best_size = exchange_search(best_selected, pool, pool_masks, max_iters=80)

    # --- Phase 4: Final prune ---
    best_selected = prune_blocks(best_selected, pool_masks)
    best_size = len(best_selected)

    # --- Phase 5: One more round of exchanges after final prune ---
    if time.time() < deadline - 5:
        best_selected, best_size = exchange_search(best_selected, pool, pool_masks, max_iters=40)
        best_selected = prune_blocks(best_selected, pool_masks)
        best_size = len(best_selected)

    # Final validation
    final_blocks = [list(pool[i]) for i in best_selected]
    cov = np.zeros(num_words, dtype=np.uint64)
    for b in final_blocks:
        cov = mask_or(cov, block_to_mask(b))

    if not masks_equal(cov, FULL_MASK):
        # Fallback
        return [list(b) for b in baseline_blocks]

    return final_blocks
