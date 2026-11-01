import itertools
import numpy as np
import time
import random
import sys
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill

    def time_left():
        return deadline - time.time()

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    t_sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}

    # Precompute all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)

    # For each block, compute which t-subsets it covers
    block_to_tsubs = []
    for block in all_k_blocks:
        subs = [t_sub_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_to_tsubs.append(subs)

    # For each t-subset, compute which blocks cover it
    t_sub_to_blocks = [[] for _ in range(n_t_subs)]
    for b_idx, subs in enumerate(block_to_tsubs):
        for s_idx in subs:
            t_sub_to_blocks[s_idx].append(b_idx)

    # Degrees: how many blocks cover each t-subset
    t_sub_degrees = [len(lst) for lst in t_sub_to_blocks]

    # Scarcity weight for each block: sum of 1/degree for t-subsets it covers
    block_weights = [0.0] * n_k_blocks
    for b_idx, subs in enumerate(block_to_tsubs):
        w = 0.0
        for s_idx in subs:
            if t_sub_degrees[s_idx] > 0:
                w += 1.0 / t_sub_degrees[s_idx]
        block_weights[b_idx] = w

    def weighted_packing(random_seed):
        """
        Phase 1: Select a set of non-conflicting blocks (no shared t-subsets)
        weighted by scarcity. This gives us a 'core' that covers hard parts.
        """
        rng = random.Random(random_seed)
        pool = set(range(n_k_blocks))
        core_blocks = []
        uncovered_mask = [1] * n_t_subs

        while pool and time_left() > 5:
            best_block = -1
            max_score = -1.0

            # Sample a subset of the pool for efficiency if pool is large
            if len(pool) > 5000:
                candidates = set(rng.sample(list(pool), 5000))
            else:
                candidates = pool

            for b_idx in candidates:
                cov_uncovered = 0
                scarcity_sum = 0.0
                for s_idx in block_to_tsubs[b_idx]:
                    if uncovered_mask[s_idx]:
                        cov_uncovered += 1
                        scarcity_sum += 1.0 / t_sub_degrees[s_idx]

                if cov_uncovered == 0:
                    continue

                # Score: prioritize covering uncovered, tie-break by scarcity
                score = cov_uncovered * 1000.0 + scarcity_sum

                if score > max_score:
                    max_score = score
                    best_block = b_idx

            if best_block == -1:
                break

            core_blocks.append(best_block)

            # Update uncovered mask
            for s_idx in block_to_tsubs[best_block]:
                uncovered_mask[s_idx] = 0

            # Remove best_block and all blocks that share a t-subset with it
            neighbors_to_remove = set()
            for s_idx in block_to_tsubs[best_block]:
                neighbors_to_remove.update(t_sub_to_blocks[s_idx])
            pool.difference_update(neighbors_to_remove)

        return core_blocks, uncovered_mask

    def complete_coverage(core_blocks, uncovered_mask):
        """
        Phase 2: Greedily add blocks to cover remaining uncovered t-subsets.
        """
        active_blocks = set(core_blocks)
        cover_count = [0] * n_t_subs

        for b_idx in active_blocks:
            for s_idx in block_to_tsubs[b_idx]:
                cover_count[s_idx] += 1

        still_uncovered = set(i for i in range(n_t_subs) if cover_count[i] == 0)

        if not still_uncovered:
            return active_blocks, cover_count

        while still_uncovered and time_left() > 3:
            best_block = -1
            max_gain = 0

            # Consider all blocks not already in active
            for b_idx in range(n_k_blocks):
                if b_idx in active_blocks:
                    continue
                gain = 0
                for s_idx in block_to_tsubs[b_idx]:
                    if s_idx in still_uncovered:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block = b_idx

            if best_block == -1 or max_gain == 0:
                break

            active_blocks.add(best_block)
            for s_idx in block_to_tsubs[best_block]:
                if s_idx in still_uncovered:
                    still_uncovered.discard(s_idx)
                cover_count[s_idx] += 1

        return active_blocks, cover_count

    def minimize_blocks(active_blocks, cover_count):
        """
        Remove redundant blocks iteratively.
        """
        changed = True
        while changed and time_left() > 2:
            changed = False
            items = list(active_blocks)
            random.shuffle(items)
            for b_idx in items:
                if b_idx not in active_blocks:
                    continue
                can_remove = True
                for s_idx in block_to_tsubs[b_idx]:
                    if cover_count[s_idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    active_blocks.discard(b_idx)
                    for s_idx in block_to_tsubs[b_idx]:
                        cover_count[s_idx] -= 1
                    changed = True
        return active_blocks, cover_count

    def local_search_swaps(active_blocks, cover_count, max_iter=50):
        """
        Try to swap blocks to enable further removals.
        """
        current_count = len(active_blocks)
        for _ in range(max_iter):
            if time_left() < 3:
                break
            if len(active_blocks) >= current_count:
                break

            # Try random swaps
            improved = False
            for _ in range(20):
                if time_left() < 3:
                    break
                if not active_blocks:
                    break

                old_block = random.choice(list(active_blocks))
                # Find candidate new blocks
                candidates = [b for b in range(n_k_blocks) if b not in active_blocks]
                if not candidates:
                    break
                new_block = random.choice(candidates)

                # Check if swap is valid: new_block must cover all uniquely-covered t-subsets of old_block
                unique_subs = [s_idx for s_idx in block_to_tsubs[old_block] if cover_count[s_idx] == 1]
                new_subs_set = set(block_to_tsubs[new_block])
                valid = all(s_idx in new_subs_set for s_idx in unique_subs)

                if not valid:
                    continue

                # Perform swap
                active_blocks.discard(old_block)
                active_blocks.add(new_block)
                for s_idx in block_to_tsubs[old_block]:
                    cover_count[s_idx] -= 1
                for s_idx in block_to_tsubs[new_block]:
                    cover_count[s_idx] += 1

                # Try to minimize again
                active_blocks, cover_count = minimize_blocks(active_blocks, cover_count)

                if len(active_blocks) < current_count:
                    current_count = len(active_blocks)
                    improved = True
                    break

                # Revert if no improvement
                active_blocks.discard(new_block)
                active_blocks.add(old_block)
                for s_idx in block_to_tsubs[new_block]:
                    cover_count[s_idx] -= 1
                for s_idx in block_to_tsubs[old_block]:
                    cover_count[s_idx] += 1

            if not improved:
                break

        return active_blocks, cover_count

    def run_iteration(seed):
        """
        One full iteration: packing + completion + minimization + local search.
        """
        core, uncovered_mask = weighted_packing(seed)
        active_blocks, cover_count = complete_coverage(core, uncovered_mask)
        active_blocks, cover_count = minimize_blocks(active_blocks, cover_count)
        active_blocks, cover_count = local_search_swaps(active_blocks, cover_count)
        return active_blocks

    best_blocks = None
    best_count = float('inf')

    # Run multiple iterations with different seeds
    num_iterations = 8
    for i in range(num_iterations):
        if time_left() < 10:
            break

        blocks = run_iteration(i * 42 + 7)

        if blocks is not None and len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    # Final minimization on the best
    if best_blocks is not None:
        cover_count = [0] * n_t_subs
        for b_idx in best_blocks:
            for s_idx in block_to_tsubs[b_idx]:
                cover_count[s_idx] += 1
        best_blocks, cover_count = minimize_blocks(best_blocks, cover_count)
        best_blocks, cover_count = local_search_swaps(best_blocks, cover_count, max_iter=30)

    # Convert indices back to actual blocks
    if best_blocks:
        result = [list(all_k_blocks[b_idx]) for b_idx in best_blocks]
        return result
    else:
        return []
