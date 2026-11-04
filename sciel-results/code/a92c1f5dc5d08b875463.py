import itertools
import time
import random
import numpy as np
from collections import defaultdict, deque

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 40s safety margin for 150s limit

    # Precompute all t-subsets and blocks as bitmasks
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsubset_mask_to_idx = {}
    for i, ts in enumerate(all_tsubsets):
        mask = 0
        for p in ts:
            mask |= (1 << p)
        tsubset_mask_to_idx[mask] = i

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Precompute which t-subset indices each block covers
    block_covered_ts_indices = []
    for blk in all_blocks:
        covered_indices = []
        for ts in itertools.combinations(blk, t):
            ts_mask = 0
            for p in ts:
                ts_mask |= (1 << p)
            idx = tsubset_mask_to_idx[ts_mask]
            covered_indices.append(idx)
        block_covered_ts_indices.append(covered_indices)

    # Precompute ts_to_blocks
    ts_to_blocks = [[] for _ in range(n_tsubsets)]
    for b_idx, ts_list in enumerate(block_covered_ts_indices):
        for ts_idx in ts_list:
            ts_to_blocks[ts_idx].append(b_idx)

    def deletion_based_reduce(seed_val):
        """Start with all blocks, iteratively remove redundant ones."""
        random.seed(seed_val)
        # counts[i] = number of selected blocks covering t-subset i
        counts = np.zeros(n_tsubsets, dtype=np.int16)
        selected = [True] * n_blocks

        # Initialize counts
        for b_idx in range(n_blocks):
            for ts_idx in block_covered_ts_indices[b_idx]:
                counts[ts_idx] += 1

        # Repeatedly remove redundant blocks
        # A block is redundant if all ts it covers have count > 1
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            # Randomize order for diversity
            order = list(range(n_blocks))
            random.shuffle(order)
            for b_idx in order:
                if not selected[b_idx]:
                    continue
                # Check if block is redundant
                redundant = True
                for ts_idx in block_covered_ts_indices[b_idx]:
                    if counts[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    selected[b_idx] = False
                    for ts_idx in block_covered_ts_indices[b_idx]:
                        counts[ts_idx] -= 1
                    changed = True
                    if time.time() > deadline:
                        break

        result = [b_idx for b_idx in range(n_blocks) if selected[b_idx]]
        return result, counts, selected

    def swap_local_search(block_list, counts, selected, seed_val, max_iters=2000):
        """Try to swap blocks: remove one, add another that helps more."""
        random.seed(seed_val)
        n_sel = len(block_list)
        if n_sel == 0:
            return block_list, counts, selected

        for it in range(max_iters):
            if time.time() > deadline:
                break
            if it % 100 == 0 and time.time() > deadline:
                break

            # Try to find a block to remove and a replacement
            # Pick a random selected block
            rem_pos = random.randint(0, n_sel - 1)
            rem_idx = block_list[rem_pos]
            rem_ts = block_covered_ts_indices[rem_idx]

            # Check if removing it would leave any ts uncovered
            would_uncover = False
            for ts_idx in rem_ts:
                if counts[ts_idx] == 1:
                    would_uncover = True
                    break
            if would_uncover:
                continue

            # Temporarily remove
            for ts_idx in rem_ts:
                counts[ts_idx] -= 1

            # Find a block that covers the most newly-uncovered ts
            # (ts that now have count 0)
            best_add = -1
            best_gain = 0
            # Sample candidate blocks
            n_candidates = min(50, n_blocks)
            candidates = random.sample(range(n_blocks), n_candidates)
            for b_idx in candidates:
                if selected[b_idx]:
                    continue
                gain = 0
                for ts_idx in block_covered_ts_indices[b_idx]:
                    if counts[ts_idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_add = b_idx

            if best_add == -1 or best_gain == 0:
                # Restore
                for ts_idx in rem_ts:
                    counts[ts_idx] += 1
                continue

            # Perform swap: remove rem_idx, add best_add
            for ts_idx in block_covered_ts_indices[best_add]:
                counts[ts_idx] += 1
            selected[rem_idx] = False
            selected[best_add] = True
            block_list[rem_pos] = best_add

        return block_list, counts, selected

    def greedy_add_reduce(seed_val):
        """Greedy add blocks, then reduce."""
        random.seed(seed_val)
        counts = np.zeros(n_tsubsets, dtype=np.int8)
        selected_block_indices = []
        remaining = set(range(n_tsubsets))

        while remaining:
            if time.time() > deadline:
                break

            pivot_ts = random.choice(tuple(remaining))
            best_block_idx = -1
            best_gain = -1

            candidates = ts_to_blocks[pivot_ts]
            for b_idx in candidates:
                gain = 0
                for ts_idx in block_covered_ts_indices[b_idx]:
                    if ts_idx in remaining:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx

            if best_block_idx == -1:
                break

            selected_block_indices.append(best_block_idx)
            for ts_idx in block_covered_ts_indices[best_block_idx]:
                counts[ts_idx] += 1
                if counts[ts_idx] == 1:
                    remaining.discard(ts_idx)

        # Reduction phase
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for i in range(len(selected_block_indices) - 1, -1, -1):
                b_idx = selected_block_indices[i]
                is_redundant = True
                for ts_idx in block_covered_ts_indices[b_idx]:
                    if counts[ts_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    selected_block_indices.pop(i)
                    for ts_idx in block_covered_ts_indices[b_idx]:
                        counts[ts_idx] -= 1
                    changed = True

        return selected_block_indices

    best_solution = None
    best_len = float('inf')

    # Phase 1: Deletion-based approach (start with all blocks, remove redundant)
    # This should converge quickly to a small covering
    for seed in range(5):
        if time.time() > deadline:
            break
        sol, counts, selected = deletion_based_reduce(seed * 7777)
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol[:]
            # Store counts and selected for swap search
            best_counts = counts.copy()
            best_selected = selected[:]

    # Phase 2: Swap-based local search on the best solution
    if best_solution is not None and time.time() < deadline:
        for seed in range(10):
            if time.time() > deadline:
                break
            sol_copy = best_solution[:]
            counts_copy = best_counts.copy()
            selected_copy = best_selected[:]
            sol_copy, counts_copy, selected_copy = swap_local_search(
                sol_copy, counts_copy, selected_copy, seed * 3333, max_iters=1500
            )
            if len(sol_copy) < best_len:
                best_len = len(sol_copy)
                best_solution = sol_copy[:]
                best_counts = counts_copy
                best_selected = selected_copy

    # Phase 3: Greedy add approach with different seeds
    for seed in range(10):
        if time.time() > deadline:
            break
        sol = greedy_add_reduce(seed * 1111 + 500)
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol[:]

    # Convert block indices back to lists of points
    final_blocks = []
    if best_solution:
        for b_idx in best_solution:
            final_blocks.append(list(all_blocks[b_idx]))
    else:
        # Fallback
        sol = greedy_add_reduce(0)
        for b_idx in sol:
            final_blocks.append(list(all_blocks[b_idx]))

    return final_blocks
