import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill

    # 1. Precompute all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)

    # 2. Generate all k-subsets (blocks) and their coverage masks
    k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(k_subsets)

    # Precompute for each block which t-subsets it covers
    block_ts_indices = []
    for block in k_subsets:
        ts_indices = []
        for t_sub in itertools.combinations(block, t):
            ts_indices.append(t_subset_to_idx[t_sub])
        block_ts_indices.append(ts_indices)

    # Build bitmasks for fast coverage checking
    block_masks = []
    for ts_indices in block_ts_indices:
        mask = 0
        for idx in ts_indices:
            mask |= (1 << idx)
        block_masks.append(mask)

    full_mask = (1 << num_t_subsets) - 1

    best_blocks = None
    best_count = float('inf')

    def greedy_cover(noise_scale=0.0):
        """Run greedy set cover with optional random noise for tie-breaking."""
        uncovered = full_mask
        selected = []

        while uncovered:
            best_gain = -1
            best_idx = -1

            for i in range(num_blocks):
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > 0:
                    # Add small random noise to break ties and explore
                    noisy_gain = gain + random.random() * noise_scale
                    if noisy_gain > best_gain:
                        best_gain = noisy_gain
                        best_idx = i

            if best_idx == -1:
                break  # Should not happen if uncovered != 0

            selected.append(best_idx)
            uncovered &= ~block_masks[best_idx]

            if time.time() > deadline:
                return selected, True  # True indicates timeout

        return selected, False

    def packing_first_build(seed=0):
        """
        Packing-First strategy:
        Phase 1: Build a high-efficiency core by greedily selecting blocks that
                 maximize NEW coverage (minimizing overlap with already-selected).
                 This creates a structurally efficient packing.
        Phase 2: Repair remaining uncovered t-subsets with targeted additions.
        """
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        selected_set = set()

        # Phase 1: Packing-first core build
        # We want to maximize new coverage per block, which naturally minimizes overlap.
        # Use a slightly different noise profile to encourage diversity in the core.
        core_noise_scales = [0.0, 0.3, 0.7, 1.5]
        core_noise = core_noise_scales[seed % len(core_noise_scales)]

        while uncovered:
            best_gain = -1
            best_idx = -1
            best_noisy = -1

            for i in range(num_blocks):
                if i in selected_set:
                    continue
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > 0:
                    noisy_gain = gain + rng.random() * core_noise
                    if noisy_gain > best_noisy:
                        best_noisy = noisy_gain
                        best_gain = gain
                        best_idx = i

            if best_idx == -1:
                break

            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~block_masks[best_idx]

            if time.time() > deadline:
                return selected, True

        # Phase 2: Should not have uncovered if all t-subsets are covered by some block,
        # but due to the greedy nature, we verify and repair if needed.
        # In practice, for C(19,7,4), every 4-subset is contained in some 7-subset,
        # so the greedy should cover everything. But let's be safe.
        if uncovered:
            # Repair phase: add blocks to cover remaining gaps
            while uncovered:
                best_gain = -1
                best_idx = -1
                for i in range(num_blocks):
                    if i in selected_set:
                        continue
                    gain = (uncovered & block_masks[i]).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                if best_idx == -1:
                    break
                selected.append(best_idx)
                selected_set.add(best_idx)
                uncovered &= ~block_masks[best_idx]
                if time.time() > deadline:
                    return selected, True

        return selected, False

    def remove_redundant(selected):
        """Remove blocks that don't uniquely cover any t-subset."""
        if not selected:
            return selected

        # Count how many times each t-subset is covered
        counts = np.zeros(num_t_subsets, dtype=np.int32)
        for idx in selected:
            for ts_idx in block_ts_indices[idx]:
                counts[ts_idx] += 1

        active = [True] * len(selected)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(selected)):
                if not active[i]:
                    continue
                # Check if this block has any uniquely covered t-subset
                has_unique = False
                for ts_idx in block_ts_indices[selected[i]]:
                    if counts[ts_idx] == 1:
                        has_unique = True
                        break
                if not has_unique:
                    active[i] = False
                    changed = True
                    for ts_idx in block_ts_indices[selected[i]]:
                        counts[ts_idx] -= 1

        return [selected[i] for i in range(len(selected)) if active[i]]

    def local_search_swap(selected, max_swaps=500):
        """Try to swap blocks to enable removal of other blocks."""
        if not selected:
            return selected

        selected = list(selected)
        n = len(selected)

        # Build counts
        counts = np.zeros(num_t_subsets, dtype=np.int32)
        for idx in selected:
            for ts_idx in block_ts_indices[idx]:
                counts[ts_idx] += 1

        improved = True
        swap_count = 0
        while improved and swap_count < max_swaps and time.time() < deadline:
            improved = False
            for i in range(len(selected)):
                if time.time() > deadline:
                    return selected

                # Find t-subsets uniquely covered by block i
                unique_ts = []
                for ts_idx in block_ts_indices[selected[i]]:
                    if counts[ts_idx] == 1:
                        unique_ts.append(ts_idx)

                if not unique_ts:
                    # Can remove block i directly
                    for ts_idx in block_ts_indices[selected[i]]:
                        counts[ts_idx] -= 1
                    selected.pop(i)
                    improved = True
                    swap_count += 1
                    break

                # Try to find a replacement block that covers all unique_ts
                # and allows removal of another block
                for j in range(len(selected)):
                    if j == i:
                        continue

                    # Check if block j covers all unique t-subsets of block i
                    covers_all = True
                    for ts_idx in unique_ts:
                        if ts_idx not in block_ts_indices[selected[j]]:
                            covers_all = False
                            break

                    if not covers_all:
                        continue

                    # Simulate: remove block i, add block j
                    # Then check if we can remove any other block

                    # Temporarily update counts
                    for ts_idx in block_ts_indices[selected[i]]:
                        counts[ts_idx] -= 1
                    for ts_idx in block_ts_indices[selected[j]]:
                        counts[ts_idx] += 1

                    # Check if any block (other than i and j) is now redundant
                    removable = -1
                    for m in range(len(selected)):
                        if m == i or m == j:
                            continue
                        is_redundant = True
                        for ts_idx in block_ts_indices[selected[m]]:
                            if counts[ts_idx] == 1:
                                is_redundant = False
                                break
                        if is_redundant:
                            removable = m
                            break

                    if removable != -1:
                        # Perform the swap and removal
                        # Remove block i
                        selected.pop(i)
                        # Now j's index may have shifted
                        if j > i:
                            j -= 1
                        # Remove the redundant block
                        if removable > j:
                            removable -= 1
                        for ts_idx in block_ts_indices[selected[removable]]:
                            counts[ts_idx] -= 1
                        selected.pop(removable)
                        improved = True
                        swap_count += 1
                        break
                    else:
                        # Revert the temporary count changes
                        for ts_idx in block_ts_indices[selected[i]]:
                            counts[ts_idx] += 1
                        for ts_idx in block_ts_indices[selected[j]]:
                            counts[ts_idx] -= 1
                    if improved:
                        break

                if improved:
                    break

        return selected

    # Run multiple strategies: packing-first with different seeds, plus original greedy
    # Packing-first first (3 variants), then original greedy (3 variants)
    strategies = [
        ('packing', 0), ('packing', 1), ('packing', 2),
        ('greedy', 0.0), ('greedy', 0.5), ('greedy', 1.0),
    ]

    for strat, param in strategies:
        if time.time() > deadline:
            break

        if strat == 'packing':
            selected, timed_out = packing_first_build(param)
        else:
            selected, timed_out = greedy_cover(param)

        if timed_out:
            continue

        # Remove redundant blocks
        selected = remove_redundant(selected)

        # Local search
        selected = local_search_swap(selected)

        # Final redundancy removal
        selected = remove_redundant(selected)

        if len(selected) < best_count:
            best_count = len(selected)
            best_blocks = [k_subsets[i] for i in selected]

            if best_count <= 131:  # Lower bound
                break

    if best_blocks is None:
        # Fallback: just run deterministic greedy
        selected, _ = greedy_cover(0.0)
        selected = remove_redundant(selected)
        best_blocks = [k_subsets[i] for i in selected]

    return best_blocks
