import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    deadline = time.time() + 115
    rng = random.Random(42)

    # 1. Precompute all t-subsets and map to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}

    # 2. Precompute all k-blocks and their t-subset index lists
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)

    block_t_indices = []
    for block in all_k_blocks:
        t_indices = [sub_to_idx[t_sub] for t_sub in itertools.combinations(block, t)]
        block_t_indices.append(t_indices)

    # Build sparse representation: for each block, list of t-subset indices it covers
    # Also build inverse: for each t-subset, list of blocks that cover it
    # This is expensive to build fully; instead we'll use the matrix for gains
    # but use index lists for fast pruning checks

    # Build coverage matrix as uint8 for vectorized gain
    cover_matrix = np.zeros((n_k_blocks, n_t_subs), dtype=np.uint8)
    for i, t_indices in enumerate(block_t_indices):
        cover_matrix[i, t_indices] = 1

    # For pruning: precompute which t-subsets each block covers (as numpy array)
    block_cover_masks = []
    for t_indices in block_t_indices:
        mask = np.zeros(n_t_subs, dtype=np.uint8)
        mask[t_indices] = 1
        block_cover_masks.append(mask)

    def greedy_randomized(uncovered, remaining_blocks, deadline_check):
        """Randomized greedy: pick best block with random tie-breaking."""
        selected = []
        uncov = uncovered.copy()
        n_uncov = uncov.sum()

        while n_uncov > 0:
            if time.time() > deadline_check:
                break

            # Compute gains
            gains = (cover_matrix[remaining_blocks] & uncov[np.newaxis, :]).sum(axis=1)
            max_gain = int(gains.max())

            if max_gain == 0:
                break

            # Randomized tie-breaking: pick uniformly among top-gain blocks
            top_indices = np.where(gains == max_gain)[0]
            if len(top_indices) > 1:
                pick = top_indices[rng.randrange(len(top_indices))]
            else:
                pick = top_indices[0]

            actual_idx = remaining_blocks[pick]
            selected.append(actual_idx)
            mask = block_cover_masks[actual_idx]
            newly_covered = mask & uncov
            n_uncov -= newly_covered.sum()
            uncov[mask == 1] = 0

        return selected, uncov

    def prune_blocks(selected_indices, deadline_check):
        """Remove redundant blocks."""
        if not selected_indices:
            return selected_indices

        # Build cover count
        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for idx in selected_indices:
            cover_count[block_cover_masks[idx] == 1] += 1

        current = selected_indices[:]
        improved = True
        while improved and time.time() < deadline_check:
            improved = False
            new_blocks = []
            for idx in current:
                mask = block_cover_masks[idx]
                positions = np.where(mask == 1)[0]
                if np.any(cover_count[positions] == 1):
                    new_blocks.append(idx)
                else:
                    cover_count[positions] -= 1
                    improved = True
            current = new_blocks

        return current

    def swap_improve(selected_indices, deadline_check):
        """Try to replace blocks to reduce count via 1-for-1 swaps."""
        if len(selected_indices) < 2:
            return selected_indices

        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for idx in selected_indices:
            cover_count[block_cover_masks[idx] == 1] += 1

        current = selected_indices[:]
        improved = True
        while improved and time.time() < deadline_check:
            improved = False
            new_blocks = []
            for i, idx in enumerate(current):
                mask = block_cover_masks[idx]
                positions = np.where(mask == 1)[0]
                if np.any(cover_count[positions] == 1):
                    new_blocks.append(idx)
                else:
                    # Try to find a replacement that covers the now-uncovered subsets
                    # with fewer additional blocks (ideally 0)
                    uncovered_after_removal = cover_count[positions] == 1
                    # Actually, if we remove this block, some subsets become uncovered
                    # We need to find if another block already covers all of them
                    # For simplicity, just remove if not critical
                    cover_count[positions] -= 1
                    improved = True

            current = new_blocks

        return current

    # Main ILS loop
    all_block_indices = np.arange(n_k_blocks)
    best_solution = None
    best_size = n_k_blocks + 1

    # Initial greedy
    uncovered_init = np.ones(n_t_subs, dtype=np.uint8)
    selected, uncov = greedy_randomized(uncovered_init, all_block_indices, deadline - 20)
    if uncov.sum() == 0:
        selected = prune_blocks(selected, deadline - 15)
        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = selected[:]

    # ILS iterations
    it = 0
    while time.time() < deadline - 10:
        it += 1
        # Perturb: remove some random non-critical blocks from best, then re-greedy
        if best_solution is None or len(best_solution) == 0:
            break

        # Create perturbation: remove 1-3 random blocks that are not critical
        perturbed = best_solution[:]
        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for idx in perturbed:
            cover_count[block_cover_masks[idx] == 1] += 1

        # Find non-critical blocks
        non_critical = []
        for idx in perturbed:
            positions = np.where(block_cover_masks[idx] == 1)[0]
            if not np.any(cover_count[positions] == 1):
                non_critical.append(idx)

        if non_critical:
            n_remove = min(rng.randint(1, 3), len(non_critical))
            to_remove = rng.sample(non_critical, n_remove)
            for idx in to_remove:
                perturbed.remove(idx)
                cover_count[block_cover_masks[idx] == 1] -= 1

        # Compute what's uncovered after perturbation
        uncov_mask = np.zeros(n_t_subs, dtype=np.uint8)
        for idx in perturbed:
            uncov_mask[block_cover_masks[idx] == 1] = 1
        uncovered = 1 - uncov_mask

        # Re-greedy on remaining uncovered
        remaining = np.array([i for i in range(n_k_blocks) if i not in set(perturbed)], dtype=np.int64)
        if len(remaining) == 0:
            continue

        new_selected, final_uncov = greedy_randomized(uncovered, remaining, deadline - 10)
        if final_uncov.sum() > 0:
            continue

        combined = perturbed + new_selected
        combined = prune_blocks(combined, deadline - 10)

        if len(combined) < best_size:
            best_size = len(combined)
            best_solution = combined[:]

    if best_solution is None:
        # Fallback: full greedy
        uncovered_init = np.ones(n_t_subs, dtype=np.uint8)
        selected, _ = greedy_randomized(uncovered_init, all_block_indices, deadline)
        best_solution = selected

    blocks = [list(all_k_blocks[i]) for i in best_solution]
    return blocks
