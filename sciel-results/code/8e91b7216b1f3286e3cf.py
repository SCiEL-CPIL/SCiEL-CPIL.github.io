import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill

    # 1. Pre-compute all t-subsets and k-blocks
    t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}

    k_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(k_blocks)

    # For each block, compute its coverage mask and list of t-subset indices
    block_masks = np.zeros(n_blocks, dtype=object)
    block_subsets = [None] * n_blocks

    for b_idx in range(n_blocks):
        block = k_blocks[b_idx]
        subs = []
        mask = 0
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            subs.append(idx)
            mask |= (1 << idx)
        block_masks[b_idx] = mask
        block_subsets[b_idx] = subs

    # 2. Initial greedy construction
    def greedy_construct():
        uncovered = (1 << n_subsets) - 1
        selected = []
        selected_set = set()
        cover_count = [0] * n_subsets

        while uncovered:
            if time.time() > deadline - 10:
                break

            best_block = -1
            best_gain = 0

            for b_idx in range(n_blocks):
                if b_idx in selected_set:
                    continue
                gain = (block_masks[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx

            if best_block == -1:
                break

            selected.append(best_block)
            selected_set.add(best_block)
            uncovered &= ~block_masks[best_block]

            for s in block_subsets[best_block]:
                cover_count[s] += 1

        return selected, selected_set, cover_count

    def remove_redundant(selected, selected_set, cover_count):
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for i in range(len(selected) - 1, -1, -1):
                b_idx = selected[i]
                redundant = True
                for s in block_subsets[b_idx]:
                    if cover_count[s] == 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_subsets[b_idx]:
                        cover_count[s] -= 1
                    selected.pop(i)
                    selected_set.discard(b_idx)
                    changed = True
        return selected, selected_set, cover_count

    # 3. Local search: try to replace pairs of blocks with a single block
    def local_search(selected, selected_set, cover_count, max_iterations=50):
        iterations = 0
        improved = True
        while improved and iterations < max_iterations:
            if time.time() > deadline - 5:
                break
            improved = False
            iterations += 1

            # Try removing each block and see if we can cover its unique subsets
            for i in range(len(selected)):
                if time.time() > deadline - 5:
                    break
                b_idx = selected[i]

                # Find subsets only covered by this block
                unique_subs = []
                for s in block_subsets[b_idx]:
                    if cover_count[s] == 1:
                        unique_subs.append(s)

                if not unique_subs:
                    # Redundant
                    for s in block_subsets[b_idx]:
                        cover_count[s] -= 1
                    selected.pop(i)
                    selected_set.discard(b_idx)
                    improved = True
                    break

                # Try to find a single block that covers all unique subsets
                needed_mask = 0
                for s in unique_subs:
                    needed_mask |= (1 << s)

                best_replacement = -1
                best_extra = 0

                for b2 in range(n_blocks):
                    if b2 in selected_set:
                        continue
                    covers_needed = (block_masks[b2] & needed_mask).bit_count()
                    if covers_needed == len(unique_subs):
                        # This block covers all unique subsets
                        extra = (block_masks[b2] & ~needed_mask).bit_count()
                        if extra > best_extra:
                            best_extra = extra
                            best_replacement = b2

                if best_replacement != -1:
                    # Remove old block
                    for s in block_subsets[b_idx]:
                        cover_count[s] -= 1
                    selected_set.discard(b_idx)

                    # Add new block
                    selected[i] = best_replacement
                    selected_set.add(best_replacement)
                    for s in block_subsets[best_replacement]:
                        cover_count[s] += 1

                    improved = True
                    break

        return selected, selected_set, cover_count

    # 4. Iterated local search with random restarts
    best_selected = None
    best_count = float('inf')

    # First greedy
    selected, selected_set, cover_count = greedy_construct()
    selected, selected_set, cover_count = remove_redundant(selected, selected_set, cover_count)
    selected, selected_set, cover_count = local_search(selected, selected_set, cover_count, max_iterations=30)
    selected, selected_set, cover_count = remove_redundant(selected, selected_set, cover_count)
    best_selected = selected[:]
    best_count = len(selected)

    # Try random restarts with perturbation
    while time.time() < deadline - 15:
        # Perturb: remove some random blocks and rebuild
        if len(best_selected) <= 10:
            break

        perturbation_size = max(1, len(best_selected) // 10)
        perturbed = best_selected[:]
        random.shuffle(perturbed)
        perturbed = perturbed[perturbation_size:]  # Remove first 'perturbation_size' random blocks

        # Build cover_count for perturbed set
        cover_count = [0] * n_subsets
        selected_set = set(perturbed)
        for b_idx in perturbed:
            for s in block_subsets[b_idx]:
                cover_count[s] += 1

        # Check if still valid
        uncovered = 0
        for s_idx in range(n_subsets):
            if cover_count[s_idx] == 0:
                uncovered |= (1 << s_idx)

        # Complete with greedy
        while uncovered and time.time() < deadline - 10:
            best_block = -1
            best_gain = 0
            for b_idx in range(n_blocks):
                if b_idx in selected_set:
                    continue
                gain = (block_masks[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            if best_block == -1:
                break
            perturbed.append(best_block)
            selected_set.add(best_block)
            uncovered &= ~block_masks[best_block]
            for s in block_subsets[best_block]:
                cover_count[s] += 1

        # Local search
        perturbed, selected_set, cover_count = local_search(perturbed, selected_set, cover_count, max_iterations=20)
        perturbed, selected_set, cover_count = remove_redundant(perturbed, selected_set, cover_count)

        if len(perturbed) < best_count:
            best_count = len(perturbed)
            best_selected = perturbed[:]

    # 5. Final answer
    final_blocks = [list(k_blocks[i]) for i in best_selected]
    return final_blocks
