import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    points = list(range(v))

    # Precompute all t-subsets
    t_subsets = list(itertools.combinations(points, t))
    n_t_subsets = len(t_subsets)
    t_subset_to_idx = {sub: i for i, sub in enumerate(t_subsets)}

    # Generate all k-blocks
    k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(k_blocks)

    # Precompute mapping from block tuple to index
    block_to_idx = {blk: i for i, blk in enumerate(k_blocks)}

    # Build incidence matrix M (n_k_blocks x n_t_subsets)
    M = np.zeros((n_k_blocks, n_t_subsets), dtype=np.int8)
    block_cover_indices = []
    for b, block in enumerate(k_blocks):
        subs = list(itertools.combinations(block, t))
        indices = [t_subset_to_idx[sub] for sub in subs]
        block_cover_indices.append(indices)
        M[b, indices] = 1

    rng = random.Random(42)
    best_blocks = None
    best_size = float('inf')

    # Multi-start with varied strategies
    num_starts = 25
    for i in range(num_starts):
        if time.time() > deadline - 10.0:
            break

        # Vary randomization: deterministic first, then increasingly random
        if i == 0:
            noise_scale = 0.0
        elif i == 1:
            noise_scale = 0.5
        elif i == 2:
            noise_scale = 1.0
        else:
            noise_scale = 2.0 + 0.5 * (i - 3)

        # Randomized greedy using vectorized gains
        uncovered_mask = np.ones(n_t_subsets, dtype=np.int8)
        selected_indices = []

        while np.any(uncovered_mask):
            # Calculate gains: number of uncovered t-subsets each block covers
            gains = M @ uncovered_mask

            # Add noise for randomization
            if noise_scale > 0:
                noise = np.random.default_rng(42 + i * 1000).random(n_k_blocks) * noise_scale
                gains_noisy = gains.astype(np.float64) + noise
                best_block = int(np.argmax(gains_noisy))
            else:
                best_block = int(np.argmax(gains))

            if gains[best_block] == 0:
                break

            selected_indices.append(best_block)

            # Update uncovered mask
            M_row = M[best_block]
            uncovered_mask[M_row == 1] = 0

        # Local search: redundancy removal (vectorized)
        if len(selected_indices) > 0:
            sel_matrix = M[selected_indices]
            counts = sel_matrix.sum(axis=0).copy()

            redundant_found = True
            while redundant_found and time.time() < deadline - 5.0:
                redundant_found = False
                new_selected = []
                for idx in selected_indices:
                    subs = block_cover_indices[idx]
                    if len(subs) > 0:
                        min_count = counts[subs].min()
                    else:
                        min_count = 0
                    if min_count > 1:
                        redundant_found = True
                        counts[subs] -= 1
                    else:
                        new_selected.append(idx)

                if len(new_selected) < len(selected_indices):
                    selected_indices = new_selected

            # Local search: swap-based improvement
            if time.time() < deadline - 5.0:
                improved = True
                while improved and time.time() < deadline - 5.0:
                    improved = False
                    indices_list = list(range(len(selected_indices)))
                    rng.shuffle(indices_list)

                    for i in indices_list:
                        if time.time() > deadline - 5.0:
                            break

                        orig_idx = selected_indices[i]
                        orig_subs = block_cover_indices[orig_idx]

                        # Generate a random new block
                        new_block = tuple(sorted(rng.sample(range(v), k)))
                        new_idx = block_to_idx.get(new_block, None)
                        if new_idx is None:
                            continue

                        new_subs = block_cover_indices[new_idx]

                        # Check if swap is valid: all t-subsets in orig_subs still covered
                        valid = True
                        for s in orig_subs:
                            counts[s] -= 1
                            if counts[s] < 0:
                                valid = False
                                break

                        if not valid:
                            for s in orig_subs:
                                counts[s] += 1
                            continue

                        # Increment new counts
                        for s in new_subs:
                            counts[s] += 1

                        # Check if any t-subset in orig_subs is now uncovered
                        for s in orig_subs:
                            if counts[s] == 0:
                                valid = False
                                break

                        if not valid:
                            for s in orig_subs:
                                counts[s] += 1
                            for s in new_subs:
                                counts[s] -= 1
                            continue

                        # Valid swap: apply it
                        selected_indices[i] = new_idx

                        # Try to remove redundant blocks
                        removed_any = False
                        indices2 = list(range(len(selected_indices)))
                        rng.shuffle(indices2)

                        for j in indices2:
                            if time.time() > deadline - 5.0:
                                break

                            subs_j = block_cover_indices[selected_indices[j]]
                            if len(subs_j) > 0:
                                min_c = counts[subs_j].min()
                            else:
                                min_c = 0

                            if min_c > 1:
                                for s in subs_j:
                                    counts[s] -= 1
                                selected_indices.pop(j)
                                removed_any = True
                                break

                        if removed_any:
                            improved = True
                            break
                        else:
                            # Revert swap if no improvement
                            selected_indices[i] = orig_idx
                            for s in orig_subs:
                                counts[s] -= 1
                            for s in new_subs:
                                counts[s] += 1

            # Update best
            current_len = len(selected_indices)
            if current_len < best_size:
                best_size = current_len
                best_blocks = [list(k_blocks[idx]) for idx in selected_indices]
                if best_size <= 151:
                    break

    if best_blocks is None:
        # Fallback: single deterministic greedy
        uncovered_mask = np.ones(n_t_subsets, dtype=np.int8)
        selected_indices = []
        while np.any(uncovered_mask):
            gains = M @ uncovered_mask
            best_block = int(np.argmax(gains))
            if gains[best_block] == 0:
                break
            selected_indices.append(best_block)
            M_row = M[best_block]
            uncovered_mask[M_row == 1] = 0

        # Redundancy removal
        if len(selected_indices) > 0:
            sel_matrix = M[selected_indices]
            counts = sel_matrix.sum(axis=0).copy()
            redundant_found = True
            while redundant_found:
                redundant_found = False
                new_selected = []
                for idx in selected_indices:
                    subs = block_cover_indices[idx]
                    if len(subs) > 0:
                        min_count = counts[subs].min()
                    else:
                        min_count = 0
                    if min_count > 1:
                        redundant_found = True
                        counts[subs] -= 1
                    else:
                        new_selected.append(idx)
                if len(new_selected) < len(selected_indices):
                    selected_indices = new_selected

        best_blocks = [list(k_blocks[idx]) for idx in selected_indices]

    return best_blocks
