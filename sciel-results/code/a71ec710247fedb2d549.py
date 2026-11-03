"""
Covering design C(19,8,5) using Lagrangian-guided greedy with NumPy acceleration.
"""
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 110.0

    # Precompute all t-subsets
    t_sets = list(itertools.combinations(range(v), t))
    t_set_index = {ts: i for i, ts in enumerate(t_sets)}
    num_t_sets = len(t_sets)

    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # Build block_cover_lists as NumPy arrays of t-set indices
    # This avoids the memory overhead of the dense matrix and allows fast sparse sum
    block_cover_lists = []
    for block in all_blocks:
        indices = [t_set_index[ts] for ts in itertools.combinations(block, t)]
        block_cover_lists.append(np.array(indices, dtype=np.int32))

    # Build t_set_to_blocks
    t_set_to_blocks = [[] for _ in range(num_t_sets)]
    for b_idx, cover_list in enumerate(block_cover_lists):
        for ts_idx in cover_list:
            t_set_to_blocks[ts_idx].append(b_idx)

    best_solution = None
    best_size = float('inf')

    # Multiple strategies with different parameters
    strategies = [
        (1, 42, 1.0, 0.1),
        (5, 101, 1.0, 0.1),
        (10, 102, 2.0, 0.2),
        (20, 103, 1.5, 0.15),
        (50, 104, 1.0, 0.05),
        (10, 201, 3.0, 0.3),
        (20, 202, 2.0, 0.2),
        (50, 203, 1.5, 0.1),
        (1, 301, 5.0, 0.5),
        (5, 302, 4.0, 0.4),
        (10, 303, 3.0, 0.3),
        (20, 304, 2.5, 0.25),
        (5, 401, 1.0, 0.05),
        (10, 402, 1.5, 0.1),
        (20, 403, 2.0, 0.15),
        (50, 404, 1.0, 0.08),
    ]

    for beam_width, seed, lambda_init, lambda_step in strategies:
        if time.time() > deadline - 15.0:
            break

        sol = _lagrangian_greedy_cover(
            v, k, t, t_set_index, all_blocks, block_cover_lists,
            num_t_sets, t_set_to_blocks, deadline,
            rng_seed=seed, beam_width=beam_width,
            lambda_init=lambda_init, lambda_step=lambda_step
        )
        if sol:
            if len(sol) < best_size:
                best_size = len(sol)
                best_solution = sol

    if best_solution is None:
        return []

    # Post-processing: local search refinement
    best_solution = _local_search_refine(best_solution, all_blocks, block_cover_lists, t_set_index, num_t_sets, deadline, rng_seed=123)

    # Ensure blocks are lists
    return [list(b) for b in best_solution]


def _lagrangian_greedy_cover(v, k, t, t_set_index, all_blocks, block_cover_lists,
                              num_t_sets, t_set_to_blocks, deadline,
                              rng_seed=42, beam_width=1, lambda_init=1.0, lambda_step=0.1):
    """
    Lagrangian-relaxation-guided greedy set cover with NumPy-accelerated scoring.
    """
    rng = random.Random(rng_seed)
    uncovered = np.ones(num_t_sets, dtype=np.bool_)
    selected_blocks = []

    # Penalty weights
    w = np.full(num_t_sets, lambda_init, dtype=np.float64)

    while uncovered.any():
        if time.time() > deadline - 3.0:
            return None

        # Pick target t-sets to guide search
        uncovered_indices = np.where(uncovered)[0]
        if len(uncovered_indices) <= beam_width:
            targets = uncovered_indices
        else:
            targets = rng.sample(list(uncovered_indices), beam_width)

        # Find candidate blocks: for each target, find the best block covering it
        candidate_blocks = {}
        for target_tset_idx in targets:
            best_block_idx = -1
            max_score = -1.0

            for b_idx in t_set_to_blocks[target_tset_idx]:
                indices = block_cover_lists[b_idx]
                # Fast score: sum of weights where uncovered
                score = w[indices].sum() - w[indices][~uncovered[indices]].sum()
                if score > max_score:
                    max_score = score
                    best_block_idx = b_idx

            if best_block_idx != -1:
                if best_block_idx not in candidate_blocks or max_score > candidate_blocks[best_block_idx]:
                    candidate_blocks[best_block_idx] = max_score

        # Pick best block among candidates
        final_best_block_idx = -1
        final_max_score = -1.0

        for b_idx, score in candidate_blocks.items():
            if score > final_max_score:
                final_max_score = score
                final_best_block_idx = b_idx

        if final_best_block_idx == -1:
            break

        selected_blocks.append(all_blocks[final_best_block_idx])

        # Update uncovered
        covered_indices = block_cover_lists[final_best_block_idx]
        uncovered[covered_indices] = False

        # Update Lagrangian multipliers: increase weights for still-uncovered, decrease for covered
        still_uncovered = uncovered[covered_indices]
        if still_uncovered.any():
            w[covered_indices[still_uncovered]] *= (1.0 + lambda_step)

        newly_covered = ~still_uncovered
        if newly_covered.any():
            w[covered_indices[newly_covered]] *= (1.0 - lambda_step * 0.5)

        w = np.maximum(w, 0.1)

    return selected_blocks


def _local_search_refine(blocks, all_blocks, block_cover_lists, t_set_index, num_t_sets, deadline, rng_seed=42):
    """
    Local search refinement: try dropping blocks and swapping blocks to reduce total count.
    """
    if not blocks:
        return blocks

    rng = random.Random(rng_seed)
    current_blocks = list(blocks)

    # Build coverage count
    coverage = np.zeros(num_t_sets, dtype=np.int32)
    block_set = set()
    for block in current_blocks:
        b_idx = t_set_index.get(tuple(sorted(block)), None)
        # We need to map block to its index in all_blocks
        pass

    # Rebuild: map each block to its index in all_blocks
    all_blocks_set = {tuple(b): i for i, b in enumerate(all_blocks)}
    current_indices = []
    for block in current_blocks:
        idx = all_blocks_set.get(tuple(block), None)
        if idx is not None:
            current_indices.append(idx)
        else:
            return blocks  # Can't map, return original

    # Build coverage
    coverage = np.zeros(num_t_sets, dtype=np.int32)
    for b_idx in current_indices:
        for idx in block_cover_lists[b_idx]:
            coverage[idx] += 1

    improved = True
    max_iterations = 50
    iteration = 0

    while improved and iteration < max_iterations:
        iteration += 1
        if time.time() > deadline - 10.0:
            break
        improved = False

        # Try dropping each block
        for i in range(len(current_indices) - 1, -1, -1):
            if time.time() > deadline - 10.0:
                break

            b_idx = current_indices[i]
            # Check if any t-set has coverage 1 and is covered by this block
            can_remove = True
            for idx in block_cover_lists[b_idx]:
                if coverage[idx] == 1:
                    can_remove = False
                    break

            if can_remove:
                for idx in block_cover_lists[b_idx]:
                    coverage[idx] -= 1
                current_indices.pop(i)
                improved = True
                break

        if improved:
            continue

        # Try swapping: remove one block, add another that covers what was lost
        if len(current_indices) > 20:
            # Sample a few blocks to try removing
            sample_size = min(5, len(current_indices))
            candidates = rng.sample(range(len(current_indices)), sample_size)

            for i in candidates:
                if time.time() > deadline - 10.0:
                    break

                b_idx = current_indices[i]
                # Find t-sets uniquely covered by this block
                unique_tsets = []
                for idx in block_cover_lists[b_idx]:
                    if coverage[idx] == 1:
                        unique_tsets.append(idx)

                if not unique_tsets:
                    # Safe to remove, then find best replacement
                    for idx in block_cover_lists[b_idx]:
                        coverage[idx] -= 1

                    # Find best block to add back
                    best_new_block = -1
                    best_gain = -1.0

                    # Look at blocks that cover at least one of the unique tsets
                    for ts_idx in unique_tsets[:3]:  # Limit to first 3 for speed
                        for nb_idx in t_set_to_blocks[ts_idx]:
                            if nb_idx in current_indices:
                                continue
                            gain = 0
                            for idx in block_cover_lists[nb_idx]:
                                if coverage[idx] == 0:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_new_block = nb_idx

                    if best_new_block != -1 and best_gain > 0:
                        # Do the swap
                        current_indices[i] = best_new_block
                        for idx in block_cover_lists[best_new_block]:
                            coverage[idx] += 1
                        improved = True
                    else:
                        # Undo removal
                        for idx in block_cover_lists[b_idx]:
                            coverage[idx] += 1

    if not current_indices:
        return blocks

    return [all_blocks[b_idx] for b_idx in current_indices]


def _remove_redundant_blocks(blocks, block_cover_lists, all_blocks, deadline):
    """Remove blocks that don't cover any t-set uniquely."""
    if not blocks:
        return blocks

    all_blocks_set = {frozenset(b): i for i, b in enumerate(all_blocks)}
    current_block_indices = []
    for block in blocks:
        block_set = frozenset(block)
        if block_set in all_blocks_set:
            current_block_indices.append(all_blocks_set[block_set])

    if len(current_block_indices) != len(blocks):
        return blocks

    # Determine num_t_sets from the maximum index in block_cover_lists
    max_idx = 0
    for bl in block_cover_lists:
        if len(bl) > 0:
            max_idx = max(max_idx, np.max(bl))
    num_t_sets = max_idx + 1
    
    if num_t_sets == 0:
        return blocks

    # Count coverage
    coverage = np.zeros(num_t_sets, dtype=np.int32)
    for b_idx in current_block_indices:
        for idx in block_cover_lists[b_idx]:
            coverage[idx] += 1

    # Iteratively remove redundant blocks
    improved = True
    while improved:
        if time.time() > deadline - 10.0:
            break
        improved = False
        
        # Iterate in reverse to handle indices safely if popping
        # We check if any t-set has coverage 1 and is covered by this block.
        # If so, the block is critical and cannot be removed.
        for i in range(len(current_block_indices) - 1, -1, -1):
            if time.time() > deadline - 10.0:
                break
            
            b_idx = current_block_indices[i]
            can_remove = True
            for idx in block_cover_lists[b_idx]:
                if coverage[idx] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove it
                for idx in block_cover_lists[b_idx]:
                    coverage[idx] -= 1
                current_block_indices.pop(i)
                improved = True
                break # Restart loop to re-evaluate others

    if not current_block_indices:
        return blocks

    return [all_blocks[b_idx] for b_idx in current_block_indices]
