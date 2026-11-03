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

    # Post-processing: remove redundant blocks
    best_solution = _remove_redundant_blocks(best_solution, block_cover_lists, all_blocks, deadline)

    # Ensure blocks are lists
    return [list(b) for b in best_solution]


def _lagrangian_greedy_cover(v, k, t, t_set_index, all_blocks, block_cover_lists,
                              num_t_sets, t_set_to_blocks, deadline,
                              rng_seed=42, beam_width=1, lambda_init=1.0, lambda_step=0.1):
    """
    Lagrangian-relaxation-guided greedy set cover with NumPy-accelerated scoring.
    """
    rng = random.Random(rng_seed)
    uncovered = set(range(num_t_sets))
    selected_blocks = []

    # Penalty weights
    w = np.full(num_t_sets, lambda_init, dtype=np.float64)

    while uncovered:
        if time.time() > deadline - 3.0:
            return None

        # Pick target t-sets to guide search
        if len(uncovered) <= beam_width:
            targets = list(uncovered)
        else:
            targets = rng.sample(list(uncovered), beam_width)

        # Find candidate blocks
        candidate_blocks = set()
        for target_tset_idx in targets:
            best_block_idx = -1
            max_score = -1.0

            for b_idx in t_set_to_blocks[target_tset_idx]:
                # Sparse score computation: sum of weights of covered t-sets
                indices = block_cover_lists[b_idx]
                score = w[indices].sum()
                if score > max_score:
                    max_score = score
                    best_block_idx = b_idx

            if best_block_idx != -1:
                candidate_blocks.add(best_block_idx)

        # Pick best block among candidates
        final_best_block_idx = -1
        final_max_score = -1.0

        for b_idx in candidate_blocks:
            indices = block_cover_lists[b_idx]
            score = w[indices].sum()
            if score > final_max_score:
                final_max_score = score
                final_best_block_idx = b_idx

        if final_best_block_idx == -1:
            break

        selected_blocks.append(all_blocks[final_best_block_idx])

        # Update uncovered
        covered_indices = block_cover_lists[final_best_block_idx]
        for idx in covered_indices:
            uncovered.discard(idx)

        # Update Lagrangian multipliers
        if uncovered:
            uncovered_list = list(uncovered)
            w[uncovered_list] *= (1.0 + lambda_step)
        
        covered_list = covered_indices.tolist()
        w[covered_list] *= (1.0 - lambda_step * 0.5)
        w = np.maximum(w, 0.1)

    return selected_blocks


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
