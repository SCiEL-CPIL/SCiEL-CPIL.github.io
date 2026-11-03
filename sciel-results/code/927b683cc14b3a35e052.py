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

    # Build block_cover_sets as sets of t-set indices
    block_cover_sets = []
    for block in all_blocks:
        indices = [t_set_index[ts] for ts in itertools.combinations(block, t)]
        block_cover_sets.append(set(indices))

    # Build t_set_to_blocks
    t_set_to_blocks = [[] for _ in range(num_t_sets)]
    for b_idx, cover_set in enumerate(block_cover_sets):
        for ts_idx in cover_set:
            t_set_to_blocks[ts_idx].append(b_idx)

    # Precompute block cover vectors as numpy arrays for fast scoring
    block_cover_vecs = np.zeros((num_blocks, num_t_sets), dtype=np.int8)
    for b_idx, cover_set in enumerate(block_cover_sets):
        for ts_idx in cover_set:
            block_cover_vecs[b_idx, ts_idx] = 1

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
            v, k, t, t_set_index, all_blocks, block_cover_sets,
            num_t_sets, t_set_to_blocks, block_cover_vecs, deadline,
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
    best_solution = _remove_redundant_blocks(best_solution, block_cover_sets, all_blocks, block_cover_vecs, deadline)

    return best_solution


def _lagrangian_greedy_cover(v, k, t, t_set_index, all_blocks, block_cover_sets,
                              num_t_sets, t_set_to_blocks, block_cover_vecs, deadline,
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
                # Vectorized score computation
                score = np.dot(block_cover_vecs[b_idx], w)
                if score > max_score:
                    max_score = score
                    best_block_idx = b_idx

            if best_block_idx != -1:
                candidate_blocks.add(best_block_idx)

        # Pick best block among candidates
        final_best_block_idx = -1
        final_max_score = -1.0

        for b_idx in candidate_blocks:
            score = np.dot(block_cover_vecs[b_idx], w)
            if score > final_max_score:
                final_max_score = score
                final_best_block_idx = b_idx

        if final_best_block_idx == -1:
            break

        selected_blocks.append(all_blocks[final_best_block_idx])

        # Update uncovered
        covered = block_cover_sets[final_best_block_idx]
        uncovered -= covered

        # Update Lagrangian multipliers
        if uncovered:
            uncovered_list = list(uncovered)
            w[uncovered_list] *= (1.0 + lambda_step)
        covered_list = list(covered)
        w[covered_list] *= (1.0 - lambda_step * 0.5)
        w = np.maximum(w, 0.1)

    return selected_blocks


def _remove_redundant_blocks(blocks, block_cover_sets, all_blocks, block_cover_vecs, deadline):
    """Remove blocks that don't cover any t-set uniquely."""
    if not blocks:
        return blocks

    num_t_sets = len(block_cover_vecs[0]) if len(block_cover_vecs) > 0 else 0
    if num_t_sets == 0:
        return blocks

    # Map blocks to their cover vectors
    all_blocks_set = {frozenset(b): i for i, b in enumerate(all_blocks)}
    current_block_indices = []
    for block in blocks:
        block_set = frozenset(block)
        if block_set in all_blocks_set:
            current_block_indices.append(all_blocks_set[block_set])

    if len(current_block_indices) != len(blocks):
        return blocks

    # Count coverage
    coverage = np.zeros(num_t_sets, dtype=np.int32)
    for b_idx in current_block_indices:
        coverage += block_cover_vecs[b_idx]

    # Iteratively remove redundant blocks
    improved = True
    while improved:
        if time.time() > deadline - 10.0:
            break
        improved = False
        for i in range(len(current_block_indices)):
            if time.time() > deadline - 10.0:
                break
            b_idx = current_block_indices[i]
            test_coverage = coverage - block_cover_vecs[b_idx]
            if np.all(test_coverage >= 1):
                coverage = test_coverage
                current_block_indices.pop(i)
                improved = True
                break

    if not current_block_indices:
        return blocks

    return [all_blocks[b_idx] for b_idx in current_block_indices]
