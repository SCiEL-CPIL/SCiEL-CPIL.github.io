import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin

    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}

    blocks = list(itertools.combinations(points, k))
    n_blocks = len(blocks)

    # C(k, t)
    c_k_t = 1
    for i in range(1, t + 1):
        c_k_t = c_k_t * (k - i + 1) // i

    # Precompute block-triple incidence
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    triple_freq = np.zeros(n_triples, dtype=np.int32)

    for i, block in enumerate(blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            t_idx = triple_to_idx[triple]
            block_triples[i, idx] = t_idx
            triple_freq[t_idx] += 1
            idx += 1

    # Weights: 1 / frequency
    weights = 1.0 / np.maximum(triple_freq, 1)

    best_solution = None
    best_size = float('inf')

    def greedy_cover(weights_vec, block_triples, n_triples, n_blocks, temperature=0.0):
        """Greedy set cover with optional randomization."""
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []
        used_blocks = np.zeros(n_blocks, dtype=bool)

        while np.any(uncovered):
            # Compute gain for each unused block
            weighted_uncovered = np.where(uncovered, weights_vec, 0.0)
            gains = np.sum(weighted_uncovered[block_triples], axis=1)
            gains[used_blocks] = -1  # Exclude already used

            if temperature > 0:
                max_gain = np.max(gains)
                if max_gain > 0:
                    # Softmax over positive gains
                    positive_mask = gains > 0
                    if not np.any(positive_mask):
                        break
                    exp_gains = np.where(positive_mask, np.exp((gains - max_gain) / temperature), 0.0)
                    total = np.sum(exp_gains)
                    if total > 0:
                        probs = exp_gains / total
                        best_idx = np.random.choice(n_blocks, p=probs)
                    else:
                        best_idx = np.argmax(gains)
                else:
                    best_idx = np.argmax(gains)
            else:
                best_idx = np.argmax(gains)

            if gains[best_idx] <= 0:
                break

            selected.append(best_idx)
            used_blocks[best_idx] = True
            uncovered[block_triples[best_idx]] = False

        return selected

    def polish(selected_indices, block_triples, n_triples):
        """Remove redundant blocks iteratively."""
        if len(selected_indices) <= 1:
            return selected_indices

        current = list(selected_indices)
        improved = True
        max_rounds = 50

        while improved and max_rounds > 0:
            improved = False
            max_rounds -= 1

            cover_count = np.zeros(n_triples, dtype=np.int32)
            for b_idx in current:
                cover_count[block_triples[b_idx]] += 1

            for i in range(len(current) - 1, -1, -1):
                b_idx = current[i]
                b_triples = block_triples[b_idx]
                if np.min(cover_count[b_triples]) > 1:
                    cover_count[b_triples] -= 1
                    current.pop(i)
                    improved = True

        return current

    def pair_swap_polish(selected_indices, block_triples, n_triples, n_blocks, max_iters=200):
        """Try to swap one block for another to reduce total size or maintain size with better coverage."""
        if len(selected_indices) <= 1:
            return selected_indices

        current = list(selected_indices)
        current_set = set(current)
        improved = True
        iters = 0

        while improved and iters < max_iters:
            improved = False
            iters += 1

            cover_count = np.zeros(n_triples, dtype=np.int32)
            for b_idx in current:
                cover_count[block_triples[b_idx]] += 1

            # Try removing each block and see if we can add a different block that covers the lost triples
            for i in range(len(current)):
                b_idx = current[i]
                b_triples = block_triples[b_idx]
                # Triples that would become uncovered
                lost = b_triples[cover_count[b_triples] == 1]
                if len(lost) == 0:
                    # Can just remove
                    cover_count[b_triples] -= 1
                    current.pop(i)
                    current_set.discard(b_idx)
                    improved = True
                    break

                # Try to find a single block that covers all lost triples
                # and is not already in the set
                # This is expensive, so limit the search
                if iters > 50:
                    continue

                # For each candidate block not in current_set, check if it covers all lost
                # Use a quick filter: the candidate must cover at least one lost triple
                # and the union of its triples with the remaining cover must cover all lost
                # Simplified: just check if the candidate covers all lost triples
                lost_set = set(lost.tolist())
                for cand in range(n_blocks):
                    if cand in current_set:
                        continue
                    cand_triples = block_triples[cand]
                    # Check if cand covers all lost
                    if len(lost_set) <= len(cand_triples):
                        # Quick check using numpy
                        cand_set = set(cand_triples.tolist())
                        if lost_set.issubset(cand_set):
                            # Swap: remove b_idx, add cand
                            cover_count[b_triples] -= 1
                            cover_count[cand_triples] += 1
                            current.pop(i)
                            current_set.discard(b_idx)
                            current.append(cand)
                            current_set.add(cand)
                            improved = True
                            break
                if improved:
                    break

        return current

    # Run multiple greedy passes
    n_restarts = 0
    while time.time() < deadline - 3.0:
        n_restarts += 1

        # Vary temperature and weight perturbation
        temp = 0.0 if n_restarts <= 3 else random.uniform(0.1, 0.5)

        if n_restarts == 1:
            perturbed_weights = weights.copy()
        else:
            noise = np.random.normal(0, 0.2, size=n_triples)
            perturbed_weights = weights * (1 + noise)
            perturbed_weights = np.maximum(perturbed_weights, 0.01)

        selected = greedy_cover(perturbed_weights, block_triples, n_triples, n_blocks, temperature=temp)
        selected = polish(selected, block_triples, n_triples)
        selected = pair_swap_polish(selected, block_triples, n_triples, n_blocks)

        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = [list(blocks[i]) for i in selected]

        if best_size <= 69:
            break

        # If we've done enough restarts, try more aggressive polishing
        if n_restarts % 10 == 0 and best_solution is not None:
            # Try to improve the best solution further
            best_indices = [blocks.index(tuple(b)) for b in best_solution]
            improved = pair_swap_polish(best_indices, block_triples, n_triples, n_blocks, max_iters=500)
            if len(improved) < best_size:
                best_size = len(improved)
                best_solution = [list(blocks[i]) for i in improved]

    if best_solution is None:
        selected = greedy_cover(weights, block_triples, n_triples, n_blocks)
        selected = polish(selected, block_triples, n_triples)
        best_solution = [list(blocks[i]) for i in selected]

    return best_solution
