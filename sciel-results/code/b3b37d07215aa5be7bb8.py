import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 115.0

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}

    # Precompute all k-subsets and their t-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k_subsets = len(all_k_subsets)

    # Build mapping
    block_to_triples = [None] * n_k_subsets
    triple_to_blocks = [[] for _ in range(n_t_subsets)]

    for i, block in enumerate(all_k_subsets):
        triples = []
        for triple in itertools.combinations(block, t):
            idx = t_subset_to_idx[triple]
            triples.append(idx)
            triple_to_blocks[idx].append(i)
        block_to_triples[i] = triples

    # Convert to sets for O(1) membership tests
    block_to_triples_set = [set(triples) for triples in block_to_triples]

    def greedy_cover(seed, strategy="random"):
        rng = random.Random(seed)
        uncovered = bytearray([1] * n_t_subsets)  # 1 = uncovered
        uncovered_count = n_t_subsets
        blocks = []

        # For efficiency, maintain a list of uncovered triple indices
        uncovered_list = list(range(n_t_subsets))

        while uncovered_list:
            if time.time() > deadline - 10:
                return None

            if strategy == "random":
                # Pick a random uncovered triple
                idx = rng.choice(uncovered_list)
            elif strategy == "least_covered":
                # Pick the uncovered triple with fewest covering blocks
                min_count = float('inf')
                idx = None
                # Sample some for speed
                sample = rng.sample(uncovered_list, min(len(uncovered_list), 100))
                for s in sample:
                    c = len(triple_to_blocks[s])
                    if c < min_count:
                        min_count = c
                        idx = s
                if idx is None:
                    idx = uncovered_list[0]
            else:
                idx = uncovered_list[0]

            # Find the block covering idx that maximizes new coverage
            best_block_id = -1
            best_gain = -1

            candidates = triple_to_blocks[idx]
            # Shuffle for tie-breaking
            if strategy == "random":
                candidates = list(candidates)
                rng.shuffle(candidates)
            else:
                candidates = list(candidates)

            for b_id in candidates:
                gain = 0
                for t_idx in block_to_triples[b_id]:
                    if uncovered[t_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_id = b_id

            if best_block_id == -1:
                return None

            blocks.append(best_block_id)

            # Update uncovered
            removed = []
            for t_idx in block_to_triples[best_block_id]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = 0
                    removed.append(t_idx)
            for t_idx in removed:
                try:
                    uncovered_list.remove(t_idx)
                except ValueError:
                    pass
            uncovered_count -= len(removed)

        return blocks

    def reduce_blocks(blocks, seed):
        if not blocks:
            return blocks

        rng = random.Random(seed)
        current_blocks = list(blocks)

        # Compute coverage counts
        cov_counts = np.zeros(n_t_subsets, dtype=np.int32)
        for b_id in current_blocks:
            for t_idx in block_to_triples[b_id]:
                cov_counts[t_idx] += 1

        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(len(current_blocks) - 1, -1, -1):
                b_id = current_blocks[i]
                can_remove = True
                for t_idx in block_to_triples[b_id]:
                    if cov_counts[t_idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for t_idx in block_to_triples[b_id]:
                        cov_counts[t_idx] -= 1
                    current_blocks.pop(i)
                    improved = True
                    break

        return [list(all_k_subsets[b_id]) for b_id in current_blocks]

    def swap_optimize(blocks, seed, max_iter=500):
        if not blocks:
            return blocks

        rng = random.Random(seed)
        current_blocks = list(blocks)

        cov_counts = np.zeros(n_t_subsets, dtype=np.int32)
        for b_id in current_blocks:
            for t_idx in block_to_triples[b_id]:
                cov_counts[t_idx] += 1

        iter_count = 0
        while iter_count < max_iter and time.time() < deadline - 5:
            iter_count += 1

            if len(current_blocks) <= 1:
                break

            # Pick a random block
            b_idx = rng.randint(0, len(current_blocks) - 1)
            old_b_id = current_blocks[b_idx]

            # Find critical triples (count == 1) in this block
            critical = []
            for t_idx in block_to_triples[old_b_id]:
                if cov_counts[t_idx] == 1:
                    critical.append(t_idx)

            if not critical:
                # Can remove
                for t_idx in block_to_triples[old_b_id]:
                    cov_counts[t_idx] -= 1
                current_blocks.pop(b_idx)
                continue

            # Try to find a replacement
            target = critical[0]
            candidates = triple_to_blocks[target]

            best_new_b_id = -1
            best_score = -1

            for cand_id in candidates:
                if cand_id == old_b_id:
                    continue
                # Check if covers all critical triples
                valid = True
                for t_idx in critical:
                    if t_idx not in block_to_triples_set[cand_id]:
                        valid = False
                        break
                if not valid:
                    continue

                # Score: how many low-coverage triples does it cover?
                score = 0
                for t_idx in block_to_triples[cand_id]:
                    if cov_counts[t_idx] == 0:
                        score += 100  # new coverage (shouldn't happen)
                    elif cov_counts[t_idx] == 1:
                        score += 50
                    elif cov_counts[t_idx] == 2:
                        score += 20
                    elif cov_counts[t_idx] <= 5:
                        score += 5
                    else:
                        score -= 2

                if score > best_score:
                    best_score = score
                    best_new_b_id = cand_id

            if best_new_b_id != -1:
                # Swap
                for t_idx in block_to_triples[old_b_id]:
                    cov_counts[t_idx] -= 1
                for t_idx in block_to_triples[best_new_b_id]:
                    cov_counts[t_idx] += 1
                current_blocks[b_idx] = best_new_b_id

                # Try to remove blocks
                for i in range(len(current_blocks) - 1, -1, -1):
                    b_id = current_blocks[i]
                    can_remove = True
                    for t_idx in block_to_triples[b_id]:
                        if cov_counts[t_idx] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for t_idx in block_to_triples[b_id]:
                            cov_counts[t_idx] -= 1
                        current_blocks.pop(i)

        return [list(all_k_subsets[b_id]) for b_id in current_blocks]

    def two_opt(blocks, seed, max_iter=200):
        """Try swapping two blocks for one better block (net -1)."""
        if len(blocks) < 2:
            return blocks

        rng = random.Random(seed)
        current_blocks = list(blocks)

        cov_counts = np.zeros(n_t_subsets, dtype=np.int32)
        for b_id in current_blocks:
            for t_idx in block_to_triples[b_id]:
                cov_counts[t_idx] += 1

        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            if len(current_blocks) < 2:
                break

            # Pick two random blocks
            i, j = rng.sample(range(len(current_blocks)), 2)
            b1 = current_blocks[i]
            b2 = current_blocks[j]

            # Find triples covered by exactly one of b1, b2 (critical for the other)
            # Simpler: find all triples where count would drop to 0 if we remove both
            # Actually, we want to remove b1 and b2, and add b3 that covers all
            # triples that would become uncovered.

            # Triples that would become uncovered if we remove b1 and b2:
            would_be_uncovered = set()
            for t_idx in block_to_triples[b1]:
                if cov_counts[t_idx] <= 1:
                    would_be_uncovered.add(t_idx)
            for t_idx in block_to_triples[b2]:
                if cov_counts[t_idx] <= 1:
                    would_be_uncovered.add(t_idx)

            if not would_be_uncovered:
                # Can remove both
                for t_idx in block_to_triples[b1]:
                    cov_counts[t_idx] -= 1
                for t_idx in block_to_triples[b2]:
                    cov_counts[t_idx] -= 1
                current_blocks.pop(j)
                current_blocks.pop(i)
                continue

            if len(would_be_uncovered) > 5:
                # Too many critical triples, unlikely to find a replacement
                continue

            # Find a block that covers all would_be_uncovered
            # Use the first one to filter
            first_critical = next(iter(would_be_uncovered))
            candidates = triple_to_blocks[first_critical]

            best_b3 = -1
            best_score = -1

            for cand in candidates:
                if cand == b1 or cand == b2:
                    continue
                valid = True
                for t_idx in would_be_uncovered:
                    if t_idx not in block_to_triples_set[cand]:
                        valid = False
                        break
                if not valid:
                    continue

                # Score: how many additional low-coverage triples does it cover?
                score = 0
                for t_idx in block_to_triples[cand]:
                    if t_idx in would_be_uncovered:
                        continue
                    if cov_counts[t_idx] == 0:
                        score += 50
                    elif cov_counts[t_idx] == 1:
                        score += 20
                    elif cov_counts[t_idx] <= 3:
                        score += 5
                    else:
                        score -= 1

                if score > best_score:
                    best_score = score
                    best_b3 = cand

            if best_b3 != -1:
                # Remove b1, b2, add best_b3
                for t_idx in block_to_triples[b1]:
                    cov_counts[t_idx] -= 1
                for t_idx in block_to_triples[b2]:
                    cov_counts[t_idx] -= 1
                for t_idx in block_to_triples[best_b3]:
                    cov_counts[t_idx] += 1
                current_blocks[j] = best_b3
                current_blocks.pop(i)

        return [list(all_k_subsets[b_id]) for b_id in current_blocks]

    best_blocks = None
    best_count = float('inf')

    strategies = ["random", "random", "least_covered", "random"]

    seed = 0
    while time.time() < deadline - 20:
        strategy = strategies[seed % len(strategies)]
        greedy_result = greedy_cover(seed, strategy)
        if greedy_result is not None:
            # Reduce
            reduced = reduce_blocks(greedy_result, seed)
            # Swap optimize
            swapped = swap_optimize(reduced, seed)
            # Two-opt
            two_optimized = two_opt(swapped, seed)

            if len(two_optimized) < best_count:
                best_count = len(two_optimized)
                best_blocks = two_optimized

        seed += 1

    if best_blocks is None:
        return []

    return best_blocks
