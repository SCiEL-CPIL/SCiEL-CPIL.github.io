"""Combined: vectorized whole-block greedy + sophisticated local search for C(v,k,t)."""

import time
import random
import itertools
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_index = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Precompute all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subsets)

    # C(k,t) - number of t-subsets in a k-block
    c_k_t = 1
    for i in range(t):
        c_k_t = c_k_t * (k - i) // (i + 1)

    # Build cover_matrix: for each k-subset, the indices of t-subsets it covers
    cover_matrix = np.zeros((n_k, c_k_t), dtype=np.int32)
    for i, block in enumerate(all_k_subsets):
        for j, triple in enumerate(itertools.combinations(block, t)):
            cover_matrix[i, j] = t_sub_index[triple]

    # Map from block tuple to its index in all_k_subsets
    block_to_k_idx = {block: i for i, block in enumerate(all_k_subsets)}

    best_blocks = None
    best_len = float('inf')

    def fast_greedy_cover(seed):
        """Vectorized whole-block greedy set cover."""
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        blocks = []

        while uncovered.any():
            # Vectorized gain: for each candidate block, count uncovered triples it covers
            gains = np.sum(uncovered[cover_matrix], axis=1)

            max_gain = np.max(gains)
            if max_gain == 0:
                break

            candidates = np.where(gains == max_gain)[0]
            chosen_idx = candidates[rng.randint(0, len(candidates) - 1)]

            blocks.append(all_k_subsets[chosen_idx])
            uncovered[cover_matrix[chosen_idx]] = False

        return blocks

    def local_search_reduce(blocks):
        """Reduce block count: remove redundant blocks, then try block replacements."""
        if not blocks:
            return blocks

        blocks = list(blocks)  # Make mutable copy

        def build_coverage(bks):
            """Build coverage count array for a list of blocks."""
            coverage = np.zeros(n_t, dtype=np.int32)
            for b in bks:
                idx = block_to_k_idx[b]
                coverage[cover_matrix[idx]] += 1
            return coverage

        # Phase 1: Remove redundant blocks (those with no unique triples)
        coverage = build_coverage(blocks)
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            to_remove = []

            for i, block in enumerate(blocks):
                idx = block_to_k_idx[block]
                # Check if this block has any triple covered only by it
                if not np.any(coverage[cover_matrix[idx]] == 1):
                    to_remove.append(i)

            if to_remove:
                remove_set = set(to_remove)
                blocks = [b for i, b in enumerate(blocks) if i not in remove_set]
                coverage = build_coverage(blocks)
                changed = True

        # Phase 2: Block replacement - try to replace blocks with alternatives
        # that cover the same critical triples but create more redundancy
        rng2 = random.Random(123)
        replacement_improved = True
        max_replace_rounds = 50

        for round_num in range(max_replace_rounds):
            if time.time() > deadline - 10:
                break

            if not replacement_improved:
                break
            replacement_improved = False

            coverage = build_coverage(blocks)

            # Shuffle for randomized search
            order = list(range(len(blocks)))
            rng2.shuffle(order)

            for i in order:
                if time.time() > deadline - 10:
                    break

                b_old = blocks[i]
                idx_old = block_to_k_idx[b_old]
                old_tris = cover_matrix[idx_old]

                # Find critical triples in this block (covered only by this block)
                critical_mask = (coverage[old_tris] == 1)
                if not np.any(critical_mask):
                    # Should have been removed in Phase 1, but handle anyway
                    coverage[old_tris] -= 1
                    blocks.pop(i)
                    replacement_improved = True
                    break

                # Determine must-points: union of all critical triples
                critical_triples = old_tris[critical_mask]
                must_points = set()
                for ti in critical_triples:
                    must_points.update(all_t_subsets[ti])

                if len(must_points) > k:
                    continue

                # Try to find a replacement block
                available_points = [p for p in range(v) if p not in must_points]
                needed = k - len(must_points)

                best_new_block = None
                best_score = -1
                attempts = 15

                for _ in range(attempts):
                    if time.time() > deadline - 10:
                        break

                    if needed > 0:
                        extra = rng2.sample(available_points, needed)
                    else:
                        extra = []

                    candidate = tuple(sorted(list(must_points) + extra))

                    if candidate in block_to_k_idx:
                        idx_new = block_to_k_idx[candidate]
                        new_tris = cover_matrix[idx_new]

                        # Check validity: all critical triples must be in candidate
                        valid = True
                        for ti in critical_triples:
                            tri = all_t_subsets[ti]
                            if not all(p in candidate for p in tri):
                                valid = False
                                break

                        if not valid:
                            continue

                        # Score: maximize number of triples in candidate that are
                        # already covered by other blocks (creates redundancy)
                        # A triple T in candidate is "redundant" if coverage[T] > 1
                        # (meaning it's covered by at least one block other than b_old)
                        score = np.sum(coverage[new_tris] > 1)

                        if score > best_score:
                            best_score = score
                            best_new_block = candidate

                if best_new_block is not None and best_new_block != b_old:
                    # Perform replacement
                    idx_new = block_to_k_idx[best_new_block]
                    new_tris = cover_matrix[idx_new]

                    # Remove old block's coverage
                    coverage[old_tris] -= 1
                    # Add new block's coverage
                    coverage[new_tris] += 1

                    blocks[i] = best_new_block
                    replacement_improved = True
                    break

        return blocks

    # Main search loop: multiple restarts with different seeds
    restart_count = 0
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000,
             7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000,
             17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000, 26000]

    for seed in seeds:
        if time.time() > deadline - 15:
            break

        # Build initial covering
        blocks = fast_greedy_cover(seed)

        if not blocks:
            continue

        # Apply local search to reduce
        refined = local_search_reduce(blocks)

        if len(refined) < best_len:
            best_len = len(refined)
            best_blocks = refined[:]

            # Early termination if we hit the known best
            if best_len <= 76:
                break

        restart_count += 1

    if best_blocks is not None:
        return [list(b) for b in best_blocks]

    # Fallback: return empty if something went wrong
    return []
