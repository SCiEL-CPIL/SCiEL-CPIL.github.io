import itertools
import numpy as np
import time
import random
from scipy import sparse

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    # Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, compute which t-subsets it covers
    k_choose_t = len(list(itertools.combinations(range(k), t)))
    block_t_indices = np.empty((n_blocks, k_choose_t), dtype=np.int32)

    for i, block in enumerate(all_blocks):
        for j, subset in enumerate(itertools.combinations(block, t)):
            block_t_indices[i, j] = t_sub_to_idx[subset]

    # Build sparse coverage matrix: rows = blocks, cols = t-subsets
    # M[i, j] = 1 if block i covers t-subset j
    row_indices = np.repeat(np.arange(n_blocks), k_choose_t)
    col_indices = block_t_indices.flatten()
    data = np.ones(n_blocks * k_choose_t, dtype=np.int8)
    cover_matrix = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n_blocks, n_t))

    def greedy_cover(seed=0):
        """Greedy set cover using sparse matrix for gain calculation."""
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        selected = []
        used = np.zeros(n_blocks, dtype=bool)

        while uncovered.any():
            if time.time() > deadline - 5.0:
                break

            # Calculate gains: number of uncovered subsets each block covers
            # Use sparse matrix-vector product
            gains = (cover_matrix @ uncovered.astype(np.int8)).astype(np.int32)

            # Mask out already-used blocks
            gains[used] = -1

            max_gain = gains.max()
            if max_gain <= 0:
                break

            # Find all blocks with max gain and pick one (with randomization for diversity)
            candidates = np.where(gains == max_gain)[0]
            if seed == 0:
                chosen = candidates[0]
            else:
                chosen = candidates[rng.randint(0, len(candidates) - 1)]

            selected.append(chosen)
            used[chosen] = True
            uncovered[block_t_indices[chosen]] = False

        return selected

    def remove_redundant(selected):
        """Remove redundant blocks iteratively."""
        if not selected:
            return selected

        cover_count = np.zeros(n_t, dtype=np.int32)
        for b in selected:
            cover_count[block_t_indices[b]] += 1

        changed = True
        while changed and time.time() < deadline - 3.0:
            changed = False
            new_selected = []
            for b in selected:
                sub_idx = block_t_indices[b]
                if np.min(cover_count[sub_idx]) > 1:
                    # Redundant
                    cover_count[sub_idx] -= 1
                    changed = True
                else:
                    new_selected.append(b)
            selected = new_selected

        return selected

    def local_search(selected, max_rounds=5):
        """Try to replace blocks with better alternatives."""
        if not selected:
            return selected

        selected = list(selected)
        improved = True
        rnd = 0

        while improved and rnd < max_rounds and time.time() < deadline - 2.0:
            improved = False
            rnd += 1

            # Build cover count
            cover_count = np.zeros(n_t, dtype=np.int32)
            for b in selected:
                cover_count[block_t_indices[b]] += 1

            # Build reverse index: for each t-subset, which selected blocks cover it
            # We'll use this to find candidate replacements

            for i in range(len(selected)):
                if time.time() > deadline - 2.0:
                    break

                b = selected[i]
                sub_idx = block_t_indices[b]

                # Find essential t-subsets (covered only by this block)
                essential = sub_idx[cover_count[sub_idx] == 1]
                if len(essential) == 0:
                    # Not essential, can potentially remove or replace
                    # Try removing first
                    cover_count[sub_idx] -= 1
                    if np.all(cover_count >= 0):
                        # Check if removal is valid (all still covered)
                        # Since we only removed one block, and no essential subsets, it's fine
                        selected.pop(i)
                        improved = True
                        break
                    else:
                        cover_count[sub_idx] += 1
                        # Try replacement
                        # Find candidates that cover all essential subsets
                        candidate_set = set()
                        for t_idx in essential:
                            # Find all blocks that cover this t-subset
                            # We need to search all blocks, which is expensive
                            # Instead, sample a few
                            pass
                        # For now, skip complex replacement
                        continue

                # If there are essential subsets, try to find a better block
                # that covers all essential subsets
                best_replace = -1
                best_gain = -1

                # Sample candidate blocks from those covering essential subsets
                # This is expensive, so limit it
                if len(essential) > 0:
                    # Get candidates from first essential subset
                    first_essential = essential[0]
                    # Find all blocks covering this t-subset
                    # We can precompute this, but for now, sample
                    n_samples = min(50, n_blocks)
                    for _ in range(n_samples):
                        cand = random.randint(0, n_blocks - 1)
                        cand_sub = block_t_indices[cand]
                        # Check if cand covers all essential subsets
                        cand_set = set(cand_sub)
                        if all(e in cand_set for e in essential):
                            # Score: number of additional subsets covered
                            # that are currently covered by other selected blocks
                            # Simple score: total coverage
                            score = len(cand_set)
                            if score > best_gain:
                                best_gain = score
                                best_replace = cand

                if best_replace != -1 and best_replace not in selected:
                    # Replace
                    old_sub = block_t_indices[selected[i]]
                    new_sub = block_t_indices[best_replace]
                    cover_count[old_sub] -= 1
                    cover_count[new_sub] += 1
                    selected[i] = best_replace
                    improved = True
                    break

        return selected

    best_blocks = None
    best_len = float('inf')

    # 1. Deterministic greedy
    selected = greedy_cover(0)
    selected = remove_redundant(selected)
    if selected:
        best_blocks = selected
        best_len = len(selected)

    # 2. Randomized greedy restarts
    n_restarts = 30
    for seed in range(1, n_restarts + 1):
        if time.time() > deadline - 10.0:
            break

        selected = greedy_cover(seed)
        selected = remove_redundant(selected)
        if selected and len(selected) < best_len:
            best_blocks = selected
            best_len = len(selected)

    # 3. Local search on best solution
    if best_blocks and time.time() < deadline - 5.0:
        best_blocks = local_search(best_blocks, max_rounds=3)
        best_blocks = remove_redundant(best_blocks)

    # 4. Final redundancy removal
    best_blocks = remove_redundant(best_blocks)

    # Convert indices back to lists of points
    result_blocks = [list(all_blocks[i]) for i in best_blocks]

    return result_blocks
