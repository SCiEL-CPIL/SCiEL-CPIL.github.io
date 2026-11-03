import itertools
import numpy as np
import scipy.sparse as sp
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin (2s before 150s kill)

    # Generate all t-subsets and map to indices
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t = len(t_subs)

    # Generate all k-subsets (candidate blocks)
    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)

    # Build sparse matrix A of size n_k x n_t
    # A[i, j] = 1 if t_subs[j] is subset of k_subs[i]
    row_indices = []
    col_indices = []
    data = []

    # For each block, find which t-subsets it contains
    for i, block in enumerate(k_subs):
        for t_sub in itertools.combinations(block, t):
            j = t_sub_to_idx.get(t_sub)
            if j is not None:
                row_indices.append(i)
                col_indices.append(j)
                data.append(1)

    A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_k, n_t), dtype=np.int8)

    # Pre-compute the column indices for each block (for fast access)
    block_cols = []
    for i in range(n_k):
        row_start = A.indptr[i]
        row_end = A.indptr[i + 1]
        cols = A.indices[row_start:row_end].copy()
        block_cols.append(cols)

    def greedy_cover(seed_blocks=None, max_iter=None, rng=None):
        """Greedy set cover with optional seed blocks."""
        if rng is None:
            rng = random
        u = np.ones(n_t, dtype=np.int8)
        selected_blocks = []

        # If we have seed blocks, start with them
        if seed_blocks:
            for idx in seed_blocks:
                cols = block_cols[idx]
                u[cols] = 0
                selected_blocks.append(idx)

        if max_iter is None:
            max_iter = 10000

        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if u.sum() == 0:
                break

            # Calculate gain for each block
            gains = A @ u
            # Find best block
            best_idx = np.argmax(gains)
            best_gain = gains[best_idx]

            if best_gain == 0:
                break

            selected_blocks.append(best_idx)

            # Update uncovered vector
            cols = block_cols[best_idx]
            u[cols] = 0

        return selected_blocks

    def remove_redundant(selected_blocks):
        """Remove redundant blocks from the covering efficiently."""
        if not selected_blocks:
            return selected_blocks

        n_sel = len(selected_blocks)
        # Build cover_count: how many selected blocks cover each t-subset
        cover_count = np.zeros(n_t, dtype=np.int32)
        for idx in selected_blocks:
            cols = block_cols[idx]
            cover_count[cols] += 1

        # Track which blocks are still active
        active = [True] * n_sel

        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False

            # Find all redundant blocks in one pass
            redundant = []
            for i in range(n_sel):
                if not active[i]:
                    continue
                cols = block_cols[selected_blocks[i]]
                # Check if any col has cover_count == 1 (essential)
                is_essential = False
                for c in cols:
                    if cover_count[c] == 1:
                        is_essential = True
                        break
                if not is_essential:
                    redundant.append(i)

            if redundant:
                changed = True
                # Remove all redundant blocks
                for i in redundant:
                    cols = block_cols[selected_blocks[i]]
                    cover_count[cols] -= 1
                    active[i] = False

        # Rebuild the list
        return [selected_blocks[i] for i in range(n_sel) if active[i]]

    def local_improve(selected_blocks, max_rounds=5):
        """Improve by trying to remove blocks and adding replacements."""
        if not selected_blocks:
            return selected_blocks

        best = list(selected_blocks)
        best_size = len(best)
        rng = random

        for round_num in range(max_rounds):
            if time.time() > deadline - 2:
                break
            if len(best) == 0:
                break

            improved = False
            n_best = len(best)

            # Try removing each block and finding a replacement
            for i in range(n_best):
                if time.time() > deadline - 2:
                    break

                # Temporarily remove block i
                removed_idx = best[i]
                removed_cols = block_cols[removed_idx]

                # Compute cover count without block i
                temp_cover = np.zeros(n_t, dtype=np.int8)
                for j, idx in enumerate(best):
                    if j == i:
                        continue
                    temp_cover[block_cols[idx]] += 1

                # Find t-subsets that would become uncovered
                uncovered = np.array([c for c in removed_cols if temp_cover[c] == 0], dtype=np.int32)

                if len(uncovered) == 0:
                    # Block is redundant, just remove it
                    best.pop(i)
                    improved = True
                    break

                # Find candidate blocks that cover at least one uncovered t-subset
                # Score candidates by how many uncovered they cover
                best_candidate = -1
                best_score = 0

                # Sample a subset of candidates for speed
                sample_size = min(2000, n_k)
                candidates = rng.sample(range(n_k), sample_size) if n_k > sample_size else range(n_k)

                for cand in candidates:
                    if cand == removed_idx:
                        continue
                    cand_cols = block_cols[cand]
                    # Count how many of the uncovered t-subsets this candidate covers
                    score = 0
                    for c in uncovered:
                        if c in cand_cols:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_candidate = cand

                if best_candidate >= 0 and best_score >= len(uncovered):
                    # Check that adding this candidate doesn't leave anything uncovered
                    # (it should, since best_score == len(uncovered))
                    best[i] = best_candidate
                    improved = True
                    break

            if not improved:
                break

        return best

    def local_improve2(selected_blocks, max_rounds=3):
        """Stronger local search: try swapping pairs or add+remove."""
        if not selected_blocks:
            return selected_blocks

        best = list(selected_blocks)
        best_size = len(best)
        rng = random

        for round_num in range(max_rounds):
            if time.time() > deadline - 2:
                break
            if len(best) == 0:
                break

            improved = False
            n_best = len(best)

            # Try 1-for-2 swaps: remove 2 blocks, add 1
            if n_best >= 2:
                # Sample pairs to try
                max_pairs = min(100, n_best * (n_best - 1) // 2)
                pairs = [(i, j) for i in range(n_best) for j in range(i + 1, n_best)]
                if len(pairs) > max_pairs:
                    pairs = rng.sample(pairs, max_pairs)

                for i, j in pairs:
                    if time.time() > deadline - 2:
                        break

                    # Remove blocks i and j
                    idx_i = best[i]
                    idx_j = best[j]

                    # Compute cover count without i and j
                    temp_cover = np.zeros(n_t, dtype=np.int8)
                    for m, idx in enumerate(best):
                        if m == i or m == j:
                            continue
                        temp_cover[block_cols[idx]] += 1

                    # Find uncovered t-subsets
                    cols_i = block_cols[idx_i]
                    cols_j = block_cols[idx_j]
                    uncovered_set = set()
                    for c in cols_i:
                        if temp_cover[c] == 0:
                            uncovered_set.add(c)
                    for c in cols_j:
                        if temp_cover[c] == 0:
                            uncovered_set.add(c)

                    if len(uncovered_set) == 0:
                        # Both are redundant
                        # Remove the one with higher index first
                        if j > i:
                            best.pop(j)
                            best.pop(i)
                        else:
                            best.pop(i)
                            best.pop(j)
                        improved = True
                        break

                    # Find a candidate that covers all uncovered
                    sample_size = min(1000, n_k)
                    candidates = rng.sample(range(n_k), sample_size) if n_k > sample_size else range(n_k)

                    best_candidate = -1
                    for cand in candidates:
                        if cand == idx_i or cand == idx_j:
                            continue
                        cand_cols = set(block_cols[cand])
                        if uncovered_set.issubset(cand_cols):
                            best_candidate = cand
                            break

                    if best_candidate >= 0:
                        # Replace both with one
                        new_best = []
                        for m, idx in enumerate(best):
                            if m != i and m != j:
                                new_best.append(idx)
                        new_best.append(best_candidate)
                        if len(new_best) < best_size:
                            best = new_best
                            best_size = len(best)
                            improved = True
                            break

            if not improved:
                break

        return best

    # Run greedy with different strategies
    best_result = None
    best_size = float('inf')

    # First, run standard greedy
    result = greedy_cover()
    result = remove_redundant(result)
    if len(result) < best_size:
        best_result = result
        best_size = len(result)

    # Try randomized restarts with different seeds
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break

        # Create diverse random seeds
        seed_size = min(10, n_k)
        seed_blocks = random.sample(range(n_k), seed_size)

        result = greedy_cover(seed_blocks=seed_blocks)
        result = remove_redundant(result)

        if len(result) < best_size:
            best_result = result
            best_size = len(result)

    # Try local improvement on the best result
    if best_result and time.time() < deadline - 15:
        improved = local_improve(best_result, max_rounds=5)
        improved = remove_redundant(improved)
        if len(improved) < best_size:
            best_result = improved
            best_size = len(improved)

    # Try stronger local search
    if best_result and time.time() < deadline - 10:
        improved = local_improve2(best_result, max_rounds=3)
        improved = remove_redundant(improved)
        if len(improved) < best_size:
            best_result = improved
            best_size = len(improved)

    # Final redundancy removal
    if best_result:
        best_result = remove_redundant(best_result)

    # Convert indices back to blocks
    if best_result is None:
        best_result = []

    final_blocks = [list(k_subs[i]) for i in best_result]

    return final_blocks
