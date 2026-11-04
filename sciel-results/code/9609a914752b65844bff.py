import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)

    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)

    # For each k-block, store the t-subset indices it covers
    block_t_indices = []
    for block in all_k_blocks:
        t_subs_in_block = list(itertools.combinations(block, t))
        indices = np.array([t_subset_to_idx[s] for s in t_subs_in_block], dtype=np.int32)
        block_t_indices.append(indices)

    # Reverse mapping: for each t-subset, which k-blocks cover it
    t_subset_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx in range(num_k_blocks):
        for idx in block_t_indices[b_idx]:
            t_subset_to_blocks[idx].append(b_idx)

    best_blocks = None
    best_score = float('inf')

    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline - 15.0:
            break

        random.seed(restart)
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        block_indices = []

        # Greedy phase: pick blocks to cover all t-subsets
        # Track uncovered count for early termination check
        uncovered_count = num_t_subsets
        uncovered_set = set(range(num_t_subsets))

        while uncovered_count > 0:
            if time.time() > deadline - 10.0:
                # Can't finish in time; break and hope we have a valid covering
                # Actually we need to complete, so let's just continue but skip restarts
                pass

            # Pick a random uncovered t-subset
            # Use random sampling from the uncovered set
            # For efficiency when uncovered is small, iterate; when large, sample
            if len(uncovered_set) < 1000:
                u_idx = random.choice(list(uncovered_set))
            else:
                # Sample until we find an uncovered one
                while True:
                    candidate = random.randint(0, num_t_subsets - 1)
                    if coverage[candidate] == 0:
                        u_idx = candidate
                        break

            # Get candidate blocks that cover u_idx
            candidates = t_subset_to_blocks[u_idx]

            # Find the block with maximum gain (most newly covered t-subsets)
            best_gain = -1
            best_block_idx = candidates[0]

            for b_idx in candidates:
                idxs = block_t_indices[b_idx]
                gain = np.count_nonzero(coverage[idxs] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                    if gain == len(block_t_indices[b_idx]):
                        # Perfect block, can't do better
                        break

            # Add the best block
            block_indices.append(best_block_idx)

            # Update coverage
            idxs = block_t_indices[best_block_idx]
            newly_covered = idxs[coverage[idxs] == 0]
            coverage[idxs] += 1
            uncovered_count -= len(newly_covered)

            # Remove newly covered from uncovered_set
            for idx in newly_covered:
                uncovered_set.discard(idx)

        # End greedy. We have a complete covering.

        # Reduction phase: remove redundant blocks
        # A block is redundant if no t-subset is covered only by it.
        # Iterate until no more removals.
        changed = True
        while changed:
            if time.time() > deadline - 8.0:
                break
            changed = False
            # Check in reverse order (blocks added later may be more redundant)
            for i in range(len(block_indices) - 1, -1, -1):
                if time.time() > deadline - 8.0:
                    break
                b_idx = block_indices[i]
                idxs = block_t_indices[b_idx]
                # Check if any t-subset has coverage exactly 1 (only this block covers it)
                if np.any(coverage[idxs] == 1):
                    continue  # Essential, keep
                # This block is redundant; remove it
                coverage[idxs] -= 1
                block_indices.pop(i)
                changed = True

        if len(block_indices) < best_score:
            best_score = len(block_indices)
            best_blocks = [list(all_k_blocks[bi]) for bi in block_indices]

    return best_blocks
