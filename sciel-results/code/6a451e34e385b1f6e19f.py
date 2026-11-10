import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    n_t = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}

    # Generate all k-blocks
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)

    # Build incidence: for each block, which t-subsets it covers
    block_subsets = [[] for _ in range(n_blocks)]
    for i, block in enumerate(all_blocks):
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            block_subsets[i].append(idx)

    # Also build subset_to_blocks for quick lookup
    subset_to_blocks = [[] for _ in range(n_t)]
    for i in range(n_blocks):
        for s_idx in block_subsets[i]:
            subset_to_blocks[s_idx].append(i)

    # Precompute block_subsets as numpy arrays for vectorized operations
    block_subsets_arr = [np.array(bs, dtype=np.int32) for bs in block_subsets]
    
    # Precompute block_subsets as sets for O(1) membership
    block_subsets_set = [set(bs) for bs in block_subsets]
    
    # Precompute for each block, the list of blocks that share at least one t-subset
    # This helps in finding good swap candidates
    block_neighbors = [set() for _ in range(n_blocks)]
    for i in range(n_blocks):
        for s_idx in block_subsets[i]:
            for j in subset_to_blocks[s_idx]:
                if j != i:
                    block_neighbors[i].add(j)

    # === Phase 1: Greedy construction ===
    in_block = np.zeros(n_blocks, dtype=bool)
    cover_count = np.zeros(n_t, dtype=np.int32)
    uncovered = n_t
    solution = []

    while uncovered > 0:
        # Find block with max new coverage using vectorized approach
        best_block = -1
        max_gain = 0

        # Vectorized gain computation for all blocks at once
        # For each block, count uncovered t-subsets it covers
        gains = np.zeros(n_blocks, dtype=np.int32)
        for i in range(n_blocks):
            if in_block[i]:
                continue
            bs = block_subsets_arr[i]
            gains[i] = np.sum(cover_count[bs] == 0)
        
        # Find the block with maximum gain
        valid_mask = ~in_block
        if np.any(valid_mask):
            max_gain = np.max(gains[valid_mask])
            if max_gain > 0:
                best_block = int(np.argmax(gains * valid_mask))

        if best_block == -1 or max_gain == 0:
            break

        in_block[best_block] = True
        solution.append(best_block)
        bs = block_subsets_arr[best_block]
        new_covered = cover_count[bs] == 0
        uncovered -= int(np.sum(new_covered))
        cover_count[bs] += 1

    # === Phase 2: Remove redundant blocks ===
    def remove_redundants():
        nonlocal in_block, cover_count, solution
        changed = True
        while changed:
            changed = False
            for idx, b in enumerate(list(solution)):
                if not in_block[b]:
                    continue
                bs = block_subsets_arr[b]
                if np.any(cover_count[bs] == 1):
                    continue
                # Block b is redundant
                in_block[b] = False
                solution.pop(idx)
                cover_count[bs] -= 1
                changed = True
                break

    remove_redundants()
    best_count = len(solution)
    best_solution = list(solution)

    # === Phase 3: Local search with swap moves ===
    def try_swap(b_out):
        """Try to find a block that covers all critical t-subsets of b_out."""
        bs_out = block_subsets_set[b_out]
        criticals = [s for s in bs_out if cover_count[s] == 1]
        if not criticals:
            return None  # b_out is redundant, handled separately

        # Find candidate blocks that cover all criticals
        # Use intersection of block lists for each critical
        candidate_sets = [set(subset_to_blocks[c]) for c in criticals]
        candidates = set.intersection(*candidate_sets) if candidate_sets else set()
        
        # Filter to only blocks not already in solution
        for b_in in candidates:
            if not in_block[b_in]:
                return b_in
        return None

    # Local search loop
    while time.time() < deadline:
        if not solution:
            break

        # Occasionally try to remove a redundant block
        if random.random() < 0.05:
            remove_redundants()
            if len(solution) < best_count:
                best_count = len(solution)
                best_solution = list(solution)
            continue

        # Pick a random block to try to replace
        b_out = random.choice(solution)

        # Find criticals
        bs_out = block_subsets_set[b_out]
        criticals = [s for s in bs_out if cover_count[s] == 1]

        if not criticals:
            # Remove redundant block
            in_block[b_out] = False
            solution.remove(b_out)
            bs_arr = block_subsets_arr[b_out]
            cover_count[bs_arr] -= 1
            remove_redundants()
            if len(solution) < best_count:
                best_count = len(solution)
                best_solution = list(solution)
            continue

        # Try to find a replacement using precomputed neighbor info
        b_in = try_swap(b_out)

        if b_in is not None:
            # Perform swap
            in_block[b_out] = False
            in_block[b_in] = True
            solution.remove(b_out)
            solution.append(b_in)

            # Update cover counts
            bs_out_arr = block_subsets_arr[b_out]
            bs_in_arr = block_subsets_arr[b_in]
            cover_count[bs_out_arr] -= 1
            cover_count[bs_in_arr] += 1

            # Remove any newly redundant blocks
            remove_redundants()

            if len(solution) < best_count:
                best_count = len(solution)
                best_solution = list(solution)

    # Return the best solution found
    return [list(all_blocks[i]) for i in best_solution]
