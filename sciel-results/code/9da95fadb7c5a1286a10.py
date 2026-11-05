import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin before 150s kill

    # Precompute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}

    # Precompute all k-blocks and their t-subset coverage
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, store the list of subset indices it covers
    block_to_subsets = [[] for _ in range(n_blocks)]
    # For each subset, store which blocks cover it
    subset_to_blocks = [[] for _ in range(n_subsets)]

    for b_idx, block in enumerate(all_blocks):
        for subset in itertools.combinations(block, t):
            u_idx = subset_to_idx[subset]
            block_to_subsets[b_idx].append(u_idx)
            subset_to_blocks[u_idx].append(b_idx)

    # Convert to numpy arrays for fast operations
    block_subsets_arr = [np.array(bs, dtype=np.int32) for bs in block_to_subsets]
    subset_blocks_arr = [np.array(sb, dtype=np.int32) for sb in subset_to_blocks]

    def greedy_cover(seed, time_limit):
        """Run greedy covering with optional randomization."""
        rng = random.Random(seed)
        uncovered = np.ones(n_subsets, dtype=bool)
        gains = np.zeros(n_blocks, dtype=np.int32)
        covered_counts = np.zeros(n_blocks, dtype=np.int32)

        # Initial gains: for each subset, add 1 to all blocks containing it
        for u in range(n_subsets):
            for b in subset_to_blocks[u]:
                gains[b] += 1

        blocks_found = []
        used_blocks = set()
        local_deadline = time.time() + time_limit

        while True:
            if time.time() > local_deadline:
                break

            if not uncovered.any():
                break

            # Compute ratio for each block: gain / (covered_count + 1)
            # Only consider blocks with gain > 0
            best_b = -1
            best_ratio = -1.0

            # Use vectorized approach for speed
            mask = (gains > 0)
            if not mask.any():
                break

            # Create a temporary array for ratios, with -inf for used or zero-gain blocks
            ratios = np.full(n_blocks, -1.0, dtype=np.float64)
            valid_mask = mask & ~np.isin(np.arange(n_blocks), list(used_blocks))
            if valid_mask.any():
                # Compute ratios only for valid blocks
                valid_indices = np.where(valid_mask)[0]
                ratios[valid_indices] = gains[valid_indices].astype(np.float64) / (covered_counts[valid_indices] + 1.0)
                
                # Add small random perturbation if seed != 0 to break ties
                if seed != 0:
                    perturbation = rng.random() * 0.01
                    ratios[valid_indices] += perturbation * (np.random.random(len(valid_indices)) - 0.5)
                
                best_idx = np.argmax(ratios)
                if ratios[best_idx] > 0:
                    best_b = best_idx
                    best_ratio = ratios[best_idx]

            if best_b < 0 or best_ratio <= 0:
                break

            # Add this block
            blocks_found.append(list(all_blocks[best_b]))
            used_blocks.add(best_b)

            # Update uncovered and gains
            newly_covered = []
            for u in block_to_subsets[best_b]:
                if uncovered[u]:
                    uncovered[u] = False
                    newly_covered.append(u)
                    # Decrement gain for all blocks containing u
                    for b in subset_to_blocks[u]:
                        gains[b] -= 1

            # For each newly covered subset, increment covered_counts for all blocks containing it
            for u in newly_covered:
                for b in subset_to_blocks[u]:
                    covered_counts[b] += 1

            # Early exit if all covered
            if not uncovered.any():
                break

        # Polish: add blocks for remaining uncovered subsets
        if uncovered.any() and time.time() < local_deadline:
            uncov_indices = np.where(uncovered)[0]
            for u_idx in uncov_indices:
                if time.time() > local_deadline - 1.0:
                    break
                best_b = -1
                best_g = 0
                for b in subset_to_blocks[u_idx]:
                    if b not in used_blocks:
                        g = gains[b]
                        if g > best_g:
                            best_g = g
                            best_b = b
                if best_b >= 0 and best_g > 0:
                    blocks_found.append(list(all_blocks[best_b]))
                    used_blocks.add(best_b)
                    for u in block_to_subsets[best_b]:
                        if uncovered[u]:
                            uncovered[u] = False
                            for b in subset_to_blocks[u]:
                                gains[b] -= 1
                                covered_counts[b] += 1
                    if not uncovered.any():
                        break

        # Local search: remove redundant blocks
        if time.time() < local_deadline - 2.0 and blocks_found:
            # Map from block tuple to its index in all_blocks
            block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_blocks)}
            selected_block_indices = [block_tuple_to_idx[tuple(b)] for b in blocks_found]

            # Count how many selected blocks cover each subset
            cover_count = np.zeros(n_subsets, dtype=np.int32)
            for b_idx in selected_block_indices:
                for u in block_to_subsets[b_idx]:
                    cover_count[u] += 1

            # Try to remove blocks that don't reduce coverage below 1
            improved = True
            while improved and time.time() < local_deadline - 1.0:
                improved = False
                for i in range(len(selected_block_indices) - 1, -1, -1):
                    if time.time() > local_deadline - 1.0:
                        break
                    b_idx = selected_block_indices[i]
                    # Check if this block is the only one covering any subset
                    redundant = True
                    for u in block_to_subsets[b_idx]:
                        if cover_count[u] == 1:
                            redundant = False
                            break
                    if redundant:
                        # Remove this block
                        selected_block_indices.pop(i)
                        # Update cover_count
                        for u in block_to_subsets[b_idx]:
                            cover_count[u] -= 1
                        improved = True
                        break  # Restart the loop since indices changed

            blocks_found = [list(all_blocks[b_idx]) for b_idx in selected_block_indices]

        return blocks_found, uncovered

    # Run multiple greedy attempts with different seeds
    best_blocks = None
    best_count = float('inf')
    best_uncovered = None

    # First, run deterministic greedy
    blocks, uncovered = greedy_cover(0, 60.0)
    if len(blocks) < best_count:
        best_blocks = blocks
        best_count = len(blocks)
        best_uncovered = uncovered.copy()

    # Then run randomized greedy with multiple seeds
    if time.time() < deadline - 30.0:
        num_restarts = 5
        time_per_restart = min(20.0, (deadline - time.time() - 30.0) / num_restarts)
        for seed in range(1, num_restarts + 1):
            if time.time() > deadline - 30.0:
                break
            blocks, uncovered = greedy_cover(seed, time_per_restart)
            if len(blocks) < best_count:
                best_blocks = blocks
                best_count = len(blocks)
                best_uncovered = uncovered.copy()

    # Final polish on the best solution
    if best_blocks is not None and time.time() < deadline - 5.0:
        # Ensure all subsets are covered
        block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_blocks)}
        selected_block_indices = [block_tuple_to_idx[tuple(b)] for b in best_blocks]
        cover_count = np.zeros(n_subsets, dtype=np.int32)
        for b_idx in selected_block_indices:
            for u in block_to_subsets[b_idx]:
                cover_count[u] += 1

        # Find uncovered subsets and add blocks for them
        uncov = np.where(cover_count == 0)[0]
        for u_idx in uncov:
            if time.time() > deadline - 2.0:
                break
            # Find any block that covers this subset
            for b in subset_to_blocks[u_idx]:
                if b not in selected_block_indices:
                    best_blocks.append(list(all_blocks[b]))
                    selected_block_indices.append(b)
                    for u in block_to_subsets[b]:
                        cover_count[u] += 1
                    break

    return best_blocks if best_blocks is not None else []
