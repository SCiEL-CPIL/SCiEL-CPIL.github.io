import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    # Step 1: Generate all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subsets)}
    n_t = len(t_subsets)

    # Step 2: Generate all k-subsets (candidate blocks)
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)

    # Step 3: Precompute for each block which t-subsets it covers
    # block_t_indices[i] is a numpy array of t-subset indices covered by block i
    # Each block covers C(k, t) t-subsets
    t_in_k = list(itertools.combinations(range(k), t))
    n_t_in_k = len(t_in_k)

    block_t_indices = np.empty((n_k, n_t_in_k), dtype=np.int32)
    for i, block in enumerate(k_subsets):
        for j, lc in enumerate(t_in_k):
            block_t_indices[i, j] = t_sub_to_idx[tuple(block[p] for p in lc)]

    # Step 4: Build t_to_blocks mapping for swap operations
    # t_to_blocks[t_idx] = list of block indices that cover t_subsets[t_idx]
    t_to_blocks = [[] for _ in range(n_t)]
    for i in range(n_k):
        for j in range(n_t_in_k):
            t_idx = block_t_indices[i, j]
            t_to_blocks[t_idx].append(i)
    
    # Convert to sets for fast intersection
    t_to_blocks_sets = [set(lst) for lst in t_to_blocks]

    rng = random.Random(42)
    best_solution = None
    best_size = float('inf')

    # Multiple restarts with randomized greedy
    while elapsed() < deadline - 10.0:
        # Reset state
        uncovered = np.ones(n_t, dtype=bool)
        selected_blocks = []
        selected_set = set()

        # Greedy Set Cover with random tie-breaking
        while elapsed() < deadline - 5.0:
            # Calculate gains: how many uncovered t-subsets each block covers
            gains = uncovered[block_t_indices].sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break

            # Find all blocks with max gain and pick randomly
            candidates = np.where(gains == max_gain)[0]
            best_idx = rng.choice(candidates)

            selected_blocks.append(best_idx)
            selected_set.add(best_idx)

            # Update uncovered
            uncovered[block_t_indices[best_idx]] = False

        if elapsed() > deadline - 5.0:
            break

        if not selected_blocks:
            continue

        # Post-optimization: Local Search with Redundancy Removal and Swaps
        # Build coverage count
        coverage_count = np.zeros(n_t, dtype=np.int32)
        for bi in selected_blocks:
            coverage_count[block_t_indices[bi]] += 1

        remaining = list(selected_blocks)
        remaining_set = set(remaining)

        improved = True
        while improved and elapsed() < deadline - 2.0:
            improved = False

            # 1. Remove redundant blocks iteratively
            changed = True
            while changed and elapsed() < deadline - 2.0:
                changed = False
                for i in range(len(remaining)):
                    bi = remaining[i]
                    ts = block_t_indices[bi]
                    if np.all(coverage_count[ts] > 1):
                        coverage_count[ts] -= 1
                        remaining.pop(i)
                        remaining_set.remove(bi)
                        changed = True
                        break  # Restart scan

            if elapsed() > deadline - 2.0:
                break

            # 2. Try swaps to escape local optima
            for i in range(len(remaining)):
                if elapsed() > deadline - 2.0:
                    break

                bi = remaining[i]
                ts = block_t_indices[bi]

                # Find unique t-subsets covered by bi (only covered by this block)
                unique_ts = ts[coverage_count[ts] == 1]

                if len(unique_ts) == 0:
                    # This block is redundant
                    coverage_count[ts] -= 1
                    remaining.pop(i)
                    remaining_set.remove(bi)
                    improved = True
                    break  # Restart outer loop

                # Find candidates B that cover all unique_ts
                # Intersection of t_to_blocks_sets for each t in unique_ts
                if len(unique_ts) == 1:
                    candidates = set(t_to_blocks_sets[unique_ts[0]])
                else:
                    candidates = set(t_to_blocks_sets[unique_ts[0]])
                    for t_idx in unique_ts[1:]:
                        candidates = candidates.intersection(t_to_blocks_sets[t_idx])
                        if not candidates:
                            break

                # Filter out blocks already in solution
                candidates.difference_update(remaining_set)

                if candidates:
                    # Pick a random candidate for swap
                    new_bi = rng.choice(list(candidates))

                    # Perform swap
                    coverage_count[block_t_indices[bi]] -= 1
                    remaining_set.remove(bi)
                    remaining.pop(i)

                    coverage_count[block_t_indices[new_bi]] += 1
                    remaining_set.add(new_bi)
                    remaining.append(new_bi)

                    improved = True
                    break  # Restart outer loop after a swap

            # If we improved, check if this is a new best
            if improved and len(remaining) < best_size:
                best_size = len(remaining)
                best_solution = [k_subsets[i] for i in remaining]

        # Update best if this solution is better
        if len(remaining) < best_size:
            best_size = len(remaining)
            best_solution = [k_subsets[i] for i in remaining]

    if best_solution is None:
        return []

    return [list(b) for b in best_solution]
