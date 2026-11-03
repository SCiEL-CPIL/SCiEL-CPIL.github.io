"""Improved covering design search for C(19,8,5) with deadline guard.
Combines greedy construction with block-index based local search using
redundancy removal and swap strategies."""
import time
import random
import itertools
import math
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def elapsed():
        return time.time() - start_time

    # Precompute all t-subsets
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}

    # Precompute all k-blocks and their tset coverage
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # block_to_tsets[b] = list of tset indices covered by block b
    # tset_to_blocks[i] = list of block indices covering tset i
    block_to_tsets = [[] for _ in range(num_blocks)]
    tset_to_blocks = [[] for _ in range(num_tsets)]

    for b_idx, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            ts_idx = tset_to_idx[ts]
            block_to_tsets[b_idx].append(ts_idx)
            tset_to_blocks[ts_idx].append(b_idx)

    # Convert to numpy arrays for faster operations
    # For each tset, store its covering blocks as a numpy array
    tset_to_blocks_np = [np.array(tset_to_blocks[i], dtype=np.int32) for i in range(num_tsets)]

    def run_greedy(seed):
        """Greedy set cover using precomputed block-tset mappings."""
        rng = random.Random(seed)
        is_uncovered = np.ones(num_tsets, dtype=bool)
        uncovered_count = num_tsets
        selected_blocks = []

        while uncovered_count > 0:
            if elapsed() > deadline - 25:
                break

            # Pick a random uncovered tset
            for _ in range(50):
                idx = rng.randint(0, num_tsets - 1)
                if is_uncovered[idx]:
                    break
            else:
                # Scan
                for i in range(num_tsets):
                    if is_uncovered[i]:
                        idx = i
                        break
                else:
                    break

            # Find the block that covers the most uncovered tsets
            # among those covering this tset
            candidates = tset_to_blocks[idx]
            if not candidates:
                break

            best_block = -1
            best_gain = -1

            # Sample if too many candidates
            if len(candidates) > 200:
                sample = rng.sample(candidates, 200)
            else:
                sample = candidates

            for b_idx in sample:
                gain = 0
                for ts_idx in block_to_tsets[b_idx]:
                    if is_uncovered[ts_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx

            if best_block == -1:
                break

            selected_blocks.append(best_block)

            # Update coverage
            for ts_idx in block_to_tsets[best_block]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    uncovered_count -= 1

        return selected_blocks

    def compute_cover_counts(block_indices):
        """Compute cover count for each tset given a list of block indices."""
        counts = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in block_indices:
            for ts_idx in block_to_tsets[b_idx]:
                counts[ts_idx] += 1
        return counts

    def local_search(initial_indices, max_time=10.0):
        """Local search: redundancy removal + swaps."""
        current = list(initial_indices)
        if not current:
            return []

        cover_counts = compute_cover_counts(current)
        best_size = len(current)
        best_solution = list(current)

        local_start = time.time()
        improved = True
        no_improvement_rounds = 0
        max_no_improvement = 4

        while improved and (time.time() - local_start) < max_time:
            improved = False

            # Phase 1: Remove redundant blocks (multiple passes until stable)
            while True:
                found_redundant = False
                for i in range(len(current) - 1, -1, -1):
                    if (time.time() - local_start) > max_time:
                        break
                    b_idx = current[i]
                    is_redundant = True
                    for ts_idx in block_to_tsets[b_idx]:
                        if cover_counts[ts_idx] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        for ts_idx in block_to_tsets[b_idx]:
                            cover_counts[ts_idx] -= 1
                        current.pop(i)
                        found_redundant = True
                        improved = True
                        if len(current) < best_size:
                            best_size = len(current)
                            best_solution = list(current)

                if not found_redundant or (time.time() - local_start) > max_time:
                    break

            if not current:
                break

            # Phase 2: 1-for-1 Swaps
            for strategy in range(3):
                if (time.time() - local_start) > max_time:
                    break

                rng = random.Random(hash((strategy, best_size)) % (2**31))
                shuffled = list(current)
                rng.shuffle(shuffled)

                for i in range(len(shuffled)):
                    if (time.time() - local_start) > max_time:
                        break

                    b_out = shuffled[i]

                    # Find tsets uniquely covered by b_out
                    unique_tsets = []
                    for ts_idx in block_to_tsets[b_out]:
                        if cover_counts[ts_idx] == 1:
                            unique_tsets.append(ts_idx)

                    if not unique_tsets:
                        continue

                    # Find candidate blocks that cover all unique tsets
                    candidates = set(tset_to_blocks[unique_tsets[0]])
                    candidates.discard(b_out)

                    for ts_idx in unique_tsets[1:]:
                        if not candidates:
                            break
                        candidates.intersection_update(tset_to_blocks[ts_idx])

                    if not candidates:
                        continue

                    best_candidate = -1
                    best_score = -1
                    b_out_tsets_set = set(block_to_tsets[b_out])

                    candidate_list = list(candidates)
                    limit = [100, 200, 400][strategy]
                    if len(candidate_list) > limit:
                        rng.shuffle(candidate_list)
                        candidate_list = candidate_list[:limit]

                    for b_in in candidate_list:
                        # Score: how many new tsets does b_in cover that b_out doesn't?
                        # Plus bonus for covering sparse tsets
                        score = 0
                        for ts_idx in block_to_tsets[b_in]:
                            if ts_idx not in b_out_tsets_set:
                                if strategy == 0:
                                    score += 1
                                elif strategy == 1:
                                    if cover_counts[ts_idx] == 1:
                                        score += 10
                                    elif cover_counts[ts_idx] == 2:
                                        score += 5
                                    else:
                                        score += 1
                                else:
                                    if cover_counts[ts_idx] == 1:
                                        score += 15
                                    elif cover_counts[ts_idx] == 2:
                                        score += 6
                                    elif cover_counts[ts_idx] == 3:
                                        score += 2
                                    else:
                                        score += 1

                        if score > best_score:
                            best_score = score
                            best_candidate = b_in

                    if best_candidate != -1:
                        # Perform swap
                        # Remove b_out
                        for ts_idx in block_to_tsets[b_out]:
                            cover_counts[ts_idx] -= 1
                        current[i] = best_candidate
                        # Add b_in
                        for ts_idx in block_to_tsets[best_candidate]:
                            cover_counts[ts_idx] += 1
                        improved = True
                        if len(current) < best_size:
                            best_size = len(current)
                            best_solution = list(current)

            if not improved:
                no_improvement_rounds += 1
                if no_improvement_rounds >= max_no_improvement:
                    break
            else:
                no_improvement_rounds = 0

        return best_solution

    # Phase 1: Run greedy with multiple seeds
    best_blocks = []
    best_len = float('inf')

    num_seeds = 80
    for i in range(num_seeds):
        if elapsed() > deadline - 35:
            break
        blocks = run_greedy(i)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks

    # Phase 2: Local search on best solution
    if best_blocks and elapsed() < deadline - 10:
        improved = local_search(best_blocks, max_time=min(20.0, deadline - time.time() - 5))
        if len(improved) < len(best_blocks):
            best_blocks = improved
            best_len = len(improved)

    # Phase 3: Try a few more seeds with local search
    for i in range(num_seeds, num_seeds + 20):
        if elapsed() > deadline - 8:
            break
        blocks = run_greedy(i)
        if len(blocks) <= best_len + 5:  # Only local search if close
            improved = local_search(blocks, max_time=min(5.0, deadline - time.time() - 3))
            if len(improved) < best_len:
                best_len = len(improved)
                best_blocks = improved

    # Final: convert block indices to actual blocks
    result = [list(all_blocks[i]) for i in best_blocks]

    # Verify no duplicates
    seen = set()
    unique_result = []
    for b in result:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_result.append(b)

    return unique_result
