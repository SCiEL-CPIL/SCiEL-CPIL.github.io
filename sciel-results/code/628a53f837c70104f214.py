import itertools
import time
import numpy as np
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # 1. Precomputation
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}

    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # Build incidence structures
    block_to_tsets = [[] for _ in range(num_blocks)]
    tset_to_blocks = [[] for _ in range(num_tsets)]

    for b_idx, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            ts_idx = tset_to_idx[ts]
            block_to_tsets[b_idx].append(ts_idx)
            tset_to_blocks[ts_idx].append(b_idx)

    # Bitmasks for each block (set of tsets it covers)
    block_bitmasks = [0] * num_blocks
    for b_idx in range(num_blocks):
        mask = 0
        for ts_idx in block_to_tsets[b_idx]:
            mask |= (1 << ts_idx)
        block_bitmasks[b_idx] = mask

    # For each tset, bitmask of blocks covering it
    tset_block_bitmasks = [0] * num_tsets
    for ts_idx in range(num_tsets):
        mask = 0
        for b_idx in tset_to_blocks[ts_idx]:
            mask |= (1 << b_idx)
        tset_block_bitmasks[ts_idx] = mask

    def get_solution(indices):
        return [list(all_blocks[i]) for i in indices]

    def run_local_search(initial_indices, max_time_per_run=10.0):
        current = list(initial_indices)
        if not current:
            return []

        cover_counts = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in current:
            for ts_idx in block_to_tsets[b_idx]:
                cover_counts[ts_idx] += 1

        best_size = len(current)
        best_solution = list(current)
        local_start = time.time()
        no_improvement_rounds = 0
        max_no_improvement = 4

        improved = True
        while improved and (time.time() - local_start) < max_time_per_run:
            improved = False

            # Phase 1: Remove redundant blocks
            while True:
                found_redundant = False
                for i in range(len(current) - 1, -1, -1):
                    if (time.time() - local_start) > max_time_per_run:
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
                if not found_redundant or (time.time() - local_start) > max_time_per_run:
                    break

            if not current:
                break

            # Phase 2: 1-for-1 Swaps with bitmask acceleration
            random.shuffle(current)
            for i in range(len(current)):
                if (time.time() - local_start) > max_time_per_run:
                    break
                b_out = current[i]

                # Find tsets uniquely covered by b_out
                unique_tsets = []
                for ts_idx in block_to_tsets[b_out]:
                    if cover_counts[ts_idx] == 1:
                        unique_tsets.append(ts_idx)

                if not unique_tsets:
                    continue

                # Use bitmasks to find candidate blocks that cover ALL unique tsets
                candidate_mask = tset_block_bitmasks[unique_tsets[0]]
                for ts_idx in unique_tsets[1:]:
                    candidate_mask &= tset_block_bitmasks[ts_idx]
                    if candidate_mask == 0:
                        break

                if candidate_mask == 0:
                    continue

                # Extract candidate block indices from bitmask
                candidates = []
                cm = candidate_mask
                while cm:
                    low_bit = cm & (-cm)
                    b_idx = low_bit.bit_length() - 1
                    if b_idx != b_out:
                        candidates.append(b_idx)
                    cm ^= low_bit

                if not candidates:
                    continue

                # Score candidates: maximize new coverage
                b_out_tset_set = set(block_to_tsets[b_out])
                best_candidate = -1
                best_score = -1

                for b_in in candidates:
                    score = 0
                    for ts_idx in block_to_tsets[b_in]:
                        if ts_idx not in b_out_tset_set:
                            if cover_counts[ts_idx] == 1:
                                score += 15
                            elif cover_counts[ts_idx] == 2:
                                score += 5
                            else:
                                score += 1
                    if score > best_score:
                        best_score = score
                        best_candidate = b_in

                if best_candidate != -1:
                    for ts_idx in block_to_tsets[b_out]:
                        cover_counts[ts_idx] -= 1
                    for ts_idx in block_to_tsets[best_candidate]:
                        cover_counts[ts_idx] += 1
                    current[i] = best_candidate
                    improved = True

            # Phase 3: 2-for-1 swaps (replace one block with two)
            if (time.time() - local_start) < max_time_per_run:
                random.shuffle(current)
                for i in range(len(current)):
                    if (time.time() - local_start) > max_time_per_run:
                        break
                    b_out = current[i]

                    unique_tsets = []
                    for ts_idx in block_to_tsets[b_out]:
                        if cover_counts[ts_idx] == 1:
                            unique_tsets.append(ts_idx)

                    if not unique_tsets:
                        continue

                    # Find blocks that cover the most unique tsets
                    best_pair = None
                    best_pair_score = -1
                    best_pair_new = 0

                    # Get blocks covering at least one unique tset
                    candidate_blocks = set()
                    for ts_idx in unique_tsets:
                        candidate_blocks.update(tset_to_blocks[ts_idx])
                    candidate_blocks.discard(b_out)

                    if len(candidate_blocks) > 200:
                        candidate_blocks = set(random.sample(list(candidate_blocks), 200))

                    candidate_list = list(candidate_blocks)
                    # Score each candidate by how many unique tsets it covers
                    scored_candidates = []
                    for b_in in candidate_list:
                        count = 0
                        for ts_idx in unique_tsets:
                            if ts_idx in block_to_tsets[b_in]:
                                count += 1
                        if count > 0:
                            scored_candidates.append((count, b_in))

                    scored_candidates.sort(reverse=True)
                    top_candidates = [b for _, b in scored_candidates[:15]]

                    # Try pairs from top candidates
                    for a_idx in range(len(top_candidates)):
                        for b_idx2 in range(a_idx + 1, len(top_candidates)):
                            b_in1 = top_candidates[a_idx]
                            b_in2 = top_candidates[b_idx2]

                            # Check if together they cover all unique tsets
                            covered_unique = set()
                            for ts_idx in block_to_tsets[b_in1]:
                                if ts_idx in set(unique_tsets):
                                    covered_unique.add(ts_idx)
                            for ts_idx in block_to_tsets[b_in2]:
                                if ts_idx in set(unique_tsets):
                                    covered_unique.add(ts_idx)

                            if len(covered_unique) < len(unique_tsets):
                                continue

                            # Score: new non-overlapping coverage
                            b_out_tset_set = set(block_to_tsets[b_out])
                            new_coverage = 0
                            for ts_idx in block_to_tsets[b_in1]:
                                if ts_idx not in b_out_tset_set:
                                    if cover_counts[ts_idx] == 1:
                                        new_coverage += 10
                                    elif cover_counts[ts_idx] == 2:
                                        new_coverage += 3
                                    else:
                                        new_coverage += 1
                            for ts_idx in block_to_tsets[b_in2]:
                                if ts_idx not in b_out_tset_set:
                                    if cover_counts[ts_idx] == 1:
                                        new_coverage += 10
                                    elif cover_counts[ts_idx] == 2:
                                        new_coverage += 3
                                    else:
                                        new_coverage += 1

                            if new_coverage > best_pair_score:
                                best_pair_score = new_coverage
                                best_pair = (b_in1, b_in2)

                    if best_pair is not None:
                        b_in1, b_in2 = best_pair
                        for ts_idx in block_to_tsets[b_out]:
                            cover_counts[ts_idx] -= 1
                        for ts_idx in block_to_tsets[b_in1]:
                            cover_counts[ts_idx] += 1
                        for ts_idx in block_to_tsets[b_in2]:
                            cover_counts[ts_idx] += 1
                        current[i] = b_in1
                        current.append(b_in2)
                        improved = True

            if not improved:
                no_improvement_rounds += 1
                if no_improvement_rounds >= max_no_improvement:
                    break
            else:
                no_improvement_rounds = 0

        return best_solution

    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = np.ones(num_tsets, dtype=np.bool_)
        uncovered_count = num_tsets
        selected = []

        # Weights: 1 / frequency
        tset_counts = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in range(num_blocks):
            for ts_idx in block_to_tsets[b_idx]:
                tset_counts[ts_idx] += 1
        safe_counts = np.maximum(tset_counts, 1).astype(np.float64)
        tset_weights = 1.0 / safe_counts

        gains = np.zeros(num_blocks, dtype=np.float64)
        for b_idx in range(num_blocks):
            if block_to_tsets[b_idx]:
                gains[b_idx] = np.sum(tset_weights[block_to_tsets[b_idx]])

        selected_mask = np.zeros(num_blocks, dtype=np.bool_)

        while uncovered_count > 0 and time.time() < deadline - 20:
            masked_gains = gains.copy()
            masked_gains[selected_mask] = -1.0
            best_idx = int(np.argmax(masked_gains))
            best_gain = gains[best_idx]

            if best_gain <= 0:
                break

            selected.append(best_idx)
            selected_mask[best_idx] = True

            for ts_idx in block_to_tsets[best_idx]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    uncovered_count -= 1
                    w = tset_weights[ts_idx]
                    for b in tset_to_blocks[ts_idx]:
                        if not selected_mask[b]:
                            gains[b] -= w

        return selected

    best_global_solution = None
    best_global_size = float('inf')

    # Run 1: Primary greedy + extended local search
    t_start = time.time()
    init1 = greedy_cover(42)
    if init1:
        ls_time = min(45.0, deadline - t_start - 10)
        sol1 = run_local_search(init1, max_time_per_run=ls_time)
        if sol1 and len(sol1) < best_global_size:
            best_global_size = len(sol1)
            best_global_solution = sol1

    # Run 2: Different seed
    if time.time() < deadline - 35:
        t_start = time.time()
        init2 = greedy_cover(123)
        if init2:
            ls_time = min(25.0, deadline - t_start - 10)
            sol2 = run_local_search(init2, max_time_per_run=ls_time)
            if sol2 and len(sol2) < best_global_size:
                best_global_size = len(sol2)
                best_global_solution = sol2

    # Run 3: Another seed
    if time.time() < deadline - 20:
        t_start = time.time()
        init3 = greedy_cover(777)
        if init3:
            ls_time = min(12.0, deadline - t_start - 5)
            sol3 = run_local_search(init3, max_time_per_run=ls_time)
            if sol3 and len(sol3) < best_global_size:
                best_global_size = len(sol3)
                best_global_solution = sol3

    # Final polish
    if best_global_solution and time.time() < deadline - 5:
        remaining_time = deadline - time.time() - 2
        if remaining_time > 3:
            improved_sol = run_local_search(best_global_solution, max_time_per_run=remaining_time)
            if improved_sol and len(improved_sol) < best_global_size:
                best_global_size = len(improved_sol)
                best_global_solution = improved_sol

    if not best_global_solution:
        if init1:
            return get_solution(init1)
        return []

    return get_solution(best_global_solution)

if __name__ == "__main__":
    v, k, t = 19, 8, 5
    res = construct(v, k, t)
    print(f"Number of blocks: {len(res)}")
    covered = set()
    for block in res:
        for ts in itertools.combinations(block, t):
            covered.add(ts)
    total_tsets = len(list(itertools.combinations(range(v), t)))
    print(f"Covered: {len(covered)}/{total_tsets}")
