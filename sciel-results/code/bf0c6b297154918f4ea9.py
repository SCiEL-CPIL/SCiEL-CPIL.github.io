import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_t_subsets)
    t_subset_to_index = {s: i for i, s in enumerate(all_t_subsets)}

    all_k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_subsets)

    block_to_tsubs = []
    block_to_tsubs_set = []
    for block in all_k_subsets:
        covered = []
        for tsub in itertools.combinations(block, t):
            idx = t_subset_to_index[tsub]
            covered.append(idx)
        block_to_tsubs.append(covered)
        block_to_tsubs_set.append(set(covered))

    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for b_idx, tsubs in enumerate(block_to_tsubs):
        for s_idx in tsubs:
            tsub_to_blocks[s_idx].append(b_idx)

    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]

    # 2. Greedy Construction
    uncovered = np.ones(num_tsubs, dtype=bool)
    k_choose_t = len(block_to_tsubs[0])
    gains = np.full(num_blocks, k_choose_t, dtype=np.int32)

    selected_block_indices = []

    while uncovered.any():
        if time.time() > deadline:
            break

        b = np.argmax(gains)

        if gains[b] == 0:
            break

        selected_block_indices.append(b)

        for s_idx in block_to_tsubs[b]:
            if uncovered[s_idx]:
                uncovered[s_idx] = False
                gains[tsub_to_blocks_np[s_idx]] -= 1

    if not uncovered.any():
        # 3. Local Optimization with Simulated Annealing
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in selected_block_indices:
            for s in block_to_tsubs[b]:
                cover_count[s] += 1

        def is_redundant(b_idx):
            for s in block_to_tsubs[b_idx]:
                if cover_count[s] == 1:
                    return False
            return True

        def find_replacement(unique_tsubs, exclude_set):
            if not unique_tsubs:
                return None

            first_s = unique_tsubs[0]
            candidates = tsub_to_blocks_np[first_s]

            valid_candidates = []
            for c in candidates:
                if c in exclude_set:
                    continue
                c_set = block_to_tsubs_set[c]
                covers_all = True
                for s in unique_tsubs:
                    if s not in c_set:
                        covers_all = False
                        break
                if covers_all:
                    valid_candidates.append(c)

            if not valid_candidates:
                return None

            return random.choice(valid_candidates)

        def find_2for1_replacement(unique_tsubs_1, unique_tsubs_2, exclude_set):
            """Find a single block that covers all unique t-subsets from two blocks."""
            all_unique = list(set(unique_tsubs_1) | set(unique_tsubs_2))
            if not all_unique:
                return None

            first_s = all_unique[0]
            candidates = tsub_to_blocks_np[first_s]

            valid_candidates = []
            for c in candidates:
                if c in exclude_set:
                    continue
                c_set = block_to_tsubs_set[c]
                covers_all = True
                for s in all_unique:
                    if s not in c_set:
                        covers_all = False
                        break
                if covers_all:
                    valid_candidates.append(c)

            if not valid_candidates:
                return None

            return random.choice(valid_candidates)

        # Simulated annealing parameters
        best_solution = list(selected_block_indices)
        best_size = len(selected_block_indices)
        current_solution = list(selected_block_indices)
        current_size = len(current_solution)

        temperature = 50.0
        cooling_rate = 0.995
        min_temperature = 1.0
        max_iterations = 5000

        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            if temperature < min_temperature:
                break

            # Try different move types
            move_type = random.random()
            new_solution = list(current_solution)
            new_cover_count = cover_count.copy()

            if move_type < 0.3:
                # Try removing a redundant block
                redundant = [b for b in new_solution if is_redundant(b)]
                if redundant:
                    b_to_remove = random.choice(redundant)
                    new_solution.remove(b_to_remove)
                    for s in block_to_tsubs[b_to_remove]:
                        new_cover_count[s] -= 1
                    if len(new_solution) < current_size:
                        current_solution = new_solution
                        cover_count = new_cover_count
                        current_size = len(current_solution)
                        if current_size < best_size:
                            best_solution = list(current_solution)
                            best_size = current_size
                        continue

            elif move_type < 0.7:
                # Try 1-for-1 swap
                unique_tsubs_map = {}
                for b in new_solution:
                    u_list = [s for s in block_to_tsubs[b] if cover_count[s] == 1]
                    unique_tsubs_map[b] = u_list

                non_redundant = [b for b in new_solution if unique_tsubs_map[b]]
                if non_redundant:
                    b_out = random.choice(non_redundant)
                    exclude_set = set(new_solution)
                    b_in = find_replacement(unique_tsubs_map[b_out], exclude_set)
                    if b_in is not None:
                        new_solution.remove(b_out)
                        new_solution.append(b_in)
                        for s in block_to_tsubs[b_out]:
                            new_cover_count[s] -= 1
                        for s in block_to_tsubs[b_in]:
                            new_cover_count[s] += 1
                        # Check validity
                        valid = True
                        for s in range(num_tsubs):
                            if new_cover_count[s] < 1:
                                valid = False
                                break
                        if valid:
                            delta = len(new_solution) - current_size
                            if delta <= 0 or random.random() < np.exp(-delta / temperature):
                                current_solution = new_solution
                                cover_count = new_cover_count
                                current_size = len(current_solution)
                                if current_size < best_size:
                                    best_solution = list(current_solution)
                                    best_size = current_size
                        else:
                            # Invalid, skip
                            pass

            else:
                # Try 2-for-1 swap (remove 2 blocks, add 1)
                if len(new_solution) >= 2:
                    b1, b2 = random.sample(new_solution, 2)
                    unique1 = [s for s in block_to_tsubs[b1] if cover_count[s] == 1]
                    unique2 = [s for s in block_to_tsubs[b2] if cover_count[s] == 1]
                    exclude_set = set(new_solution)
                    b_in = find_2for1_replacement(unique1, unique2, exclude_set)
                    if b_in is not None:
                        new_solution.remove(b1)
                        new_solution.remove(b2)
                        new_solution.append(b_in)
                        for s in block_to_tsubs[b1]:
                            new_cover_count[s] -= 1
                        for s in block_to_tsubs[b2]:
                            new_cover_count[s] -= 1
                        for s in block_to_tsubs[b_in]:
                            new_cover_count[s] += 1
                        valid = True
                        for s in range(num_tsubs):
                            if new_cover_count[s] < 1:
                                valid = False
                                break
                        if valid:
                            delta = len(new_solution) - current_size
                            if delta <= 0 or random.random() < np.exp(-delta / temperature):
                                current_solution = new_solution
                                cover_count = new_cover_count
                                current_size = len(current_solution)
                                if current_size < best_size:
                                    best_solution = list(current_solution)
                                    best_size = current_size

            temperature *= cooling_rate

        # Final cleanup: remove any redundant blocks from best solution
        final_cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in best_solution:
            for s in block_to_tsubs[b]:
                final_cover_count[s] += 1

        changed = True
        while changed:
            changed = False
            new_final = []
            for b in best_solution:
                redundant = True
                for s in block_to_tsubs[b]:
                    if final_cover_count[s] == 1:
                        redundant = False
                        break
                if not redundant:
                    new_final.append(b)
                else:
                    changed = True
                    for s in block_to_tsubs[b]:
                        final_cover_count[s] -= 1
            best_solution = new_final

        result = [list(all_k_subsets[i]) for i in best_solution]
        return result
    else:
        return []
