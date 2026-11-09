import itertools
import time
import random
import sys
from collections import defaultdict

def construct(v, k, t):
    """
    Solve C(v,k,t) using Simulated Annealing over block additions/removals.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin from 120s limit

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # For each block, which t-subsets it covers
    block_covered_t = [None] * num_blocks
    t_to_blocks = [[] for _ in range(num_t_subsets)]

    for i, b in enumerate(all_blocks):
        covered_indices = []
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx[ts]
            covered_indices.append(idx)
            t_to_blocks[idx].append(i)
        block_covered_t[i] = covered_indices

    def run_greedy(seed):
        """Standard randomized greedy set cover."""
        random.seed(seed)
        uncovered_set = set(range(num_t_subsets))
        selected = []

        while uncovered_set:
            if time.time() > deadline:
                break
            target_t_idx = random.choice(list(uncovered_set))
            candidates = t_to_blocks[target_t_idx]

            best_block = -1
            best_gain = -1
            best_blocks_with_gain = []

            for b_idx in candidates:
                gain = 0
                for t_idx in block_covered_t[b_idx]:
                    if t_idx in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    best_blocks_with_gain = [b_idx]
                elif gain == best_gain:
                    best_blocks_with_gain.append(b_idx)

            if best_block == -1:
                break

            best_block = random.choice(best_blocks_with_gain)
            selected.append(best_block)
            for t_idx in block_covered_t[best_block]:
                if t_idx in uncovered_set:
                    uncovered_set.discard(t_idx)

        return selected

    def try_remove_blocks(blocks):
        """Remove redundant blocks (those whose all covered t-subsets have count > 1)."""
        if not blocks:
            return blocks

        cover_count = [0] * num_t_subsets
        for b_idx in blocks:
            for t_idx in block_covered_t[b_idx]:
                cover_count[t_idx] += 1

        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5.0:
                    return blocks
                b_idx = blocks[i]
                is_redundant = True
                for t_idx in block_covered_t[b_idx]:
                    if cover_count[t_idx] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for t_idx in block_covered_t[b_idx]:
                        cover_count[t_idx] -= 1
                    blocks.pop(i)
                    changed = True
                    break
        return blocks

    def simulated_annealing(initial_blocks, time_budget=30.0):
        """
        SA on the block list. Moves:
        1. Add a random block, then try to remove 2 redundant blocks (net -1)
        2. Swap one block for another (net 0, may enable future removals)
        3. Remove one redundant block if possible (net -1)
        Accept worse states with probability exp(-delta/T).
        """
        if not initial_blocks:
            return initial_blocks

        current_blocks = list(initial_blocks)
        cover_count = [0] * num_t_subsets
        for b_idx in current_blocks:
            for t_idx in block_covered_t[b_idx]:
                cover_count[t_idx] += 1

        best_blocks = list(current_blocks)
        best_len = len(current_blocks)

        start_temp = 5.0
        end_temp = 0.1
        cooling = 0.995

        current_time = start_time
        sa_deadline = current_time + time_budget

        temp = start_temp
        iteration = 0

        while time.time() < sa_deadline and temp > end_temp:
            iteration += 1
            temp *= cooling

            move_type = random.random()
            new_blocks = list(current_blocks)
            new_cover = list(cover_count)
            delta = 0  # change in block count (negative = improvement)

            if move_type < 0.5:
                # Try: add a block, then remove up to 2 redundant blocks
                # Pick a random block to add
                add_idx = random.randint(0, num_blocks - 1)
                # Check if this block is already in the list
                if add_idx not in new_blocks:
                    new_blocks.append(add_idx)
                    for t_idx in block_covered_t[add_idx]:
                        new_cover[t_idx] += 1

                    # Try to remove 2 redundant blocks
                    removed = 0
                    for attempt in range(2):
                        if not new_blocks:
                            break
                        # Find a redundant block
                        redundant_idx = -1
                        for i in range(len(new_blocks)):
                            b_idx = new_blocks[i]
                            is_redundant = True
                            for t_idx in block_covered_t[b_idx]:
                                if new_cover[t_idx] <= 1:
                                    is_redundant = False
                                    break
                            if is_redundant:
                                redundant_idx = i
                                break
                        if redundant_idx == -1:
                            break
                        # Remove it
                        b_idx = new_blocks[redundant_idx]
                        for t_idx in block_covered_t[b_idx]:
                            new_cover[t_idx] -= 1
                        new_blocks.pop(redundant_idx)
                        removed += 1

                    delta = removed - 1  # net change in block count
                    # delta = -1 if we removed 2, 0 if we removed 1, 1 if we removed 0

            elif move_type < 0.8:
                # Swap: replace one block with another
                if len(current_blocks) > 0:
                    swap_out_idx = random.randint(0, len(current_blocks) - 1)
                    old_block = new_blocks[swap_out_idx]

                    # Find a replacement block that covers all essential t-subsets of old_block
                    essential_t = []
                    for t_idx in block_covered_t[old_block]:
                        if new_cover[t_idx] == 1:
                            essential_t.append(t_idx)

                    if not essential_t:
                        # Old block is redundant, just remove it
                        for t_idx in block_covered_t[old_block]:
                            new_cover[t_idx] -= 1
                        new_blocks.pop(swap_out_idx)
                        delta = -1
                    else:
                        # Try to find a replacement
                        best_cand = -1
                        best_gain = -1
                        # Sample candidates
                        candidates = random.sample(range(num_blocks), min(2000, num_blocks))
                        for cand_idx in candidates:
                            if cand_idx == old_block:
                                continue
                            # Check if cand covers all essential_t
                            covers_all = True
                            for t_idx in essential_t:
                                if t_idx not in block_covered_t[cand_idx]:
                                    covers_all = False
                                    break
                            if not covers_all:
                                continue
                            # Calculate gain: how many low-coverage t-subsets does it cover?
                            gain = 0
                            for t_idx in block_covered_t[cand_idx]:
                                if t_idx not in block_covered_t[old_block]:
                                    if new_cover[t_idx] <= 2:
                                        gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_cand = cand_idx

                        if best_cand != -1:
                            # Perform swap
                            for t_idx in block_covered_t[old_block]:
                                new_cover[t_idx] -= 1
                            for t_idx in block_covered_t[best_cand]:
                                new_cover[t_idx] += 1
                            new_blocks[swap_out_idx] = best_cand
                            delta = 0  # net 0 change, but may enable future improvements

            else:
                # Just try to remove a redundant block
                if new_blocks:
                    redundant_idx = -1
                    for i in range(len(new_blocks)):
                        b_idx = new_blocks[i]
                        is_redundant = True
                        for t_idx in block_covered_t[b_idx]:
                            if new_cover[t_idx] <= 1:
                                is_redundant = False
                            break
                        if is_redundant:
                            redundant_idx = i
                            break
                    if redundant_idx != -1:
                        b_idx = new_blocks[redundant_idx]
                        for t_idx in block_covered_t[b_idx]:
                            new_cover[t_idx] -= 1
                        new_blocks.pop(redundant_idx)
                        delta = -1
                    else:
                        delta = 0  # No change

            # Acceptance criterion
            if delta < 0:
                # Improvement, always accept
                current_blocks = new_blocks
                cover_count = new_cover
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = list(current_blocks)
            elif delta == 0:
                # Neutral, accept with some probability to explore
                if random.random() < 0.5:
                    current_blocks = new_blocks
                    cover_count = new_cover
            else:
                # Worse, accept with probability exp(-delta/temp)
                if random.random() < (2.718281828 ** (-delta / temp)):
                    current_blocks = new_blocks
                    cover_count = new_cover

        return best_blocks

    # Main search loop
    best_blocks = None
    best_len = float('inf')

    # Try several greedy starts, then SA on each
    seeds = [42, 123, 999, 0, 1, 2, 3, 4, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]

    for seed in seeds:
        if time.time() > deadline - 30.0:
            break

        greedy_res = run_greedy(seed)
        if not greedy_res:
            continue

        # First, remove obvious redundancies
        processed_res = try_remove_blocks(greedy_res)

        # Run SA
        sa_res = simulated_annealing(processed_res, time_budget=min(20.0, deadline - time.time() - 10.0))

        if len(sa_res) < best_len:
            best_len = len(sa_res)
            best_blocks = sa_res

        if best_len <= 69:
            break

    # If we have time, try more random seeds
    if time.time() < deadline - 20.0:
        while time.time() < deadline - 10.0:
            seed = random.randint(0, 1000000)
            greedy_res = run_greedy(seed)
            if not greedy_res:
                continue

            processed_res = try_remove_blocks(greedy_res)
            sa_res = simulated_annealing(processed_res, time_budget=min(15.0, deadline - time.time() - 10.0))

            if len(sa_res) < best_len:
                best_len = len(sa_res)
                best_blocks = sa_res

            if best_len <= 69:
                break

    if best_blocks is None:
        return []

    # Convert block indices back to actual blocks
    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_blocks]

    # Ensure distinctness and sorting
    seen = set()
    unique_blocks = []
    for b in final_blocks:
        b.sort()
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)

    return unique_blocks
