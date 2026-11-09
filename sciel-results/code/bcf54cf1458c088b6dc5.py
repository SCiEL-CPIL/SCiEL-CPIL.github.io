import itertools
import time
import random
import sys
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}

    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # Precompute coverage of each block as a list of t-subset indices
    # and as a bitmask for fast intersection
    block_subsets = [[] for _ in range(num_blocks)]
    block_masks = [0] * num_blocks

    for i, block in enumerate(all_blocks):
        mask = 0
        subs = []
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
            subs.append(idx)
        block_masks[i] = mask
        block_subsets[i] = subs

    # Inverted index: for each t-subset, which blocks contain it
    # Using lists for memory efficiency, converted to sets only when needed
    subset_to_blocks = [[] for _ in range(num_t_subsets)]
    for i in range(num_blocks):
        for idx in block_subsets[i]:
            subset_to_blocks[idx].append(i)
    # Convert to frozensets for faster intersection
    subset_to_blocks_frozen = [frozenset(lst) for lst in subset_to_blocks]

    # Full mask for all t-subsets
    full_mask = (1 << num_t_subsets) - 1

    def prune_solution(chosen_indices):
        """
        Removes redundant blocks. A block is redundant if all its t-subsets
        are covered by at least one other block in the chosen set.
        Returns a new list of indices.
        """
        if not chosen_indices:
            return []

        # Count coverage for each t-subset
        counts = [0] * num_t_subsets
        for idx in chosen_indices:
            for s_idx in block_subsets[idx]:
                counts[s_idx] += 1

        current_list = list(chosen_indices)
        changed = True

        while changed:
            changed = False
            new_list = []
            for idx in current_list:
                essential = False
                for s_idx in block_subsets[idx]:
                    if counts[s_idx] == 1:
                        essential = True
                        break
                if not essential:
                    # Remove this block
                    for s_idx in block_subsets[idx]:
                        counts[s_idx] -= 1
                    changed = True
                else:
                    new_list.append(idx)
            current_list = new_list

        return current_list

    def fast_local_search(chosen_indices, time_limit=5.0):
        """
        Optimized local search that avoids full prune on every swap.
        Uses incremental count updates and batched pruning.
        """
        current_solution = prune_solution(list(chosen_indices))
        if not current_solution:
            return current_solution
        
        sol_set = set(current_solution)
        start_ls = time.time()
        swaps_since_prune = 0
        max_swaps_before_prune = 3

        improved = True
        while improved and time.time() - start_ls < time_limit:
            improved = False
            swaps_since_prune = 0

            # Compute counts once per outer iteration
            counts = [0] * num_t_subsets
            for idx in current_solution:
                for s_idx in block_subsets[idx]:
                    counts[s_idx] += 1

            # Try 1-swap moves
            for i, idx in enumerate(current_solution):
                if time.time() - start_ls > time_limit:
                    break

                # Find essential subsets for idx
                essential_subs = [s for s in block_subsets[idx] if counts[s] == 1]

                if not essential_subs:
                    # Redundant block - remove it
                    for s in block_subsets[idx]:
                        counts[s] -= 1
                    current_solution.pop(i)
                    sol_set.discard(idx)
                    improved = True
                    break

                # Find candidate blocks that cover ALL essential subsets
                if len(essential_subs) == 1:
                    pool = subset_to_blocks_frozen[essential_subs[0]]
                else:
                    # Start with the smallest set
                    min_set_idx = min(essential_subs, key=lambda s: len(subset_to_blocks_frozen[s]))
                    pool = subset_to_blocks_frozen[min_set_idx]
                    for s in essential_subs:
                        if s != min_set_idx:
                            pool = pool & subset_to_blocks_frozen[s]
                            if not pool:
                                break

                if not pool:
                    continue

                # Filter candidates
                found_swap = False
                candidates = list(pool - sol_set - {idx})
                random.shuffle(candidates)

                for cand_idx in candidates:
                    if time.time() - start_ls > time_limit:
                        break

                    # Simulate swap: remove idx, add cand_idx
                    # Check if this creates redundancy that can be pruned
                    # Quick check: does cand_idx cover all essential_subs of idx?
                    # (It should by construction, but verify no new essential subsets become uncovered)

                    # New counts after swap
                    # Remove idx's contribution
                    for s in block_subsets[idx]:
                        counts[s] -= 1
                    # Add cand_idx's contribution
                    for s in block_subsets[cand_idx]:
                        counts[s] += 1

                    # Check if any t-subset is now uncovered (count == 0)
                    valid_swap = True
                    for s in block_subsets[idx]:
                        if counts[s] == 0:
                            valid_swap = False
                            break

                    if valid_swap:
                        # Check if we can prune any blocks (including possibly cand_idx itself)
                        # Do a quick prune check
                        new_sol = list(current_solution)
                        new_sol[i] = cand_idx
                        pruned = prune_solution(new_sol)

                        if len(pruned) < len(current_solution):
                            current_solution = pruned
                            sol_set = set(pruned)
                            improved = True
                            found_swap = True
                            swaps_since_prune += 1
                            # Reset counts since we pruned
                            counts = [0] * num_t_subsets
                            for idx2 in current_solution:
                                for s_idx in block_subsets[idx2]:
                                    counts[s_idx] += 1
                            break
                        else:
                            # Roll back counts
                            for s in block_subsets[cand_idx]:
                                counts[s] -= 1
                            for s in block_subsets[idx]:
                                counts[s] += 1
                    else:
                        # Roll back counts
                        for s in block_subsets[cand_idx]:
                            counts[s] -= 1
                        for s in block_subsets[idx]:
                            counts[s] += 1

                if found_swap:
                    break

            # Try 2-swap moves: remove 2 blocks, add 1 block
            if not improved and len(current_solution) >= 2 and time.time() - start_ls < time_limit:
                # Recompute counts
                counts = [0] * num_t_subsets
                for idx in current_solution:
                    for s_idx in block_subsets[idx]:
                        counts[s_idx] += 1

                # Find pairs of blocks to remove
                n = len(current_solution)
                for i in range(n):
                    if time.time() - start_ls > time_limit:
                        break
                    for j in range(i + 1, n):
                        if time.time() - start_ls > time_limit:
                            break

                        idx1 = current_solution[i]
                        idx2 = current_solution[j]

                        # Essential subsets of idx1 and idx2
                        ess1 = [s for s in block_subsets[idx1] if counts[s] == 1]
                        ess2 = [s for s in block_subsets[idx2] if counts[s] == 1]

                        # If either has no essential subsets, it's redundant
                        if not ess1 or not ess2:
                            continue

                        # Union of essential subsets
                        all_ess = set(ess1) | set(ess2)

                        # Find blocks that cover all essential subsets
                        if len(all_ess) == 1:
                            pool = subset_to_blocks_frozen[list(all_ess)[0]]
                        else:
                            min_s = min(all_ess, key=lambda s: len(subset_to_blocks_frozen[s]))
                            pool = subset_to_blocks_frozen[min_s]
                            for s in all_ess:
                                if s != min_s:
                                    pool = pool & subset_to_blocks_frozen[s]
                                    if not pool:
                                        break

                        if not pool:
                            continue

                        candidates = list(pool - sol_set - {idx1, idx2})
                        random.shuffle(candidates)

                        for cand_idx in candidates[:10]:  # Limit candidates
                            if time.time() - start_ls > time_limit:
                                break

                            # Simulate: remove idx1, idx2, add cand_idx
                            for s in block_subsets[idx1]:
                                counts[s] -= 1
                            for s in block_subsets[idx2]:
                                counts[s] -= 1
                            for s in block_subsets[cand_idx]:
                                counts[s] += 1

                            valid = True
                            for s in set(block_subsets[idx1]) | set(block_subsets[idx2]):
                                if counts[s] == 0:
                                    valid = False
                                    break

                            if valid:
                                new_sol = [x for x in current_solution if x not in (idx1, idx2)]
                                new_sol.append(cand_idx)
                                pruned = prune_solution(new_sol)

                                if len(pruned) < len(current_solution):
                                    current_solution = pruned
                                    sol_set = set(pruned)
                                    improved = True
                                    # Reset counts
                                    counts = [0] * num_t_subsets
                                    for idx3 in current_solution:
                                        for s_idx in block_subsets[idx3]:
                                            counts[s_idx] += 1
                                    break
                            else:
                                # Roll back
                                for s in block_subsets[cand_idx]:
                                    counts[s] -= 1
                                for s in block_subsets[idx1]:
                                    counts[s] += 1
                                for s in block_subsets[idx2]:
                                    counts[s] += 1

                        if improved:
                            break
                    if improved:
                        break

        return current_solution

    def greedy_construct(randomize=False):
        """
        Greedy set cover with early termination for zero-gain blocks.
        """
        uncovered = full_mask
        chosen = []
        used = [False] * num_blocks

        while uncovered != 0:
            if time.time() > deadline - 5.0:
                break

            best_gain = -1
            best_idx = -1
            second_best_gain = -1
            second_best_idx = -1

            for i in range(num_blocks):
                if used[i]:
                    continue
                gain = (block_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    second_best_gain = best_gain
                    second_best_idx = best_idx
                    best_gain = gain
                    best_idx = i
                elif gain == best_gain and gain > 0:
                    if randomize and random.random() < 0.5:
                        second_best_gain = best_gain
                        second_best_idx = best_idx
                        best_gain = gain
                        best_idx = i

            if best_idx == -1:
                break

            if randomize and second_best_idx != -1 and random.random() < 0.1:
                best_idx = second_best_idx

            chosen.append(best_idx)
            used[best_idx] = True
            uncovered &= ~block_masks[best_idx]

        return chosen

    def greedy_repair(start_set):
        """
        Repair a partial covering by greedily adding blocks.
        """
        uncovered = full_mask
        chosen = list(start_set)
        used = [False] * num_blocks
        for idx in chosen:
            used[idx] = True
            uncovered &= ~block_masks[idx]

        while uncovered != 0:
            if time.time() > deadline - 5.0:
                break
            best_gain = -1
            best_idx = -1
            for i in range(num_blocks):
                if used[i]:
                    continue
                gain = (block_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            chosen.append(best_idx)
            used[best_idx] = True
            uncovered &= ~block_masks[best_idx]
        return chosen

    # Main Loop
    best_solution = None
    best_len = float('inf')

    # 1. Initial Deterministic Greedy
    sol = greedy_construct(randomize=False)
    sol = fast_local_search(sol, 8.0)
    best_solution = sol
    best_len = len(sol)

    # 2. Iterated Local Search with adaptive time budget
    no_improve_count = 0
    max_no_improve = 15
    ls_time_budget = 5.0

    while time.time() < deadline - 10.0:
        # Adaptive time budget for local search
        remaining = deadline - 10.0 - time.time()
        current_ls_time = min(ls_time_budget, max(2.0, remaining * 0.15))

        if no_improve_count > max_no_improve:
            # Restart with random greedy
            sol = greedy_construct(randomize=True)
            sol = fast_local_search(sol, current_ls_time)
            if len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol
                no_improve_count = 0
                ls_time_budget = min(8.0, ls_time_budget * 1.2)
            else:
                no_improve_count += 1
                ls_time_budget = max(2.0, ls_time_budget * 0.9)
            continue

        # Kick: Remove a block from best_solution and repair
        if best_len > 0:
            # Compute counts to find least essential
            counts = [0] * num_t_subsets
            for idx in best_solution:
                for s_idx in block_subsets[idx]:
                    counts[s_idx] += 1

            # Find block with minimum unique coverage
            min_unique = float('inf')
            kick_idx = 0
            for i, idx in enumerate(best_solution):
                unique_count = sum(1 for s in block_subsets[idx] if counts[s] == 1)
                if unique_count < min_unique:
                    min_unique = unique_count
                    kick_idx = i

            # Sometimes use random kick
            if random.random() < 0.4:
                kick_idx = random.randint(0, len(best_solution) - 1)

            new_base = [b for i, b in enumerate(best_solution) if i != kick_idx]

            # Repair with Greedy
            sol = greedy_repair(new_base)
            sol = fast_local_search(sol, current_ls_time)

            if len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol
                no_improve_count = 0
                ls_time_budget = min(8.0, ls_time_budget * 1.1)
            else:
                no_improve_count += 1
                ls_time_budget = max(2.0, ls_time_budget * 0.95)

    # Final polish with longer time budget
    if best_solution and time.time() < deadline - 3.0:
        remaining = deadline - 3.0 - time.time()
        best_solution = fast_local_search(best_solution, min(10.0, remaining))

    # Convert to lists of points
    result_blocks = [list(all_blocks[i]) for i in best_solution]

    return result_blocks

if __name__ == "__main__":
    blocks = construct(19, 7, 4)
    print(f"Number of blocks: {len(blocks)}")
