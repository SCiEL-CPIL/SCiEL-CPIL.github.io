import itertools
import time
import random
import sys

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
    # Using sets for fast intersection
    subset_to_blocks = [set() for _ in range(num_t_subsets)]
    for i in range(num_blocks):
        for idx in block_subsets[i]:
            subset_to_blocks[idx].add(i)

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

    def local_search(chosen_indices, time_limit=5.0):
        """
        1-Swap Local Search with neutral move acceptance.
        Tries to replace a block with another that covers its essential parts,
        potentially allowing further pruning.
        """
        current_solution = prune_solution(list(chosen_indices))
        start_ls = time.time()

        improved = True
        while improved and time.time() - start_ls < time_limit:
            improved = False

            # Compute counts
            counts = [0] * num_t_subsets
            for idx in current_solution:
                for s_idx in block_subsets[idx]:
                    counts[s_idx] += 1

            for i, idx in enumerate(current_solution):
                # Find essential subsets for idx
                essential_subs = [s for s in block_subsets[idx] if counts[s] == 1]

                if not essential_subs:
                    # Should not happen after prune, but just in case
                    for s in block_subsets[idx]:
                        counts[s] -= 1
                    current_solution.pop(i)
                    improved = True
                    break

                # Find candidate blocks that cover ALL essential subsets
                if len(essential_subs) == 1:
                    pool = subset_to_blocks[essential_subs[0]]
                else:
                    min_set_idx = min(essential_subs, key=lambda s: len(subset_to_blocks[s]))
                    pool = set(subset_to_blocks[min_set_idx])
                    for s in essential_subs:
                        if s == min_set_idx:
                            continue
                        pool &= subset_to_blocks[s]
                        if not pool:
                            break

                if not pool:
                    continue

                found_swap = False
                candidates = list(pool)
                random.shuffle(candidates)

                for cand_idx in candidates:
                    if cand_idx == idx:
                        continue
                    if cand_idx in current_solution:
                        continue

                    test_sol = list(current_solution)
                    test_sol[i] = cand_idx
                    pruned_test = prune_solution(test_sol)

                    if len(pruned_test) < len(current_solution):
                        current_solution = pruned_test
                        improved = True
                        found_swap = True
                        break
                    elif len(pruned_test) == len(current_solution):
                        # Accept neutral move with small probability to explore
                        if random.random() < 0.1:
                            current_solution = pruned_test
                            improved = True
                            found_swap = True
                            break

                if found_swap:
                    break

        return current_solution

    def penalty_local_search(chosen_indices, time_limit=10.0):
        """
        Randomized local search with penalty-based objective.
        Moves: swap one block for another, or add/remove single blocks.
        Objective: uncovered_count + penalty * block_count
        Accepts worse moves with simulated annealing probability to escape local optima.
        """
        current_solution = list(chosen_indices)
        start_ls = time.time()
        temperature = 50.0  # Initial temperature for simulated annealing
        cooling = 0.95
        best_solution = list(current_solution)
        best_objective = float('inf')
        no_improve = 0
        max_no_improve = 200

        while time.time() - start_ls < time_limit:
            # Compute current objective
            uncovered_mask = full_mask
            for idx in current_solution:
                uncovered_mask &= ~block_masks[idx]
            uncovered_count = uncovered_mask.bit_count()
            current_objective = uncovered_count + 0.1 * len(current_solution)

            if current_objective < best_objective:
                best_objective = current_objective
                best_solution = list(current_solution)
                no_improve = 0
            else:
                no_improve += 1

            if no_improve > max_no_improve:
                # Restart from best
                current_solution = list(best_solution)
                no_improve = 0
                temperature = max(temperature, 20.0)

            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.6 and len(current_solution) > 0:
                # Swap move: replace one block with another
                i = random.randint(0, len(current_solution) - 1)
                old_idx = current_solution[i]
                
                # Find candidates that cover at least some of the old block's subsets
                # Use a random subset of the old block's t-subsets to find candidates
                old_subs = block_subsets[old_idx]
                if old_subs:
                    # Pick a random essential subset to find candidates
                    test_sub = random.choice(old_subs)
                    candidates = list(subset_to_blocks[test_sub])
                    candidates = [c for c in candidates if c != old_idx and c not in current_solution]
                    if candidates:
                        new_idx = random.choice(candidates)
                        test_sol = list(current_solution)
                        test_sol[i] = new_idx
                        # Evaluate
                        test_uncovered = full_mask
                        for idx in test_sol:
                            test_uncovered &= ~block_masks[idx]
                        test_obj = test_uncovered.bit_count() + 0.1 * len(test_sol)
                        delta_obj = test_obj - current_objective
                        # Accept if better or with probability exp(-delta/T)
                        if delta_obj < 0 or (temperature > 1.0 and random.random() < (2.718281828 ** (-delta_obj / temperature))):
                            current_solution = test_sol
            elif move_type < 0.8:
                # Add move: add a random block
                # Pick a block that covers many uncovered subsets
                uncovered_mask = full_mask
                for idx in current_solution:
                    uncovered_mask &= ~block_masks[idx]
                
                if uncovered_mask:
                    # Find blocks with good gain
                    best_gain = -1
                    best_candidates = []
                    for i in range(num_blocks):
                        if i in current_solution:
                            continue
                        gain = (block_masks[i] & uncovered_mask).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_candidates = [i]
                        elif gain == best_gain and gain > 0:
                            best_candidates.append(i)
                    
                    if best_candidates and best_gain > 0:
                        new_idx = random.choice(best_candidates)
                        test_sol = list(current_solution) + [new_idx]
                        test_uncovered = full_mask
                        for idx in test_sol:
                            test_uncovered &= ~block_masks[idx]
                        test_obj = test_uncovered.bit_count() + 0.1 * len(test_sol)
                        delta_obj = test_obj - current_objective
                        if delta_obj < 0 or (temperature > 1.0 and random.random() < (2.718281828 ** (-delta_obj / temperature))):
                            current_solution = test_sol
            else:
                # Remove move: remove a random block (only if coverage is maintained or penalty improves)
                if len(current_solution) > 1:
                    i = random.randint(0, len(current_solution) - 1)
                    old_idx = current_solution[i]
                    test_sol = [b for j, b in enumerate(current_solution) if j != i]
                    
                    # Check if coverage is still complete
                    test_uncovered = full_mask
                    for idx in test_sol:
                        test_uncovered &= ~block_masks[idx]
                    test_obj = test_uncovered.bit_count() + 0.1 * len(test_sol)
                    delta_obj = test_obj - current_objective
                    if delta_obj < 0 or (temperature > 1.0 and random.random() < (2.718281828 ** (-delta_obj / temperature))):
                        current_solution = test_sol

            # Cool down
            temperature *= cooling
            if temperature < 1.0:
                temperature = 1.0

        # Return the best valid solution found
        # Ensure the best solution is valid (covers all t-subsets)
        best_uncovered = full_mask
        for idx in best_solution:
            best_uncovered &= ~block_masks[idx]
        
        if best_uncovered == 0:
            return best_solution
        else:
            # Fallback to current solution if it's valid
            cur_uncovered = full_mask
            for idx in current_solution:
                cur_uncovered &= ~block_masks[idx]
            if cur_uncovered == 0:
                return current_solution
            else:
                # Return the one with fewer uncovered
                if best_uncovered.bit_count() <= cur_uncovered.bit_count():
                    return best_solution
                else:
                    return current_solution

    def greedy_construct(randomize=False):
        """
        Greedy set cover.
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
                    if randomize:
                        if random.random() < 0.5:
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
    sol = local_search(sol, 5.0)
    best_solution = sol
    best_len = len(sol)

    # 2. Iterated Local Search with penalty-based search
    no_improve_count = 0
    max_no_improve = 8

    while time.time() < deadline - 10.0:
        if no_improve_count > max_no_improve:
            # Restart with random greedy
            sol = greedy_construct(randomize=True)
            sol = local_search(sol, 3.0)
            if len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol
                no_improve_count = 0
            else:
                no_improve_count += 1
            continue

        # Kick: Remove a random block from best_solution and repair
        if best_len > 0:
            # Choose a block to remove.
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
            if random.random() < 0.5:
                kick_idx = random.randint(0, len(best_solution) - 1)

            new_base = [b for i, b in enumerate(best_solution) if i != kick_idx]

            # Repair with Greedy
            sol = greedy_repair(new_base)
            sol = local_search(sol, 4.0)

            # Also try penalty-based local search
            if time.time() < deadline - 15.0:
                sol2 = penalty_local_search(sol, 5.0)
                if len(sol2) < len(sol):
                    sol = sol2

            if len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol
                no_improve_count = 0
            else:
                no_improve_count += 1

    # Final polish
    if best_solution and time.time() < deadline - 5.0:
        best_solution = local_search(best_solution, 3.0)
        # Try penalty-based polish
        if time.time() < deadline - 3.0:
            polished = penalty_local_search(best_solution, 2.0)
            if len(polished) < len(best_solution):
                best_solution = polished

    # Convert to lists of points
    result_blocks = [list(all_blocks[i]) for i in best_solution]

    return result_blocks

if __name__ == "__main__":
    blocks = construct(19, 7, 4)
    print(f"Number of blocks: {len(blocks)}")
