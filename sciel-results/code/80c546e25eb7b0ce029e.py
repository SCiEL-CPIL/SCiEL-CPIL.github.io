import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = 110.0 # Safety margin

    def elapsed():
        return time.time() - start_time

    # 1. Precomputation
    # Generate all t-sets and assign indices
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsets)}
    
    # Generate all blocks and their bitmasks
    # Optimization: Pre-compute list of tset indices for each block to speed up local search
    all_blocks = []
    block_masks = []
    block_tsets = [] # List of lists of tset indices covered by each block
    
    for block in itertools.combinations(range(v), k):
        mask = 0
        block_ts_list = []
        for ts in itertools.combinations(block, t):
            idx = tset_index[ts]
            mask |= (1 << idx)
            block_ts_list.append(idx)
        all_blocks.append(block)
        block_masks.append(mask)
        block_tsets.append(block_ts_list)
        
    num_blocks = len(all_blocks)
    
    # Initial full mask
    full_mask = (1 << num_tsets) - 1

    def greedy_cover(seed):
        """
        Standard Greedy Set Cover.
        At each step, select the block that covers the maximum number of currently uncovered t-sets.
        This is guaranteed to produce a valid cover if completed.
        """
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        
        # To break ties randomly (for diversity in restarts), we can add a small random noise
        # or just pick the first max. For speed, we'll pick the first max, but shuffle block order?
        # No, shuffling block order every iteration is expensive.
        # Instead, we'll just do standard greedy. The local search will handle the optimization.
        
        while uncovered:
            if elapsed() > deadline - 5:
                break
                
            best_idx = -1
            best_count = -1
            
            # Iterate through all blocks to find the one covering the most uncovered t-sets
            for i in range(num_blocks):
                # If the block is already selected, skip it (though greedy doesn't re-select)
                # Optimization: We can skip blocks that are subsets of previously selected ones?
                # No, simple bitmask intersection is fast.
                
                count = (block_masks[i] & uncovered).bit_count()
                
                # Tie-breaking: if count is equal, we can use random tie-breaking to get diverse solutions
                if count > best_count:
                    best_count = count
                    best_idx = i
                elif count == best_count and count > 0:
                    # Random tie-breaker to encourage diversity between restarts
                    if rng.random() < 0.5:
                        best_idx = i
                        
            if best_idx == -1 or best_count == 0:
                # No block covers any new t-set? This shouldn't happen unless uncovered is empty or we ran out of time/logic error
                break
                
            selected.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        return selected

    # 3. Local Search (1-for-1 swap and deletion)
    # Build tset_to_blocks once
    tset_to_blocks = [[] for _ in range(num_tsets)]
    for b_idx, ts_list in enumerate(block_tsets):
        for ts_idx in ts_list:
            tset_to_blocks[ts_idx].append(b_idx)
            
    def local_search_v2(initial_sol, seed):
        rng = random.Random(seed)
        sol = list(initial_sol)
        in_sol = [False] * num_blocks
        for b in sol:
            in_sol[b] = True

        cover_counts = [0] * num_tsets

        def update_counts(block_idx, delta):
            for ts_idx in block_tsets[block_idx]:
                cover_counts[ts_idx] += delta

        for b in sol:
            update_counts(b, 1)

        best_sol = list(sol)
        best_size = len(sol)
        no_improve = 0
        max_no_improve = 80

        # Simulated annealing temperature
        temperature = 10.0
        cooling = 0.97

        while elapsed() < deadline - 10 and no_improve < max_no_improve:
            improved = False
            accepted_neutral = False

            # Phase 1: Remove redundant
            order = list(range(len(sol)))
            rng.shuffle(order)
            for pos in order:
                if pos >= len(sol):
                    continue
                b = sol[pos]
                redundant = True
                for ts_idx in block_tsets[b]:
                    if cover_counts[ts_idx] == 1:
                        redundant = False
                        break

                if redundant:
                    update_counts(b, -1)
                    in_sol[b] = False
                    sol.pop(pos)
                    improved = True
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    break

            if improved:
                no_improve = 0
                temperature = 10.0
                continue

            # Phase 2: 1-for-1 Swap
            if sol:
                found_swap = False
                max_swap_attempts = 30
                if elapsed() < deadline - 30:
                    max_swap_attempts = 60

                for _ in range(max_swap_attempts):
                    if not sol:
                        break
                    pos = rng.randint(0, len(sol) - 1)
                    b_out = sol[pos]

                    critical_mask = 0
                    crit_ts_list = []
                    for ts_idx in block_tsets[b_out]:
                        if cover_counts[ts_idx] == 1:
                            critical_mask |= (1 << ts_idx)
                            crit_ts_list.append(ts_idx)

                    if critical_mask == 0:
                        # Redundant
                        update_counts(b_out, -1)
                        in_sol[b_out] = False
                        sol.pop(pos)
                        improved = True
                        found_swap = True
                        break

                    if not crit_ts_list:
                        continue

                    # Optimization: Find the critical tset with the fewest covering blocks
                    min_ts_idx = crit_ts_list[0]
                    min_len = len(tset_to_blocks[min_ts_idx])
                    for ts_idx in crit_ts_list[1:]:
                        curr_len = len(tset_to_blocks[ts_idx])
                        if curr_len < min_len:
                            min_ts_idx = ts_idx
                            min_len = curr_len

                    # Filter candidates from the shortest list
                    valid_candidates = []
                    for b_idx in tset_to_blocks[min_ts_idx]:
                        if in_sol[b_idx]:
                            continue
                        if (block_masks[b_idx] & critical_mask) == critical_mask:
                            valid_candidates.append(b_idx)

                    if valid_candidates:
                        b_in = rng.choice(valid_candidates)
                        update_counts(b_out, -1)
                        in_sol[b_out] = False
                        update_counts(b_in, 1)
                        in_sol[b_in] = True
                        sol[pos] = b_in
                        found_swap = True
                        improved = True
                        break

                if found_swap:
                    no_improve = 0
                    temperature = 10.0
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    continue

                # Phase 3: 2-for-1 replacement (replace two blocks with one)
                if len(sol) >= 2:
                    found_2for1 = False
                    max_2for1_attempts = 20
                    if elapsed() < deadline - 30:
                        max_2for1_attempts = 40

                    for _ in range(max_2for1_attempts):
                        if len(sol) < 2:
                            break
                        # Pick two distinct positions
                        pos1 = rng.randint(0, len(sol) - 1)
                        pos2 = rng.randint(0, len(sol) - 1)
                        if pos2 == pos1:
                            pos2 = (pos2 + 1) % len(sol)

                        b1 = sol[pos1]
                        b2 = sol[pos2]

                        # Compute combined critical mask (t-sets covered by exactly one of b1, b2)
                        # A t-set is critical for the pair if removing both would uncover it
                        combined_crit = 0
                        for ts_idx in block_tsets[b1]:
                            if cover_counts[ts_idx] == 1:
                                combined_crit |= (1 << ts_idx)
                        for ts_idx in block_tsets[b2]:
                            if cover_counts[ts_idx] == 1:
                                combined_crit |= (1 << ts_idx)

                        if combined_crit == 0:
                            # Both are redundant
                            # Remove both
                            update_counts(b1, -1)
                            in_sol[b1] = False
                            update_counts(b2, -1)
                            in_sol[b2] = False
                            # Remove from sol (handle indices carefully)
                            if pos1 > pos2:
                                pos1, pos2 = pos2, pos1
                            sol.pop(pos2)
                            sol.pop(pos1)
                            improved = True
                            found_2for1 = True
                            break

                        # Find a single block that covers all critical t-sets
                        # Use the rarest critical t-set as seed
                        crit_list = []
                        cm = combined_crit
                        while cm:
                            lsb = cm & (-cm)
                            ts_idx = lsb.bit_length() - 1
                            crit_list.append(ts_idx)
                            cm ^= lsb

                        if not crit_list:
                            continue

                        min_ts_idx = crit_list[0]
                        min_len = len(tset_to_blocks[min_ts_idx])
                        for ts_idx in crit_list[1:]:
                            curr_len = len(tset_to_blocks[ts_idx])
                            if curr_len < min_len:
                                min_ts_idx = ts_idx
                                min_len = curr_len

                        valid_candidates = []
                        for b_idx in tset_to_blocks[min_ts_idx]:
                            if in_sol[b_idx] and b_idx != b1 and b_idx != b2:
                                continue
                            if b_idx == b1 or b_idx == b2:
                                continue
                            if (block_masks[b_idx] & combined_crit) == combined_crit:
                                valid_candidates.append(b_idx)

                        if valid_candidates:
                            b_in = rng.choice(valid_candidates)
                            # Remove b1 and b2, add b_in
                            update_counts(b1, -1)
                            in_sol[b1] = False
                            update_counts(b2, -1)
                            in_sol[b2] = False
                            update_counts(b_in, 1)
                            in_sol[b_in] = True
                            if pos1 > pos2:
                                pos1, pos2 = pos2, pos1
                            sol.pop(pos2)
                            sol.pop(pos1)
                            sol.append(b_in)
                            found_2for1 = True
                            improved = True
                            break

                    if found_2for1:
                        no_improve = 0
                        temperature = 10.0
                        if len(sol) < best_size:
                            best_size = len(sol)
                            best_sol = list(sol)
                        continue

                # Phase 4: Simulated annealing - accept neutral or slightly worse moves
                # to escape local minima
                if temperature > 1.0:
                    pos = rng.randint(0, len(sol) - 1)
                    b_out = sol[pos]

                    critical_mask = 0
                    crit_ts_list = []
                    for ts_idx in block_tsets[b_out]:
                        if cover_counts[ts_idx] == 1:
                            critical_mask |= (1 << ts_idx)
                            crit_ts_list.append(ts_idx)

                    if critical_mask == 0:
                        # Redundant - remove
                        update_counts(b_out, -1)
                        in_sol[b_out] = False
                        sol.pop(pos)
                        improved = True
                        no_improve = 0
                        temperature = 10.0
                        if len(sol) < best_size:
                            best_size = len(sol)
                            best_sol = list(sol)
                        continue

                    # Find candidates that cover at least some critical t-sets
                    min_ts_idx = crit_ts_list[0]
                    min_len = len(tset_to_blocks[min_ts_idx])
                    for ts_idx in crit_ts_list[1:]:
                        curr_len = len(tset_to_blocks[ts_idx])
                        if curr_len < min_len:
                            min_ts_idx = ts_idx
                            min_len = curr_len

                    partial_candidates = []
                    for b_idx in tset_to_blocks[min_ts_idx]:
                        if in_sol[b_idx]:
                            continue
                        # Count how many critical t-sets this block covers
                        covered = (block_masks[b_idx] & critical_mask).bit_count()
                        if covered > 0:
                            partial_candidates.append((b_idx, covered))

                    if partial_candidates:
                        # Accept with probability based on how much coverage we gain
                        # and temperature
                        b_in, covered = rng.choice(partial_candidates)
                        # Acceptance: if covered == len(crit_ts_list), it's a full swap (always accept)
                        # Otherwise, accept with probability exp(temperature * (covered - len(crit_ts_list)) / len(crit_ts_list))
                        if covered == len(crit_ts_list):
                            accept = True
                        else:
                            delta = (covered - len(crit_ts_list)) / len(crit_ts_list)
                            accept = rng.random() < np.exp(temperature * delta)

                        if accept:
                            # Perform the swap (may leave some critical t-sets uncovered temporarily)
                            # But we must ensure validity: only accept if result is still valid
                            # Check: after swap, are all t-sets still covered?
                            # We need to verify that no t-set becomes uncovered
                            # A t-set becomes uncovered if it was covered only by b_out and not by b_in
                            # Since b_in covers 'covered' of the critical t-sets, the rest remain covered by other blocks
                            # Actually, critical t-sets are those with cover_count == 1, so if b_in doesn't cover them,
                            # they become uncovered. So we can only accept if covered == len(crit_ts_list)
                            # OR if we're willing to allow temporary invalidity (not allowed here)
                            # So for safety, only accept full swaps in annealing
                            pass
                        # For safety, only do full swaps in this phase (already handled in Phase 2)
                        # Instead, try a different block as a neutral move
                        # Pick a block that covers the same critical set
                        for b_idx, cov in partial_candidates:
                            if cov == len(crit_ts_list):
                                update_counts(b_out, -1)
                                in_sol[b_out] = False
                                update_counts(b_idx, 1)
                                in_sol[b_idx] = True
                                sol[pos] = b_idx
                                accepted_neutral = True
                                break

                    if accepted_neutral:
                        no_improve = 0
                        continue

            else:
                break

            temperature *= cooling
            no_improve += 1

        return best_sol

    # Main Loop
    best_solution = None
    best_size = float('inf')
    
    # Run multiple restarts of (Greedy + Local Search)
    # Since Greedy is fast, we can afford more restarts to find diverse starting points
    # for the Local Search.
    num_restarts = 20
    
    for seed in range(num_restarts):
        if elapsed() > deadline - 20:
            break
            
        sol = greedy_cover(seed)
        
        # Verify coverage (safety check)
        if sol:
            # Check if sol covers everything
            temp_uncovered = full_mask
            for b_idx in sol:
                temp_uncovered &= ~block_masks[b_idx]
            
            if temp_uncovered == 0:
                ls_sol = local_search_v2(sol, seed)
                if len(ls_sol) < best_size:
                    best_size = len(ls_sol)
                    best_solution = ls_sol
            else:
                # If greedy didn't finish (timeout), we might have an incomplete cover.
                # We should try to complete it with a quick greedy finish if time permits,
                # but for v=19, greedy should finish instantly.
                # If it didn't, it's likely a bug or extreme timeout.
                pass

    # Additional polish: try more aggressive local search on the best solution
    if best_solution is not None and elapsed() < deadline - 10:
        # Run additional local search iterations with different random seeds
        for extra_seed in range(5):
            if elapsed() > deadline - 5:
                break
            polished = local_search_v2(best_solution, 1000 + extra_seed)
            if len(polished) < best_size:
                best_size = len(polished)
                best_solution = polished
    
    if best_solution is None:
        # Fallback: if we somehow didn't get a solution, return empty (invalid)
        # But for this problem size, this should not happen.
        return []
        
    return [list(all_blocks[i]) for i in best_solution]
