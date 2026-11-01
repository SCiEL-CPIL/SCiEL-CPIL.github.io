import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 150s limit

    # Precompute all k-blocks and t-subsets as bitmasks
    all_blocks = []
    for block in itertools.combinations(range(v), k):
        mask = 0
        for x in block:
            mask |= (1 << x)
        all_blocks.append(mask)
    num_blocks = len(all_blocks)

    all_ts = []
    for ts in itertools.combinations(range(v), t):
        mask = 0
        for x in ts:
            mask |= (1 << x)
        all_ts.append(mask)
    num_ts = len(all_ts)

    # Build incidence: for each block, list of ts indices it covers
    # And for each ts, list of block indices that cover it
    # Use the fact that ts is subset of block iff (ts_mask & block_mask) == ts_mask
    block_to_ts = [[] for _ in range(num_blocks)]
    ts_to_blocks = [[] for _ in range(num_ts)]

    for bi, bm in enumerate(all_blocks):
        for ti, tm in enumerate(all_ts):
            if (tm & bm) == tm:
                block_to_ts[bi].append(ti)
                ts_to_blocks[ti].append(bi)

    # Convert to numpy arrays for speed
    block_to_ts_arr = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in block_to_ts]
    ts_to_blocks_arr = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in ts_to_blocks]

    # Convert all_blocks to list of lists for output
    block_lists = [list(comb) for comb in itertools.combinations(range(v), k)]

    # ============ SCARCITY-WEIGHTED GREEDY ============
    def scarcity_greedy(seed=0, time_budget=None):
        """
        Greedy that uses dynamic scarcity weights:
        Each uncovered t-subset has a weight = 1 / (number of blocks that can still cover it).
        A block's score = sum of weights of uncovered ts it covers.
        This prioritizes covering rare/hard-to-cover configurations early.
        """
        if time_budget is None:
            time_budget = deadline - 5.0
        random.seed(seed)

        is_uncovered = np.full(num_ts, True, dtype=bool)
        is_selected = np.full(num_blocks, False, dtype=bool)
        gain = np.zeros(num_blocks, dtype=np.float64)

        # Initialize scarcity: weight of each ts = 1 / len(ts_to_blocks[ti])
        scarcity = np.zeros(num_ts, dtype=np.float64)
        for ti in range(num_ts):
            n_covers = len(ts_to_blocks_arr[ti])
            scarcity[ti] = 1.0 / max(n_covers, 1)

        # Initialize gain for each block
        for bi in range(num_blocks):
            gain[bi] = np.sum(scarcity[block_to_ts_arr[bi]])

        solution = []

        while True:
            if time.time() > time_budget:
                break

            max_gain = np.max(gain)
            if max_gain <= 1e-12:
                break

            # Find all blocks with max gain
            candidates = np.where((np.abs(gain - max_gain) < 1e-12) & (~is_selected))[0]
            if len(candidates) == 0:
                break

            # Pick one (randomly for diversity)
            bi = candidates[random.randint(0, len(candidates) - 1)]

            solution.append(bi)
            is_selected[bi] = True
            gain[bi] = -1.0

            # Update: mark covered ts, update scarcity, update gains
            for ti in block_to_ts_arr[bi]:
                if is_uncovered[ti]:
                    is_uncovered[ti] = False
                    # Update scarcity for this ts: it's now covered, but other blocks
                    # that also cover it lose this contribution
                    # Actually, scarcity changes when a block is selected:
                    # The remaining blocks that cover this ts now have one fewer "competitor"
                    # But simpler: just update gain directly
                    for nb in ts_to_blocks_arr[ti]:
                        if not is_selected[nb]:
                            gain[nb] -= scarcity[ti]

        return solution

    # ============ STANDARD GREEDY (for comparison / fallback) ============
    def standard_greedy(seed=0, time_budget=None):
        if time_budget is None:
            time_budget = deadline - 5.0
        random.seed(seed)

        is_uncovered = np.full(num_ts, True, dtype=bool)
        is_selected = np.full(num_blocks, False, dtype=bool)
        gain = np.full(num_blocks, len(block_to_ts_arr[0]), dtype=np.int32)

        # Actually gain should be the number of uncovered ts each block covers
        gain = np.zeros(num_blocks, dtype=np.int32)
        for bi in range(num_blocks):
            gain[bi] = len(block_to_ts_arr[bi])

        solution = []

        while True:
            if time.time() > time_budget:
                break

            max_gain = np.max(gain)
            if max_gain <= 0:
                break

            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break

            bi = candidates[random.randint(0, len(candidates) - 1)]

            solution.append(bi)
            is_selected[bi] = True
            gain[bi] = -1

            for ti in block_to_ts_arr[bi]:
                if is_uncovered[ti]:
                    is_uncovered[ti] = False
                    for nb in ts_to_blocks_arr[ti]:
                        if not is_selected[nb]:
                            gain[nb] -= 1

        return solution

    # ============ LOCAL SEARCH: REMOVE REDUNDANT BLOCKS ============
    def remove_redundant(solution, time_budget=None):
        if time_budget is None:
            time_budget = deadline - 2.0

        # Build cover count
        cover_count = np.zeros(num_ts, dtype=np.int32)
        for bi in solution:
            for ti in block_to_ts_arr[bi]:
                cover_count[ti] += 1

        changed = True
        while changed:
            changed = False
            for i in range(len(solution)):
                if time.time() > time_budget:
                    break
                bi = solution[i]
                # Check if any ts is uniquely covered by this block
                redundant = True
                for ti in block_to_ts_arr[bi]:
                    if cover_count[ti] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove it
                    for ti in block_to_ts_arr[bi]:
                        cover_count[ti] -= 1
                    solution.pop(i)
                    changed = True
                    break
        return solution

    # ============ LOCAL SEARCH: 1-FOR-1 SWAP ============
    def local_search_swap(solution, time_budget=None):
        if time_budget is None:
            time_budget = deadline - 2.0

        cover_count = np.zeros(num_ts, dtype=np.int32)
        for bi in solution:
            for ti in block_to_ts_arr[bi]:
                cover_count[ti] += 1

        sol_set = set(solution)
        improved = False

        for i in range(len(solution)):
            if time.time() > time_budget:
                break

            bi = solution[i]

            # Find ts uniquely covered by block i
            unique_ts = []
            for ti in block_to_ts_arr[bi]:
                if cover_count[ti] == 1:
                    unique_ts.append(ti)

            if not unique_ts:
                # This block is fully redundant - handled by remove_redundant
                continue

            # Find candidate blocks that cover ALL unique_ts
            # Start with blocks covering the first unique ts
            candidates = set(ts_to_blocks_arr[unique_ts[0]].tolist())
            for ti in unique_ts[1:]:
                candidates &= set(ts_to_blocks_arr[ti].tolist())
                if not candidates:
                    break

            if not candidates:
                continue

            candidates.discard(bi)
            valid = [c for c in candidates if c not in sol_set]

            if not valid:
                continue

            # Pick a candidate (prefer one that covers most additional ts)
            best_cand = None
            best_score = -1
            for c in valid[:20]:  # Limit candidates for speed
                score = 0
                for ti in block_to_ts_arr[c]:
                    if cover_count[ti] >= 1:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_cand = c

            if best_cand is None:
                continue

            # Perform swap
            for ti in block_to_ts_arr[bi]:
                cover_count[ti] -= 1
            for ti in block_to_ts_arr[best_cand]:
                cover_count[ti] += 1

            # Verify validity
            if np.all(cover_count >= 1):
                solution[i] = best_cand
                sol_set.discard(bi)
                sol_set.add(best_cand)
                improved = True
            else:
                # Rollback
                for ti in block_to_ts_arr[best_cand]:
                    cover_count[ti] -= 1
                for ti in block_to_ts_arr[bi]:
                    cover_count[ti] += 1

        return solution, improved

    # ============ RANDOMIZED SWAP SEARCH ============
    def randomized_swap_search(solution, num_iters=200, time_budget=None):
        if time_budget is None:
            time_budget = deadline - 2.0

        cover_count = np.zeros(num_ts, dtype=np.int32)
        for bi in solution:
            for ti in block_to_ts_arr[bi]:
                cover_count[ti] += 1

        sol_set = set(solution)
        improved = False

        for _ in range(num_iters):
            if time.time() > time_budget:
                break

            if not solution:
                break

            # Pick random block to remove
            i = random.randint(0, len(solution) - 1)
            bi_out = solution[i]

            # Temporarily remove
            for ti in block_to_ts_arr[bi_out]:
                cover_count[ti] -= 1

            # Check if still valid
            if np.all(cover_count >= 1):
                solution.pop(i)
                sol_set.discard(bi_out)
                improved = True
                return solution, improved

            # Find uncovered ts
            uncovered = np.where(cover_count == 0)[0]

            if len(uncovered) > 50:
                # Too many uncovered, rollback
                for ti in block_to_ts_arr[bi_out]:
                    cover_count[ti] += 1
                continue

            # Try to find a block that covers all uncovered ts
            if len(uncovered) == 0:
                continue

            target = uncovered[random.randint(0, len(uncovered) - 1)]
            candidates = ts_to_blocks_arr[target].tolist()
            random.shuffle(candidates)

            found = False
            for bi_in in candidates[:30]:
                if bi_in in sol_set or bi_in == bi_out:
                    continue

                # Check if covers all uncovered
                valid = True
                for ti in uncovered:
                    if (all_blocks[bi_in] & all_ts[ti]) != all_ts[ti]:
                        valid = False
                        break

                if valid:
                    # Perform swap
                    solution[i] = bi_in
                    sol_set.discard(bi_out)
                    sol_set.add(bi_in)

                    for ti in block_to_ts_arr[bi_out]:
                        cover_count[ti] += 1
                    for ti in block_to_ts_arr[bi_in]:
                        cover_count[ti] += 1

                    improved = True
                    found = True
                    break

            if not found:
                # Rollback
                for ti in block_to_ts_arr[bi_out]:
                    cover_count[ti] += 1

        return solution, improved

    # ============ MAIN SEARCH LOOP ============
    best_solution = None
    best_len = float('inf')

    # Phase 1: Try scarcity-weighted greedy with multiple seeds
    for seed in range(50):
        if time.time() > deadline - 30.0:
            break
        sol = scarcity_greedy(seed)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol[:]

    # Phase 2: Try standard greedy with multiple seeds
    for seed in range(50):
        if time.time() > deadline - 25.0:
            break
        sol = standard_greedy(seed)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol[:]

    if best_solution is None:
        # Fallback: just do standard greedy
        best_solution = standard_greedy(0)
        best_len = len(best_solution)

    # Phase 3: Local search on best solution
    if time.time() < deadline - 15.0:
        best_solution = remove_redundant(best_solution)
        best_len = len(best_solution)

    # Phase 4: Iterative swap local search
    for iteration in range(30):
        if time.time() > deadline - 5.0:
            break
        best_solution, improved = local_search_swap(best_solution)
        if not improved:
            break
        best_solution = remove_redundant(best_solution)
        if len(best_solution) < best_len:
            best_len = len(best_solution)

    # Phase 5: Randomized swap search
    for iteration in range(20):
        if time.time() > deadline - 3.0:
            break
        best_solution, improved = randomized_swap_search(best_solution, num_iters=100)
        if not improved:
            break
        best_solution = remove_redundant(best_solution)
        if len(best_solution) < best_len:
            best_len = len(best_solution)

    # Final cleanup
    best_solution = remove_redundant(best_solution)

    # Convert back to list of lists
    result = [block_lists[bi] for bi in best_solution]

    return result
