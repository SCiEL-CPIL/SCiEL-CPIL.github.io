import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 145.0  # Safety margin for 150s limit

    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)

    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}

    # Precompute for each block the list of t-subset indices it covers
    block_tsubs = []
    for b in all_blocks:
        subs = [t_idx[s] for s in itertools.combinations(b, t)]
        block_tsubs.append(subs)

    # For each t-subset, which blocks cover it
    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_blocks[si].append(i)

    # Precompute sets for fast subset checks
    block_tsubs_sets = [set(subs) for subs in block_tsubs]

    def run_greedy(seed):
        random.seed(seed)
        block_gain = np.array([len(subs) for subs in block_tsubs], dtype=np.int32)
        is_covered = np.zeros(N_t, dtype=np.int8)
        uncovered_count = N_t
        selected_indices = []

        while uncovered_count > 0 and time.time() < DEADLINE:
            max_gain = np.max(block_gain)
            if max_gain <= 0:
                break
            candidates = np.where(block_gain == max_gain)[0]
            best_block_idx = int(random.choice(candidates.tolist()))

            selected_indices.append(best_block_idx)

            subs = block_tsubs[best_block_idx]
            for si in subs:
                if is_covered[si] == 0:
                    is_covered[si] = 1
                    uncovered_count -= 1
                    block_gain[tsub_blocks[si]] -= 1

        return selected_indices

    def reduce_solution_fast(selected_indices):
        if not selected_indices:
            return []

        n_sel = len(selected_indices)
        local_of_global = np.full(N_blocks, -1, dtype=np.int32)
        for i, g in enumerate(selected_indices):
            local_of_global[g] = i

        total_count = np.zeros(N_t, dtype=np.int32)
        crit_count = np.zeros(n_sel, dtype=np.int32)

        for i in range(n_sel):
            g = selected_indices[i]
            for si in block_tsubs[g]:
                total_count[si] += 1

        for i in range(n_sel):
            g = selected_indices[i]
            for si in block_tsubs[g]:
                if total_count[si] == 1:
                    crit_count[i] += 1

        queue = [i for i in range(n_sel) if crit_count[i] == 0]
        removed = np.zeros(n_sel, dtype=np.bool_)

        while queue:
            i = queue.pop()
            if removed[i] or crit_count[i] > 0:
                continue

            removed[i] = True
            g = selected_indices[i]

            for si in block_tsubs[g]:
                old_c = total_count[si]
                total_count[si] -= 1
                new_c = total_count[si]

                if old_c == 2 and new_c == 1:
                    for g_other in tsub_blocks[si]:
                        if g_other == g:
                            continue
                        i_other = local_of_global[g_other]
                        if i_other >= 0 and not removed[i_other]:
                            crit_count[i_other] += 1

        return [selected_indices[i] for i in range(n_sel) if not removed[i]]

    def local_search_1for1(selected_indices, max_rounds=10):
        """1-for-1 swap local search with optimized candidate filtering."""
        if not selected_indices:
            return selected_indices

        in_sol_set = set(selected_indices)
        total_count = np.zeros(N_t, dtype=np.int32)

        for g in selected_indices:
            for si in block_tsubs[g]:
                total_count[si] += 1

        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < DEADLINE - 3.0:
            improved = False
            rounds += 1
            n_sel = len(selected_indices)
            for i in range(n_sel):
                if time.time() > DEADLINE - 3.0:
                    return selected_indices
                g = selected_indices[i]
                # Find t-subsets uniquely covered by this block
                unique_si = [si for si in block_tsubs[g] if total_count[si] == 1]
                if not unique_si:
                    continue

                # Tentatively remove g
                for si in block_tsubs[g]:
                    total_count[si] -= 1

                # Find blocks that cover all unique_si
                best_gain = 0
                best_g = -1
                unique_set = set(unique_si)
                for g_new in range(N_blocks):
                    if g_new in in_sol_set:
                        continue
                    if not unique_set.issubset(block_tsubs_sets[g_new]):
                        continue
                    gain = sum(1 for si in block_tsubs[g_new] if total_count[si] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_g = g_new

                if best_g >= 0:
                    in_sol_set.discard(g)
                    in_sol_set.add(best_g)
                    for si in block_tsubs[best_g]:
                        total_count[si] += 1
                    selected_indices[i] = best_g
                    improved = True
                else:
                    for si in block_tsubs[g]:
                        total_count[si] += 1

        return selected_indices

    def local_search_2for2(selected_indices, max_rounds=5):
        """Try removing 2 blocks and adding 2 different blocks (net zero or negative)."""
        if not selected_indices or len(selected_indices) < 3:
            return selected_indices

        in_sol_set = set(selected_indices)
        total_count = np.zeros(N_t, dtype=np.int32)
        for g in selected_indices:
            for si in block_tsubs[g]:
                total_count[si] += 1

        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < DEADLINE - 3.0:
            improved = False
            rounds += 1
            n_sel = len(selected_indices)
            # Try pairs of blocks
            for i in range(n_sel):
                if time.time() > DEADLINE - 3.0:
                    return selected_indices
                for j in range(i + 1, n_sel):
                    g1 = selected_indices[i]
                    g2 = selected_indices[j]
                    # Find t-subsets uniquely covered by g1 or g2 (not by any other selected block)
                    unique_si = set()
                    for si in block_tsubs[g1]:
                        if total_count[si] == 1:
                            unique_si.add(si)
                    for si in block_tsubs[g2]:
                        if total_count[si] == 1:
                            unique_si.add(si)
                    if not unique_si:
                        continue

                    # Tentatively remove both
                    for si in block_tsubs[g1]:
                        total_count[si] -= 1
                    for si in block_tsubs[g2]:
                        total_count[si] -= 1

                    # Find 2 blocks that cover all unique_si together
                    # Strategy: find best single block, then best complement
                    best_pair_gain = -1
                    best_pair = (-1, -1)
                    candidates = []
                    for g_new in range(N_blocks):
                        if g_new in in_sol_set:
                            continue
                        if not unique_si.issubset(block_tsubs_sets[g_new]):
                            # Check how many unique_si it covers
                            covered = sum(1 for si in unique_si if si in block_tsubs_sets[g_new])
                            if covered >= len(unique_si) * 0.7:
                                candidates.append(g_new)
                    
                    if len(candidates) < 2:
                        for si in block_tsubs[g1]:
                            total_count[si] += 1
                        for si in block_tsubs[g2]:
                            total_count[si] += 1
                        continue

                    # Limit candidates for speed
                    if len(candidates) > 50:
                        candidates = candidates[:50]

                    for a_idx in range(len(candidates)):
                        for b_idx in range(a_idx + 1, len(candidates)):
                            ga = candidates[a_idx]
                            gb = candidates[b_idx]
                            # Check combined coverage of unique_si
                            combined = block_tsubs_sets[ga] | block_tsubs_sets[gb]
                            if not unique_si.issubset(combined):
                                continue
                            gain = 0
                            for si in block_tsubs[ga]:
                                if total_count[si] == 0:
                                    gain += 1
                            for si in block_tsubs[gb]:
                                if total_count[si] == 0:
                                    gain += 1
                            if gain > best_pair_gain:
                                best_pair_gain = gain
                                best_pair = (ga, gb)

                    if best_pair[0] >= 0:
                        ga, gb = best_pair
                        in_sol_set.discard(g1)
                        in_sol_set.discard(g2)
                        in_sol_set.add(ga)
                        in_sol_set.add(gb)
                        for si in block_tsubs[ga]:
                            total_count[si] += 1
                        for si in block_tsubs[gb]:
                            total_count[si] += 1
                        selected_indices[i] = ga
                        selected_indices[j] = gb
                        improved = True
                    else:
                        for si in block_tsubs[g1]:
                            total_count[si] += 1
                        for si in block_tsubs[g2]:
                            total_count[si] += 1

        return selected_indices

    def branch_and_bound_refine(selected_indices, time_budget=10.0):
        """
        Use a limited branch-and-bound to try to reduce the solution further.
        Start from the current solution, try to remove one block at a time
        and repair if needed, using B&B for the repair step.
        """
        if not selected_indices:
            return selected_indices

        sol = list(selected_indices)
        best = len(sol)
        start_bb = time.time()

        total_count = np.zeros(N_t, dtype=np.int32)
        for g in sol:
            for si in block_tsubs[g]:
                total_count[si] += 1

        n_sel = len(sol)
        for i in range(n_sel):
            if time.time() - start_bb > time_budget:
                break
            g_remove = sol[i]
            # Tentatively remove
            for si in block_tsubs[g_remove]:
                total_count[si] -= 1

            # Find uncovered t-subsets
            uncovered = [si for si in block_tsubs[g_remove] if total_count[si] == 0]
            if not uncovered:
                # Can remove freely
                sol.pop(i)
                n_sel -= 1
                continue

            # Try to find 1 block that covers all uncovered
            unique_set = set(uncovered)
            replacement = -1
            for g_new in range(N_blocks):
                if g_new in sol:
                    continue
                if unique_set.issubset(block_tsubs_sets[g_new]):
                    # Check net gain
                    gain = sum(1 for si in block_tsubs[g_new] if total_count[si] == 0)
                    if gain > 0:
                        replacement = g_new
                        break

            if replacement >= 0:
                sol[i] = replacement
                for si in block_tsubs[replacement]:
                    total_count[si] += 1
            else:
                # Revert
                for si in block_tsubs[g_remove]:
                    total_count[si] += 1

        return sol

    # First run: deterministic greedy
    sol_idx = run_greedy(0)
    sol_idx = reduce_solution_fast(sol_idx)
    best_sol_idx = list(sol_idx)
    best_len = len(best_sol_idx)

    # Randomized restarts with local search
    while time.time() < DEADLINE - 20.0:
        seed = random.randint(0, 2**31)
        cand_idx = run_greedy(seed)
        cand_idx = reduce_solution_fast(cand_idx)

        if len(cand_idx) < best_len:
            best_len = len(cand_idx)
            best_sol_idx = list(cand_idx)

        # Apply local search to every candidate (not just new bests) for more exploration
        if time.time() < DEADLINE - 25.0:
            improved = local_search_1for1(list(cand_idx), max_rounds=5)
            improved = reduce_solution_fast(improved)
            if len(improved) < best_len:
                best_len = len(improved)
                best_sol_idx = list(improved)

    # Final intensive local search on the best solution
    if time.time() < DEADLINE - 10.0:
        improved = local_search_1for1(list(best_sol_idx), max_rounds=15)
        improved = reduce_solution_fast(improved)
        if len(improved) < best_len:
            best_len = len(improved)
            best_sol_idx = list(improved)

    # Try 2-for-2 swaps
    if time.time() < DEADLINE - 8.0:
        improved = local_search_2for2(list(best_sol_idx), max_rounds=3)
        improved = reduce_solution_fast(improved)
        if len(improved) < best_len:
            best_len = len(improved)
            best_sol_idx = list(improved)

    # Branch-and-bound refinement
    if time.time() < DEADLINE - 5.0:
        bb_time = min(5.0, max(1.0, DEADLINE - 5.0 - time.time()))
        improved = branch_and_bound_refine(list(best_sol_idx), time_budget=bb_time)
        improved = reduce_solution_fast(improved)
        if len(improved) < best_len:
            best_len = len(improved)
            best_sol_idx = list(improved)

    # One more round of 1-for-1 after B&B
    if time.time() < DEADLINE - 2.0:
        improved = local_search_1for1(list(best_sol_idx), max_rounds=3)
        improved = reduce_solution_fast(improved)
        if len(improved) < best_len:
            best_len = len(improved)
            best_sol_idx = list(improved)

    return [all_blocks[i] for i in best_sol_idx]
