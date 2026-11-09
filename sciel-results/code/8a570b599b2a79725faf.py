import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 118.0  # Safety margin for 150s limit

    points = list(range(v))

    # Precomputation
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)

    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}

    block_tsubs = []
    for b in all_blocks:
        subs = [t_idx[s] for s in itertools.combinations(b, t)]
        block_tsubs.append(subs)

    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_blocks[si].append(i)

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

    def local_search(selected_indices, max_iters=50):
        """1-for-1 swap local search with set-based membership for speed."""
        if not selected_indices:
            return selected_indices

        in_sol_set = set(selected_indices)
        total_count = np.zeros(N_t, dtype=np.int32)

        for g in selected_indices:
            for si in block_tsubs[g]:
                total_count[si] += 1

        # Precompute set of t-subset indices for each block for fast lookup
        block_tsubs_sets = [set(subs) for subs in block_tsubs]

        improved = True
        while improved and time.time() < DEADLINE - 2.0:
            improved = False
            n_sel = len(selected_indices)
            for i in range(n_sel):
                if time.time() > DEADLINE - 2.0:
                    return selected_indices
                g = selected_indices[i]
                # Find t-subsets uniquely covered by this block
                unique_si = []
                for si in block_tsubs[g]:
                    if total_count[si] == 1:
                        unique_si.append(si)
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
                    # Check if g_new covers all unique_si using set intersection
                    if not unique_set.issubset(block_tsubs_sets[g_new]):
                        continue
                    # Count how many previously-uncovered t-subsets it now covers
                    gain = 0
                    for si in block_tsubs[g_new]:
                        if total_count[si] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_g = g_new

                if best_g >= 0:
                    # Apply swap: remove g, add best_g
                    in_sol_set.discard(g)
                    in_sol_set.add(best_g)
                    for si in block_tsubs[best_g]:
                        total_count[si] += 1
                    selected_indices[i] = best_g
                    improved = True
                else:
                    # Revert
                    for si in block_tsubs[g]:
                        total_count[si] += 1

        return selected_indices

    def genetic_improve(selected_indices, pop_size=6, generations=15):
        """Genetic algorithm to improve the covering with better exploration."""
        if not selected_indices or len(selected_indices) < 2:
            return selected_indices

        def fitness(sol):
            return len(sol)

        def crossover(p1, p2):
            """Weighted crossover: prefer blocks from shorter parent."""
            sol_set = set()
            # Start with all blocks from shorter parent
            if len(p1) <= len(p2):
                shorter, longer = p1, p2
            else:
                shorter, longer = p2, p1

            for b in shorter:
                sol_set.add(b)
            # Add some from longer with lower probability
            for b in longer:
                if random.random() < 0.25:
                    sol_set.add(b)
            return list(sol_set)

        def mutate(sol):
            """Swap a block with a random other block, or add/remove."""
            if not sol:
                return sol
            op = random.random()
            if op < 0.6:
                # Swap
                i = random.randint(0, len(sol) - 1)
                g = sol[i]
                g_new = random.randint(0, N_blocks - 1)
                if g_new != g:
                    sol[i] = g_new
            elif op < 0.8:
                # Add a random block
                g_new = random.randint(0, N_blocks - 1)
                if g_new not in sol:
                    sol.append(g_new)
            else:
                # Remove a random block (if we have more than minimum)
                if len(sol) > 1:
                    i = random.randint(0, len(sol) - 1)
                    sol.pop(i)
            return sol

        def is_valid_cover(sol):
            """Check if sol covers all t-subsets."""
            covered = np.zeros(N_t, dtype=np.int8)
            for g in sol:
                for si in block_tsubs[g]:
                    covered[si] = 1
            return np.all(covered == 1)

        population = [list(selected_indices)]
        # Add diverse variants
        for _ in range(pop_size - 1):
            variant = list(selected_indices)
            r = random.random()
            if r < 0.4:
                variant = mutate(variant)
            elif r < 0.7:
                variant = mutate(variant)
                variant = mutate(variant)
            else:
                # More aggressive: swap 2-3 blocks
                for _ in range(random.randint(2, 3)):
                    if variant:
                        variant = mutate(variant)
            population.append(variant)

        best = min(population, key=fitness)
        best_len = fitness(best)

        for gen in range(generations):
            if time.time() > DEADLINE - 3.0:
                break

            next_pop = [list(best)]  # Elitism

            while len(next_pop) < pop_size:
                # Tournament selection
                p1_idx = random.randint(0, len(population) - 1)
                p2_idx = random.randint(0, len(population) - 1)
                p1 = population[p1_idx]
                p2 = population[p2_idx]

                child = crossover(p1, p2)
                if random.random() < 0.4:
                    child = mutate(child)

                # Reduce
                child = reduce_solution_fast(child)

                # Check validity
                if is_valid_cover(child):
                    next_pop.append(child)
                else:
                    # Repair: add blocks from best to cover gaps
                    covered = np.zeros(N_t, dtype=np.int8)
                    for g in child:
                        for si in block_tsubs[g]:
                            covered[si] = 1
                    for g in best:
                        if g not in child:
                            for si in block_tsubs[g]:
                                if covered[si] == 0:
                                    child.append(g)
                                    for si2 in block_tsubs[g]:
                                        covered[si2] = 1
                                    break
                    child = reduce_solution_fast(child)
                    if is_valid_cover(child):
                        next_pop.append(child)

            population = next_pop
            new_best = min(population, key=fitness)
            if fitness(new_best) < best_len:
                best = new_best
                best_len = fitness(best)

        return best

    # First run: deterministic greedy
    sol_idx = run_greedy(0)
    sol_idx = reduce_solution_fast(sol_idx)
    best_sol_idx = list(sol_idx)
    best_len = len(best_sol_idx)

    # Randomized restarts
    while time.time() < DEADLINE - 12.0:
        seed = random.randint(0, 2**31)
        cand_idx = run_greedy(seed)
        cand_idx = reduce_solution_fast(cand_idx)

        if len(cand_idx) < best_len:
            best_len = len(cand_idx)
            best_sol_idx = list(cand_idx)
            # Try local search on the new best
            if time.time() < DEADLINE - 15.0:
                improved = local_search(list(cand_idx))
                improved = reduce_solution_fast(improved)
                if len(improved) < best_len:
                    best_len = len(improved)
                    best_sol_idx = list(improved)

    # Final local search pass on the best solution
    if time.time() < DEADLINE - 5.0:
        improved = local_search(list(best_sol_idx))
        improved = reduce_solution_fast(improved)
        if len(improved) < best_len:
            best_len = len(improved)
            best_sol_idx = list(improved)

    # Genetic improvement if time permits
    if time.time() < DEADLINE - 10.0:
        ga_result = genetic_improve(list(best_sol_idx), pop_size=5, generations=12)
        ga_result = reduce_solution_fast(ga_result)
        ga_result = local_search(ga_result)
        ga_result = reduce_solution_fast(ga_result)
        if len(ga_result) < best_len:
            best_len = len(ga_result)
            best_sol_idx = list(ga_result)

    return [all_blocks[i] for i in best_sol_idx]
