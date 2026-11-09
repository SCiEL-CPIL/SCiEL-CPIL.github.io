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

    # Build incidence matrix: block x t-subset, 1 if t-subset is in block
    # For memory efficiency, use sparse representation
    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_blocks[si].append(i)

    # Precompute block gain (number of t-subsets covered)
    block_tsub_counts = np.array([len(subs) for subs in block_tsubs], dtype=np.int32)

    def run_greedy(seed):
        random.seed(seed)
        block_gain = block_tsub_counts.copy()
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
                    # Update gains for all blocks covering this t-subset
                    for b_idx in tsub_blocks[si]:
                        block_gain[b_idx] -= 1

        return selected_indices

    def reduce_solution_fast(selected_indices):
        if not selected_indices:
            return []

        n_sel = len(selected_indices)
        # Use a set for O(1) lookup
        selected_set = set(selected_indices)

        total_count = np.zeros(N_t, dtype=np.int32)
        crit_count = np.zeros(n_sel, dtype=np.int32)

        # Count how many times each t-subset is covered
        for g in selected_indices:
            for si in block_tsubs[g]:
                total_count[si] += 1

        # Count critical t-subsets for each block (t-subsets covered only by this block)
        for i, g in enumerate(selected_indices):
            for si in block_tsubs[g]:
                if total_count[si] == 1:
                    crit_count[i] += 1

        queue = [i for i in range(n_sel) if crit_count[i] == 0]
        removed = [False] * n_sel

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
                    # This t-subset becomes critical for remaining blocks
                    for g_other in tsub_blocks[si]:
                        if g_other == g:
                            continue
                        if g_other in selected_set:
                            # Find index of g_other in selected_indices
                            # This is O(n_sel) but we can optimize
                            for j in range(n_sel):
                                if selected_indices[j] == g_other and not removed[j]:
                                    crit_count[j] += 1
                                    break

        return [selected_indices[i] for i in range(n_sel) if not removed[i]]

    def local_search(selected_indices, max_iters=50):
        """Try to remove a block and add fewer blocks to re-cover uncovered t-subsets."""
        if not selected_indices:
            return selected_indices

        n_sel = len(selected_indices)
        selected_set = set(selected_indices)
        total_count = np.zeros(N_t, dtype=np.int32)

        for g in selected_indices:
            for si in block_tsubs[g]:
                total_count[si] += 1

        improved = True
        iter_count = 0
        while improved and iter_count < max_iters and time.time() < DEADLINE:
            improved = False
            iter_count += 1
            # Try removing each block
            for i in range(n_sel):
                if time.time() > DEADLINE:
                    break
                g = selected_indices[i]
                # Check if any t-subset would become uncovered
                would_uncover = False
                for si in block_tsubs[g]:
                    if total_count[si] <= 1:
                        would_uncover = True
                        break
                if would_uncover:
                    continue

                # Tentatively remove
                for si in block_tsubs[g]:
                    total_count[si] -= 1

                # Find uncovered t-subsets
                uncovered = [si for si in block_tsubs[g] if total_count[si] == 0]

                if not uncovered:
                    # Safe to remove
                    selected_indices.pop(i)
                    selected_set.discard(g)
                    n_sel -= 1
                    improved = True
                    break

                # Try to cover uncovered with 1 new block
                found_one = False
                for g_new in all_blocks:
                    if g_new in selected_set:
                        continue
                    covers_all = all(si in block_tsubs[g_new] for si in uncovered)
                    if covers_all:
                        found_one = True
                        # Add this block
                        for si in block_tsubs[g_new]:
                            total_count[si] += 1
                        selected_indices.append(g_new)
                        selected_set.add(g_new)
                        n_sel += 1
                        selected_indices.pop(i)
                        selected_set.discard(g)
                        n_sel -= 1
                        improved = True
                        break

                if not found_one:
                    # Revert
                    for si in block_tsubs[g]:
                        total_count[si] += 1
                    continue

        return selected_indices

    def genetic_improve(selected_indices, pop_size=4, generations=10):
        """Genetic algorithm to improve the covering."""
        if not selected_indices or len(selected_indices) < 2:
            return selected_indices

        def fitness(sol):
            return len(sol)

        def crossover(p1, p2):
            """Uniform crossover with repair."""
            if len(p1) <= len(p2):
                shorter, longer = p1, p2
            else:
                shorter, longer = p2, p1

            new_sol = set()
            for b in shorter:
                if random.random() < 0.5:
                    new_sol.add(b)
            for b in longer:
                if random.random() < 0.3:
                    new_sol.add(b)
            return list(new_sol)

        def mutate(sol):
            """Swap a block with a random other block."""
            if not sol:
                return sol
            i = random.randint(0, len(sol) - 1)
            g = sol[i]
            for _ in range(20):
                g_new = random.randint(0, N_blocks - 1)
                if g_new != g:
                    sol[i] = g_new
                    return sol
            return sol

        def is_valid_cover(sol):
            """Check if sol covers all t-subsets."""
            covered = np.zeros(N_t, dtype=np.int8)
            for g in sol:
                for si in block_tsubs[g]:
                    covered[si] = 1
            return np.all(covered == 1)

        population = [list(selected_indices)]
        for _ in range(pop_size - 1):
            variant = list(selected_indices)
            if random.random() < 0.5 and variant:
                variant = mutate(variant)
            population.append(variant)

        best = min(population, key=fitness)

        for gen in range(generations):
            if time.time() > DEADLINE:
                break

            next_pop = [list(best)]  # Elitism

            while len(next_pop) < pop_size:
                p1_idx = random.randint(0, len(population) - 1)
                p2_idx = random.randint(0, len(population) - 1)
                p1 = population[p1_idx]
                p2 = population[p2_idx]

                child = crossover(p1, p2)
                if random.random() < 0.3:
                    child = mutate(child)

                child = reduce_solution_fast(child)

                if is_valid_cover(child):
                    next_pop.append(child)
                else:
                    # Repair
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
            if fitness(new_best) < fitness(best):
                best = new_best

        return best

    # First run: deterministic greedy
    sol_idx = run_greedy(0)
    sol_idx = reduce_solution_fast(sol_idx)
    best_sol = [all_blocks[i] for i in sol_idx]
    best_len = len(best_sol)

    # Randomized restarts with genetic improvement on best candidates
    while time.time() < DEADLINE - 10.0:
        seed = random.randint(0, 2**31)
        cand_idx = run_greedy(seed)
        cand_idx = reduce_solution_fast(cand_idx)

        if len(cand_idx) < best_len:
            best_len = len(cand_idx)
            best_sol = [all_blocks[i] for i in cand_idx]
            if time.time() < DEADLINE - 15.0:
                improved = genetic_improve(list(cand_idx), pop_size=3, generations=5)
                improved = reduce_solution_fast(improved)
                if len(improved) < best_len:
                    best_len = len(improved)
                    best_sol = [all_blocks[i] for i in improved]

    # Final local search pass on the best solution
    if time.time() < DEADLINE - 5.0:
        final_idx = [all_blocks.index(b) for b in best_sol]
        improved = local_search(list(final_idx), max_iters=30)
        improved = reduce_solution_fast(improved)
        if len(improved) < best_len:
            best_len = len(improved)
            best_sol = [all_blocks[i] for i in improved]

    return best_sol
