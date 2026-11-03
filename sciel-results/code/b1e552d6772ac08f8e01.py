"""Kissing number in dimension 11: Genetic algorithm with bitset chromosomes for large independent sets."""
import numpy as np
import itertools
import time
import random


def construct():
    start_time = time.time()
    deadline = start_time + 118.0

    D = 11

    # Generate candidate vectors
    candidates_list = []

    # Weight-4 vectors (norm^2=4): C(11,4)*16 = 5280
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    # Weight-3 vectors (norm^2=3): C(11,3)*8 = 1320
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    # Weight-2 vectors (norm^2=2): C(11,2)*4 = 220
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    # Weight-1 vectors (norm^2=1): 22
    for pos in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[pos] = s
            candidates_list.append(v)

    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)

    # Precompute norms squared
    norms_sq = np.sum(candidates.astype(np.int32) ** 2, axis=1)
    max_norm_sq = int(norms_sq.max())

    # Precompute dot product matrix
    dot = candidates.astype(np.int32) @ candidates.T.astype(np.int32)

    # Constraint: ||v_i - v_j||^2 >= max_norm_sq
    threshold = (norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - max_norm_sq) / 2.0
    compat = (dot <= threshold).astype(np.uint8)
    np.fill_diagonal(compat, 1)

    # conflict[i,j] = 1 if i and j CANNOT both be selected
    conflict = (1 - compat).astype(np.uint8)
    np.fill_diagonal(conflict, 0)

    # Precompute full degree
    full_deg = conflict.sum(axis=1)

    # Precompute neighbor lists for fast repair
    # neighbors[i] = list of indices j where conflict[i,j] = 1
    neighbors = [np.where(conflict[i] > 0)[0] for i in range(n)]

    def repair_set(sel_list, time_limit):
        """Remove vertices that cause conflicts, greedily removing the one with most conflicts."""
        sel = set(sel_list)
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            # Find all conflicts within sel
            conflict_count = {}
            sel_arr = list(sel)
            for i in range(len(sel_arr)):
                for j in range(i + 1, len(sel_arr)):
                    a, b = sel_arr[i], sel_arr[j]
                    if conflict[a, b]:
                        conflict_count[a] = conflict_count.get(a, 0) + 1
                        conflict_count[b] = conflict_count.get(b, 0) + 1
            if not conflict_count:
                break
            # Remove the vertex with highest conflict count
            worst = max(conflict_count, key=conflict_count.get)
            sel.discard(worst)
            changed = True
        return list(sel)

    def greedy_min_degree(remaining_init, time_limit):
        """Min-degree greedy: pick vertex with fewest conflicts among remaining."""
        remaining = remaining_init.copy()
        selected = []
        while time.time() < time_limit and remaining.any():
            deg = (conflict * remaining[np.newaxis, :]).sum(axis=1)
            deg[remaining == 0] = n + 1
            min_deg = deg.min()
            if min_deg >= n:
                break
            mask = (deg == min_deg) & (remaining > 0)
            picks = np.where(mask)[0]
            if len(picks) == 0:
                break
            new_idx = picks[0]
            selected.append(new_idx)
            remaining = remaining & (1 - conflict[new_idx])
            remaining[new_idx] = 0
        return selected

    def evaluate(sel_list):
        """Count size minus conflicts (penalty)."""
        if not sel_list:
            return 0
        sel_arr = np.array(sel_list)
        k = len(sel_arr)
        if k < 2:
            return k
        sub_conflict = 0.5 * (conflict[np.ix_(sel_arr, sel_arr)]).sum()
        return k - sub_conflict

    def get_conflicts(sel_list):
        """Return list of conflicting pairs."""
        sel_arr = np.array(sel_list)
        if len(sel_arr) < 2:
            return []
        sub = conflict[np.ix_(sel_arr, sel_arr)]
        idx = np.where(sub > 0)
        pairs = []
        for i, j in zip(idx[0], idx[1]):
            if i < j:
                pairs.append((sel_arr[i], sel_arr[j]))
        return pairs

    def genetic_algorithm(initial_population, time_limit, pop_size=60, max_generations=5000):
        """GA with bitset chromosomes, crossover, mutation, and repair."""
        # Ensure population is valid (independent sets)
        population = []
        for ind in initial_population:
            repaired = repair_set(ind, time_limit)
            population.append(repaired)

        # Track best
        best_ind = max(population, key=len)
        best_len = len(best_ind)

        for gen in range(max_generations):
            if time.time() > time_limit:
                break

            # Evaluate fitness
            fitness = [len(ind) for ind in population]
            total_fit = sum(fitness)
            if total_fit == 0:
                break

            # Selection: tournament
            new_population = []
            # Keep top 10%
            sorted_indices = np.argsort(fitness)[::-1]
            elite_count = max(2, pop_size // 10)
            for i in range(elite_count):
                new_population.append(population[sorted_indices[i]])

            # Fill rest with offspring
            while len(new_population) < pop_size and time.time() < time_limit:
                # Tournament selection
                def tournament_select():
                    idxs = np.random.choice(len(population), size=3, replace=False)
                    best_idx = idxs[np.argmax(fitness[idxs])]
                    return population[best_idx]

                parent1 = tournament_select()
                parent2 = tournament_select()

                # Crossover: uniform crossover with repair
                if np.random.random() < 0.7:
                    # Uniform crossover
                    mask = np.random.random(n) < 0.5
                    child = [i for i in range(n) if mask[i] and (i in parent1 or i in parent2)]
                    # Simplified: take union with random mask
                    set1 = set(parent1)
                    set2 = set(parent2)
                    union = set1 | set2
                    child = [i for i in union if np.random.random() < 0.5]
                else:
                    # Just copy parent with mutation
                    child = list(parent1)

                # Mutation
                if np.random.random() < 0.8:
                    # Add random vertex
                    if len(child) < n:
                        new_v = np.random.randint(n)
                        if new_v not in child:
                            child.append(new_v)
                if np.random.random() < 0.5:
                    # Remove random vertex
                    if child:
                        rem_idx = np.random.randint(len(child))
                        child.pop(rem_idx)

                # Repair
                child = repair_set(child, time_limit)
                new_population.append(child)

            population = new_population[:pop_size]

            # Update best
            for ind in population:
                if len(ind) > best_len:
                    best_len = len(ind)
                    best_ind = ind[:]

            # Stagnation check: if no improvement in last 200 gens, restart some individuals
            # (simplified: just continue)

        return best_ind

    # Main search
    np.random.seed(42)
    random.seed(42)

    # Phase 1: Generate initial diverse population
    initial_pop = []

    # Greedy min-degree on full set
    sel1 = greedy_min_degree(np.ones(n, dtype=np.uint8), deadline - 60.0)
    initial_pop.append(sel1)

    # Multiple randomized greedy restarts
    for trial in range(15):
        if time.time() > deadline - 50.0:
            break
        frac = np.random.uniform(0.4, 1.0)
        subset_size = int(n * frac)
        subset_indices = np.random.choice(n, size=subset_size, replace=False)
        remaining_init = np.zeros(n, dtype=np.uint8)
        remaining_init[subset_indices] = 1
        sel = greedy_min_degree(remaining_init, deadline - 50.0)
        initial_pop.append(sel)

    # Add some random independent sets
    for trial in range(5):
        if time.time() > deadline - 45.0:
            break
        # Random greedy
        remaining = np.ones(n, dtype=np.uint8)
        sel = []
        order = np.random.permutation(n)
        for idx in order:
            if time.time() > deadline - 45.0:
                break
            if remaining[idx]:
                # Check if compatible with all selected
                ok = True
                for s in sel:
                    if conflict[idx, s]:
                        ok = False
                        break
                if ok:
                    sel.append(idx)
                    remaining = remaining & (1 - conflict[idx])
                    remaining[idx] = 0
        initial_pop.append(sel)

    # Phase 2: Run genetic algorithm
    best_selected = max(initial_pop, key=len)
    if time.time() < deadline - 20.0:
        ga_result = genetic_algorithm(initial_pop, deadline - 20.0, pop_size=50, max_generations=3000)
        if len(ga_result) > len(best_selected):
            best_selected = ga_result

    # Phase 3: Local search refinement on best
    if best_selected and time.time() < deadline - 10.0:
        # Try to add more vectors
        sel_set = set(best_selected)
        compat_with_all = np.ones(n, dtype=np.uint8)
        for s in best_selected:
            compat_with_all = compat_with_all & compat[s]
        for s in best_selected:
            compat_with_all[s] = 0
        addable = np.where(compat_with_all > 0)[0]
        for idx in addable[np.argsort(full_deg[addable])]:
            if time.time() > deadline - 10.0:
                break
            best_selected.append(idx)
            sel_set.add(idx)
            compat_with_all = compat_with_all & compat[idx]
            compat_with_all[idx] = 0

    # Phase 4: One more GA run with the improved solution in population
    if best_selected and time.time() < deadline - 5.0:
        pop2 = [best_selected]
        for _ in range(10):
            if time.time() > deadline - 5.0:
                break
            # Perturb best
            perturbed = list(best_selected)
            for _ in range(random.randint(1, 5)):
                if perturbed and random.random() < 0.5:
                    perturbed.pop(random.randint(0, len(perturbed) - 1))
                if random.random() < 0.5:
                    perturbed.append(random.randint(n))
            perturbed = repair_set(perturbed, deadline - 5.0)
            pop2.append(perturbed)
        ga_result2 = genetic_algorithm(pop2, deadline - 5.0, pop_size=30, max_generations=1000)
        if len(ga_result2) > len(best_selected):
            best_selected = ga_result2

    # Final greedy fill
    if best_selected and time.time() < deadline - 2.0:
        sel_set = set(best_selected)
        compat_with_all = np.ones(n, dtype=np.uint8)
        for s in best_selected:
            compat_with_all = compat_with_all & compat[s]
        for s in best_selected:
            compat_with_all[s] = 0
        addable = np.where(compat_with_all > 0)[0]
        for idx in addable[np.argsort(full_deg[addable])]:
            if time.time() > deadline - 2.0:
                break
            best_selected.append(idx)
            sel_set.add(idx)
            compat_with_all = compat_with_all & compat[idx]
            compat_with_all[idx] = 0

    if not best_selected:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)

    result = candidates[np.array(best_selected, dtype=np.int64)]
    return result.astype(np.int64)
