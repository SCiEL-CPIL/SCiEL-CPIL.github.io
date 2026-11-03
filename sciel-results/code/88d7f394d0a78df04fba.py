import itertools
import random
import time
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def valid_cover(blocks):
        """Check if all t-subsets are covered."""
        covered = set()
        for block in blocks:
            for combo in itertools.combinations(block, t):
                covered.add(combo)
        return len(covered) == len(list(itertools.combinations(range(v), t)))

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    t_index = {s: i for i, s in enumerate(all_t_subsets)}

    # Precompute all k-blocks and their covered t-subsets
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(all_k_blocks)
    block_t_subsets = []
    for block in all_k_blocks:
        covered = set()
        for combo in itertools.combinations(block, t):
            covered.add(t_index[combo])
        block_t_subsets.append(covered)

    # Build reverse mapping
    t_to_blocks = [[] for _ in range(num_t)]
    for i, covered in enumerate(block_t_subsets):
        for idx in covered:
            t_to_blocks[idx].append(i)

    # Compute coverage counts for all blocks
    full_cover_count = np.zeros(num_t, dtype=np.int32)
    for i in range(num_k):
        for idx in block_t_subsets[i]:
            full_cover_count[idx] += 1

    # Weighted greedy initialization
    def greedy_init():
        safe_counts = np.maximum(full_cover_count, 1).astype(np.float64)
        t_weights = 1.0 / safe_counts

        gain = np.zeros(num_k, dtype=np.float64)
        for i in range(num_k):
            if block_t_subsets[i]:
                gain[i] = np.sum(t_weights[list(block_t_subsets[i])])

        uncovered = np.ones(num_t, dtype=np.bool_)
        uncovered_count = num_t
        selected = []
        selected_mask = np.zeros(num_k, dtype=np.bool_)

        while uncovered_count > 0 and elapsed() < deadline - 60:
            best_idx = int(np.argmax(gain))
            if gain[best_idx] <= 0:
                break

            selected.append(best_idx)
            selected_mask[best_idx] = True

            for idx in block_t_subsets[best_idx]:
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
                    w = t_weights[idx]
                    for b in t_to_blocks[idx]:
                        if not selected_mask[b]:
                            gain[b] -= w

        return selected

    # Remove redundant blocks
    def remove_redundant(selected):
        if not selected:
            return selected

        cover_count = np.zeros(num_t, dtype=np.int32)
        for i in selected:
            for idx in block_t_subsets[i]:
                cover_count[idx] += 1

        changed = True
        while changed and elapsed() < deadline - 20:
            changed = False
            new_selected = []
            for i in selected:
                needed = False
                for idx in block_t_subsets[i]:
                    if cover_count[idx] == 1:
                        needed = True
                        break
                if needed:
                    new_selected.append(i)
                else:
                    changed = True
                    for idx in block_t_subsets[i]:
                        cover_count[idx] -= 1
            if changed:
                selected = new_selected

        return selected

    # Greedy repair: add blocks to cover all t-subsets
    def greedy_repair(selected):
        cover_count = np.zeros(num_t, dtype=np.int32)
        for i in selected:
            for idx in block_t_subsets[i]:
                cover_count[idx] += 1

        uncovered = set(np.where(cover_count == 0)[0])
        while uncovered and elapsed() < deadline - 10:
            best_idx = -1
            best_gain = -1
            for i in range(num_k):
                if i in selected:
                    continue
                gain = len(block_t_subsets[i] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1 or best_gain == 0:
                break
            selected.append(best_idx)
            for idx in block_t_subsets[best_idx]:
                if idx in uncovered:
                    uncovered.discard(idx)

        return selected

    # Local search: try 1-for-1 swaps
    def local_search(selected, max_iters=5):
        selected_set = set(selected)
        cover_count = np.zeros(num_t, dtype=np.int32)
        for i in selected:
            for idx in block_t_subsets[i]:
                cover_count[idx] += 1

        for _ in range(max_iters):
            if elapsed() > deadline - 15:
                break
            improved = False
            for i in list(selected):
                if elapsed() > deadline - 15:
                    break
                unique_to_i = []
                for idx in block_t_subsets[i]:
                    if cover_count[idx] == 1:
                        unique_to_i.append(idx)

                if not unique_to_i:
                    continue

                best_swap = -1
                best_new_cover = 0
                for j in range(num_k):
                    if j in selected_set:
                        continue
                    covers_all = all(idx in block_t_subsets[j] for idx in unique_to_i)
                    if not covers_all:
                        continue
                    new_cover = sum(1 for idx in block_t_subsets[j] if cover_count[idx] == 0)
                    if new_cover > best_new_cover:
                        best_new_cover = new_cover
                        best_swap = j

                if best_swap >= 0 and best_new_cover > 0:
                    selected.remove(i)
                    selected.append(best_swap)
                    selected_set.remove(i)
                    selected_set.add(best_swap)

                    for idx in block_t_subsets[i]:
                        cover_count[idx] -= 1
                    for idx in block_t_subsets[best_swap]:
                        cover_count[idx] += 1
                    improved = True

            if not improved:
                break

        return selected

    # Genetic algorithm
    def genetic_algorithm(init_pop, pop_size=30, max_gen=20):
        population = [list(init_pop) for _ in range(pop_size)]
        # Add some random valid covers
        for _ in range(pop_size - 1):
            if elapsed() > deadline - 40:
                break
            rand_sel = greedy_init()
            rand_sel = remove_redundant(rand_sel)
            population.append(rand_sel)

        best = min(population, key=len)

        for gen in range(max_gen):
            if elapsed() > deadline - 30:
                break

            # Evaluate fitness (length, shorter is better)
            population.sort(key=len)
            best = population[0]

            # Selection: keep top half
            survivors = population[:pop_size // 2]

            # Crossover + repair
            new_population = list(survivors)
            while len(new_population) < pop_size:
                if elapsed() > deadline - 30:
                    break
                # Pick two parents
                p1 = random.choice(survivors)
                p2 = random.choice(survivors)

                # Crossover: randomly pick blocks from each parent
                child = []
                for block in p1:
                    if random.random() < 0.5:
                        child.append(block)
                for block in p2:
                    if random.random() < 0.5:
                        if block not in child:
                            child.append(block)

                # Repair
                child = greedy_repair(child)
                child = remove_redundant(child)
                new_population.append(child)

            # Mutation: remove a block, re-add minimum to restore validity
            for i in range(len(new_population)):
                if random.random() < 0.3 and elapsed() < deadline - 25:
                    sel = new_population[i]
                    if len(sel) > 5:
                        # Remove a random block
                        rem_idx = random.randint(0, len(sel) - 1)
                        mutated = sel[:rem_idx] + sel[rem_idx + 1:]
                        mutated = greedy_repair(mutated)
                        mutated = remove_redundant(mutated)
                        if len(mutated) < len(new_population[i]):
                            new_population[i] = mutated

            population = new_population

        return best

    # Main algorithm
    # Step 1: Greedy initialization
    init_selected = greedy_init()
    init_selected = remove_redundant(init_selected)

    # Step 2: Local search
    init_selected = local_search(init_selected, max_iters=3)

    # Step 3: Genetic algorithm if time permits
    if elapsed() < deadline - 50:
        ga_result = genetic_algorithm(init_selected, pop_size=25, max_gen=15)
        if len(ga_result) < len(init_selected):
            init_selected = ga_result

    # Final local search
    init_selected = local_search(init_selected, max_iters=2)
    init_selected = remove_redundant(init_selected)

    result = [list(all_k_blocks[i]) for i in init_selected]

    # Safety: ensure validity
    if elapsed() < deadline - 5:
        cover_count_final = np.zeros(num_t, dtype=np.int32)
        for i in init_selected:
            for idx in block_t_subsets[i]:
                cover_count_final[idx] += 1
        if not np.all(cover_count_final > 0):
            uncovered_final = set(np.where(cover_count_final == 0)[0])
            while uncovered_final and elapsed() < deadline - 2:
                best_idx = -1
                best_gain = -1
                for i in range(num_k):
                    if i in init_selected:
                        continue
                    gain = len(block_t_subsets[i] & uncovered_final)
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                if best_idx == -1 or best_gain == 0:
                    break
                init_selected.append(best_idx)
                for idx in block_t_subsets[best_idx]:
                    if idx in uncovered_final:
                        uncovered_final.discard(idx)
            result = [list(all_k_blocks[i]) for i in init_selected]

    return result
