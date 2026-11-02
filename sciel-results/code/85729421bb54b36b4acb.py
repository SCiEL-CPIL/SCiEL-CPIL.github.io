import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin for 150s limit

    def C(n, r):
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        result = 1
        for i in range(1, r + 1):
            result = result * (n - i + 1) // i
        return result

    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)

    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)

    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Build block-to-ts mapping and ts-to-blocks mapping
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]

    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)

    block_to_ts_np = [np.array(lst, dtype=np.int32) for lst in block_to_ts]
    ts_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in ts_to_blocks]

    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}

    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    def run_greedy(seed, explore_prob=0.0):
        rng = random.Random(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []

        while True:
            if time.time() > deadline - 5.0:
                return None

            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break

            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break

            if explore_prob > 0 and rng.random() < explore_prob:
                threshold = max_gain - 3
                explore_cands = np.where((gain >= threshold) & (~is_selected))[0]
                if len(explore_cands) > 0:
                    b_idx = int(explore_cands[rng.randint(0, len(explore_cands) - 1)])
                else:
                    b_idx = int(candidates[rng.randint(0, len(candidates) - 1)])
            else:
                b_idx = int(candidates[rng.randint(0, len(candidates) - 1)])

            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1

            bts = block_to_ts_np[b_idx]
            newly_covered_mask = is_uncovered[bts]
            newly_covered = bts[newly_covered_mask]

            if len(newly_covered) == 0:
                continue

            is_uncovered[newly_covered] = False

            if len(newly_covered) > 0:
                all_blocks_to_decrement = []
                for ts_idx in newly_covered:
                    all_blocks_to_decrement.extend(ts_to_blocks[ts_idx])
                if all_blocks_to_decrement:
                    counts = np.bincount(all_blocks_to_decrement, minlength=num_k_blocks)
                    gain -= counts
                    gain[is_selected] = -1000

        return sol

    def local_search(blocks, max_iters=3000):
        n_blocks = len(blocks)
        if n_blocks == 0:
            return blocks, 0

        block_ts_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_ts_indices.append(block_to_ts_np[idx])
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(ts_index_map[ts])
                arr = np.array(ts_list, dtype=np.int32)
                block_ts_indices.append(arr)

        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for ts_arr in block_ts_indices:
            np.add.at(cover_count, ts_arr, 1)

        sol_block_set = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_set.add(idx)

        improved_count = 0

        # Phase A: Iteratively remove redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 3.0:
                    return blocks, improved_count
                ts_arr = block_ts_indices[i]
                if np.all(cover_count[ts_arr] >= 2):
                    np.subtract.at(cover_count, ts_arr, 1)
                    blocks.pop(i)
                    block_ts_indices.pop(i)
                    improved_count += 1
                    changed = True
                    break

        # Phase B: 1-exchange and 2-for-1 local search
        iters = 0
        while iters < max_iters:
            iters += 1
            if time.time() > deadline - 3.0:
                break
            if len(blocks) == 0:
                break

            # Try 1-exchange first
            i = random.randint(0, len(blocks) - 1)
            ts_arr_i = block_ts_indices[i]

            unique_mask = cover_count[ts_arr_i] == 1
            unique_ts = ts_arr_i[unique_mask]

            if len(unique_ts) == 0:
                np.subtract.at(cover_count, ts_arr_i, 1)
                blocks.pop(i)
                block_ts_indices.pop(i)
                improved_count += 1
                continue

            candidates = set(ts_to_blocks[int(unique_ts[0])])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(ts_to_blocks[int(ts_idx)])
                if not candidates:
                    break

            if not candidates:
                continue

            valid_candidates = [c for c in candidates if c not in sol_block_set]
            if not valid_candidates:
                continue

            new_block_idx = random.choice(valid_candidates)
            new_ts_arr = block_to_ts_np[new_block_idx]

            np.subtract.at(cover_count, ts_arr_i, 1)
            np.add.at(cover_count, new_ts_arr, 1)

            if np.min(cover_count) >= 1:
                blocks[i] = list(all_k_blocks[new_block_idx])
                block_ts_indices[i] = new_ts_arr
                sol_block_set.discard(get_block_idx(blocks[i]))
                sol_block_set.add(new_block_idx)
                improved_count += 1
            else:
                np.subtract.at(cover_count, new_ts_arr, 1)
                np.add.at(cover_count, ts_arr_i, 1)
                # If 1-exchange fails, try 2-for-1 exchange
                if len(blocks) >= 2:
                    j = random.randint(0, len(blocks) - 1)
                    if j == i:
                        j = (j + 1) % len(blocks)
                    
                    ts_arr_j = block_ts_indices[j]
                    combined_unique = np.concatenate([
                        ts_arr_i[cover_count[ts_arr_i] == 1],
                        ts_arr_j[cover_count[ts_arr_j] == 1]
                    ])
                    
                    if len(combined_unique) > 0:
                        cand_set = set(ts_to_blocks[int(combined_unique[0])])
                        for ts_idx in combined_unique[1:]:
                            cand_set.intersection_update(ts_to_blocks[int(ts_idx)])
                            if not cand_set:
                                break
                        
                        if cand_set:
                            valid_cands = [c for c in cand_set if c not in sol_block_set]
                            if valid_cands:
                                new_idx = random.choice(valid_cands)
                                new_ts_arr2 = block_to_ts_np[new_idx]
                                
                                test_cover = cover_count.copy()
                                np.subtract.at(test_cover, ts_arr_i, 1)
                                np.subtract.at(test_cover, ts_arr_j, 1)
                                np.add.at(test_cover, new_ts_arr2, 1)
                                
                                if np.min(test_cover) >= 1:
                                    np.subtract.at(cover_count, ts_arr_i, 1)
                                    np.subtract.at(cover_count, ts_arr_j, 1)
                                    np.add.at(cover_count, new_ts_arr2, 1)
                                    
                                    if j > i:
                                        blocks.pop(j)
                                        block_ts_indices.pop(j)
                                        i = i
                                    else:
                                        blocks.pop(i)
                                        block_ts_indices.pop(i)
                                        i = j - 1 if j > i else j
                                    
                                    blocks.insert(i, list(all_k_blocks[new_idx]))
                                    block_ts_indices.insert(i, new_ts_arr2)
                                    sol_block_set.discard(get_block_idx(blocks[i]))
                                    sol_block_set.add(new_idx)
                                    improved_count += 1

        return blocks, improved_count

    def run_ga(initial_blocks, pop_size=20, generations=100):
        """Genetic algorithm for covering design optimization."""
        if not initial_blocks:
            return initial_blocks
        
        # Create initial population
        population = [list(initial_blocks)]
        for _ in range(1 - pop_size):
            # Mutate initial solution
            mutated = list(initial_blocks)
            for _ in range(random.randint(1, 3)):
                if random.random() < 0.5 and len(mutated) > 0:
                    # Remove a block
                    idx = random.randint(0, len(mutated) - 1)
                    mutated.pop(idx)
                else:
                    # Add a random block
                    new_block = list(all_k_blocks[random.randint(0, num_k_blocks - 1)])
                    if new_block not in mutated:
                        mutated.append(new_block)
            population.append(mutated)
        
        best_solution = None
        best_len = float('inf')
        
        for gen in range(generations):
            if time.time() > deadline - 5.0:
                break
            
            # Evaluate population
            scored_population = []
            for blocks in population:
                # Check coverage
                cover_count = np.zeros(num_t_subsets, dtype=np.int32)
                valid = True
                for b in blocks:
                    idx = get_block_idx(b)
                    if idx >= 0:
                        np.add.at(cover_count, block_to_ts_np[idx], 1)
                    else:
                        for ts in itertools.combinations(b, t):
                            ts_idx = ts_index_map[ts]
                            cover_count[ts_idx] += 1
                
                if np.min(cover_count) >= 1:
                    scored_population.append((len(blocks), blocks))
                    if len(blocks) < best_len:
                        best_len = len(blocks)
                        best_solution = list(blocks)
                else:
                    # Invalid - penalize
                    scored_population.append((len(blocks) + 1000, blocks))
            
            # Sort by fitness
            scored_population.sort(key=lambda x: x[0])
            
            # Elitism
            new_population = [scored_population[0][1]]
            
            # Selection and crossover
            while len(new_population) < pop_size:
                # Tournament selection
                candidates = random.sample(scored_population, min(3, len(scored_population)))
                parent1 = min(candidates, key=lambda x: x[0])[1]
                candidates = random.sample(scored_population, min(3, len(scored_population)))
                parent2 = min(candidates, key=lambda x: x[0])[1]
                
                # Crossover: combine parents
                child = list(parent1)
                for b in parent2:
                    if b not in child and random.random() < 0.3:
                        child.append(b)
                
                # Mutation
                for _ in range(random.randint(0, 2)):
                    if random.random() < 0.5 and len(child) > 0:
                        idx = random.randint(0, len(child) - 1)
                        child.pop(idx)
                    else:
                        new_block = list(all_k_blocks[random.randint(0, num_k_blocks - 1)])
                        if new_block not in child:
                            child.append(new_block)
                
                new_population.append(child)
            
            population = new_population
        
        return best_solution if best_solution is not None else initial_blocks

    # Initial deterministic greedy
    best_solution = run_greedy(0)
    if best_solution is None:
        best_solution = [list(all_k_blocks[0])]
    best_len = len(best_solution)

    # Run multiple randomized greedy
    num_runs = 150
    for i in range(1, num_runs):
        if time.time() > deadline - 40.0:
            break

        explore = 0.0 if i < 50 else (0.1 if i < 100 else 0.2)
        sol = run_greedy(i, explore_prob=explore)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    # Phase 2: Local search on best solution
    best_solution, _ = local_search(best_solution, max_iters=5000)
    if len(best_solution) < best_len:
        best_len = len(best_solution)

    # Phase 3: Genetic algorithm
    if time.time() < deadline - 10.0:
        ga_result = run_ga(best_solution, pop_size=15, generations=50)
        if ga_result and len(ga_result) < best_len:
            best_len = len(ga_result)
            best_solution = ga_result

    # Final local search
    best_solution, _ = local_search(best_solution, max_iters=3000)
    if len(best_solution) < best_len:
        best_len = len(best_solution)

    return best_solution
