import numpy as np
import itertools
import time
import random
import sys

def construct(v, k, t):
    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute block_to_subsets and subset_to_blocks
    block_to_subsets = [[] for _ in range(num_blocks)]
    subset_to_blocks = [[] for _ in range(num_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        for s in itertools.combinations(block, t):
            s_idx = subset_to_idx[s]
            block_to_subsets[b_idx].append(s_idx)
            subset_to_blocks[s_idx].append(b_idx)
    
    # Precompute numpy arrays for fast operations
    block_subset_arrays = [np.array(subs, dtype=np.int32) for subs in block_to_subsets]
    subset_block_arrays = [np.array(blocks, dtype=np.int32) for blocks in subset_to_blocks]
    
    # Precompute subset_to_blocks as sets for fast membership and intersection in swap
    subset_to_blocks_sets = [set(blocks.tolist()) for blocks in subset_block_arrays]
    
    # Precompute block sizes (number of t-subsets per block) for tie-breaking
    block_sizes = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)
    
    # Precompute for GA: for each block, the set of t-subsets it covers (as frozenset for fast ops)
    block_subsets_frozen = [frozenset(subs) for subs in block_to_subsets]
    
    def run_greedy_full(random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered = np.ones(num_subsets, dtype=bool)
        gains = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)
        
        # Maintain count of remaining blocks covering each subset
        remaining_count = np.array([len(blocks) for blocks in subset_block_arrays], dtype=np.int32)
        
        selected_blocks = []
        used_blocks = np.zeros(num_blocks, dtype=bool)
        
        while True:
            if not np.any(uncovered):
                break
                
            # Find top candidates based on simple gain first
            noise = rng.uniform(0, 0.5, size=num_blocks)
            scores = gains.astype(np.float64) + noise
            scores[used_blocks] = -1
            top_indices = np.argpartition(scores, -200)[-200:]
            
            best_idx = -1
            best_score = -1.0
            
            # For the top candidates, compute a more sophisticated score
            # Score = Gain + Sum(1/remaining_count[s] for s in uncovered & block)
            for idx in top_indices:
                if gains[idx] == 0:
                    continue
                b_subs = block_subset_arrays[idx]
                # Filter to uncovered subsets
                unc_subs = b_subs[uncovered[b_subs]]
                if len(unc_subs) == 0:
                    continue
                
                # Calculate criticality bonus
                crit_bonus = np.sum(1.0 / remaining_count[unc_subs].astype(np.float64))
                final_score = float(gains[idx]) + 0.5 * crit_bonus + rng.uniform(0, 0.1)
                
                if final_score > best_score:
                    best_score = final_score
                    best_idx = idx
            
            if best_idx == -1 or gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            used_blocks[best_idx] = True
            
            b_subs = block_subset_arrays[best_idx]
            newly_covered = b_subs[uncovered[b_subs]]
            
            if len(newly_covered) == 0:
                continue
            
            uncovered[newly_covered] = False
            
            # Update remaining counts
            remaining_count[b_subs] -= 1
            
            # Update gains for other blocks
            for s in newly_covered:
                for b in subset_block_arrays[s]:
                    gains[b] -= 1
                    
        return selected_blocks
    
    def compute_coverage_counts(blocks_indices):
        """Compute coverage count array for a list of block indices."""
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in blocks_indices:
            for s_idx in block_subset_arrays[b_idx]:
                cover_count[s_idx] += 1
        return cover_count
    
    def run_greedy_partial(uncovered_mask, random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            return []
            
        gains = np.zeros(num_blocks, dtype=np.int32)
        for s_idx in uncovered_indices:
            for b_idx in subset_block_arrays[s_idx]:
                gains[b_idx] += 1
                
        selected_blocks = []
        current_uncovered = uncovered_mask.copy()
        
        while True:
            if not np.any(current_uncovered):
                break
                
            noise = rng.uniform(0, 0.3, size=num_blocks)
            scores = gains + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_subset_arrays[best_idx]
            newly_covered = b_subs[current_uncovered[b_subs]]
            
            if len(newly_covered) == 0:
                continue
                
            current_uncovered[newly_covered] = False
            for s in newly_covered:
                for b in subset_block_arrays[s]:
                    gains[b] -= 1
                    
        return selected_blocks
    
    def repair_covering(blocks_list, max_time=2.0, random_seed=None):
        """Repair a potentially invalid covering by adding blocks to cover missing subsets."""
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
        
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in blocks_list:
            for s_idx in block_subset_arrays[b_idx]:
                cover_count[s_idx] += 1
        
        uncovered_mask = (cover_count == 0)
        if not np.any(uncovered_mask):
            return list(blocks_list)
        
        result = list(blocks_list)
        result_set = set(blocks_list)
        
        deadline = time.time() + max_time
        while np.any(uncovered_mask) and time.time() < deadline:
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
            
            # Find blocks that cover the most uncovered subsets
            gains = np.zeros(num_blocks, dtype=np.int32)
            for s_idx in uncovered_indices:
                for b_idx in subset_block_arrays[s_idx]:
                    if b_idx not in result_set:
                        gains[b_idx] += 1
            
            if np.max(gains) == 0:
                break
            
            noise = rng.uniform(0, 0.1, size=num_blocks)
            scores = gains + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
            
            result.append(best_idx)
            result_set.add(best_idx)
            
            b_subs = block_subset_arrays[best_idx]
            newly_covered = b_subs[uncovered_mask[b_subs]]
            if len(newly_covered) == 0:
                continue
            
            uncovered_mask[newly_covered] = False
            for s in newly_covered:
                for b in subset_block_arrays[s]:
                    if b not in result_set:
                        gains[b] -= 1
        
        return result
    
    def crossover(parent1, parent2, rng):
        """Merge two valid coverings into a new one, then try to minimize."""
        # Start with union, but try to be selective
        set1 = set(parent1)
        set2 = set(parent2)
        
        # For each block in set1, check if it's "needed" given set2
        # A block is needed if it covers some subset that no block in set2 covers
        # This is expensive, so we use a heuristic: keep blocks that are in set1 but not set2
        # with some probability, and always keep blocks that are in both
        
        common = set1 & set2
        only1 = set1 - set2
        only2 = set2 - set1
        
        # Keep all common blocks
        child = list(common)
        
        # For blocks only in one parent, keep them with probability 0.5
        # This creates diversity
        for b in only1:
            if rng.random() < 0.5:
                child.append(b)
        for b in only2:
            if rng.random() < 0.5:
                child.append(b)
        
        return child
    
    def mutate_criticality_weighted(blocks_list, cover_count, rng, num_mutations=3):
        """Replace blocks based on criticality. Less critical blocks are more likely to be replaced."""
        result = list(blocks_list)
        result_set = set(blocks_list)
        
        if not result:
            return result
        
        for _ in range(num_mutations):
            if not result:
                break
            
            # Compute criticality for each block: number of subsets it uniquely covers
            criticality = {}
            for b_idx in result:
                unique_count = 0
                for s_idx in block_subset_arrays[b_idx]:
                    if cover_count[s_idx] == 1:
                        unique_count += 1
                criticality[b_idx] = unique_count
            
            # Weight for selection: lower criticality = higher weight for replacement
            total_weight = sum(1.0 / (1.0 + c) for c in criticality.values())
            
            # Select a block to replace
            r = rng.random() * total_weight
            cumsum = 0.0
            selected_block = result[0]
            for b_idx in result:
                cumsum += 1.0 / (1.0 + criticality[b_idx])
                if r <= cumsum:
                    selected_block = b_idx
                    break
            
            # Find a replacement block that covers the unique subsets of selected_block
            unique_subs = []
            for s_idx in block_subset_arrays[selected_block]:
                if cover_count[s_idx] == 1:
                    unique_subs.append(s_idx)
            
            if unique_subs:
                # Find candidates that cover all unique subsets
                candidates = None
                for s_idx in unique_subs:
                    if candidates is None:
                        candidates = set(subset_to_blocks_sets[s_idx])
                    else:
                        candidates.intersection_update(subset_to_blocks_sets[s_idx])
                        if not candidates:
                            break
                
                if candidates:
                    candidates.discard(selected_block)
                    candidates -= result_set
                    if candidates:
                        new_block = rng.choice(list(candidates))
                        # Replace
                        for s_idx in block_subset_arrays[selected_block]:
                            cover_count[s_idx] -= 1
                        for s_idx in block_subset_arrays[new_block]:
                            cover_count[s_idx] += 1
                        result_set.discard(selected_block)
                        result_set.add(new_block)
                        result[result.index(selected_block)] = new_block
            
            else:
                # Block is not critical, just remove it
                for s_idx in block_subset_arrays[selected_block]:
                    cover_count[s_idx] -= 1
                result_set.discard(selected_block)
                result.remove(selected_block)
        
        return result

    def run_greedy_partial(uncovered_mask, random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            return []
            
        gains = np.zeros(num_blocks, dtype=np.int32)
        for s_idx in uncovered_indices:
            for b_idx in subset_block_arrays[s_idx]:
                gains[b_idx] += 1
                
        selected_blocks = []
        current_uncovered = uncovered_mask.copy()
        
        while True:
            if not np.any(current_uncovered):
                break
                
            noise = rng.uniform(0, 0.3, size=num_blocks)
            scores = gains + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_subset_arrays[best_idx]
            newly_covered = b_subs[current_uncovered[b_subs]]
            
            if len(newly_covered) == 0:
                continue
                
            current_uncovered[newly_covered] = False
            for s in newly_covered:
                for b in subset_block_arrays[s]:
                    gains[b] -= 1
                    
        return selected_blocks

    def minimize_with_swaps(blocks_indices, coverage_counts, time_limit, random_seed=None):
        """Local search with deletion and swap operations."""
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        current_block_set = set(current_blocks)
        
        if not current_blocks:
            return current_blocks, coverage_counts.copy()
            
        # Work with a copy of coverage counts
        cover_count = coverage_counts.copy()
        
        improved = True
        start_time = time.time()
        
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            
            # Deletion pass
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            
            for i in indices:
                if i >= len(current_blocks):
                    continue
                if (time.time() - start_time) > time_limit:
                    break
                b_idx = current_blocks[i]
                is_redundant = True
                for s_idx in block_subset_arrays[b_idx]:
                    if cover_count[s_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for s_idx in block_subset_arrays[b_idx]:
                        cover_count[s_idx] -= 1
                    current_block_set.discard(b_idx)
                    current_blocks.pop(i)
                    improved = True
                    break  # Restart scan after removal
            
            # Swap pass
            if (time.time() - start_time) > time_limit:
                break
                
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            
            swaps_done = 0
            max_swaps = 30
            
            for i in indices:
                if (time.time() - start_time) > time_limit:
                    break
                if swaps_done >= max_swaps:
                    break
                if i >= len(current_blocks):
                    continue
                    
                old_block = current_blocks[i]
                
                # Find unique subsets covered by old_block
                unique_subs = []
                for s_idx in block_subset_arrays[old_block]:
                    if cover_count[s_idx] == 1:
                        unique_subs.append(s_idx)
                
                if not unique_subs:
                    # Block is not critical - try swapping with a random block
                    # to diversify the solution
                    new_block = rng.randint(0, num_blocks - 1)
                    if new_block == old_block:
                        continue
                    if new_block in current_block_set:
                        continue
                        
                    # Perform swap
                    for s_idx in block_subset_arrays[old_block]:
                        cover_count[s_idx] -= 1
                    for s_idx in block_subset_arrays[new_block]:
                        cover_count[s_idx] += 1
                    current_block_set.discard(old_block)
                    current_block_set.add(new_block)
                    current_blocks[i] = new_block
                    swaps_done += 1
                else:
                    # Block is critical - find candidates that cover all unique_subs
                    candidates = None
                    for s_idx in unique_subs:
                        if candidates is None:
                            candidates = set(subset_to_blocks_sets[s_idx])
                        else:
                            candidates.intersection_update(subset_to_blocks_sets[s_idx])
                            if not candidates:
                                break
                    
                    if candidates:
                        candidates.discard(old_block)
                        if candidates:
                            # Pick a random candidate
                            new_block = rng.choice(list(candidates))
                            # Check if new_block is already in current_blocks
                            if new_block in current_block_set:
                                continue
                                
                            # Perform swap
                            for s_idx in block_subset_arrays[old_block]:
                                cover_count[s_idx] -= 1
                            for s_idx in block_subset_arrays[new_block]:
                                cover_count[s_idx] += 1
                            current_block_set.discard(old_block)
                            current_block_set.add(new_block)
                            current_blocks[i] = new_block
                            swaps_done += 1
                            improved = True
        
        return current_blocks, cover_count
    
    def run_genetic_algorithm(initial_pop, population_size, num_generations, time_limit, random_seed=None):
        """Run a genetic algorithm to evolve better coverings."""
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
        
        start_time = time.time()
        
        # Initialize population
        population = []
        for blocks in initial_pop:
            cover_count = compute_coverage_counts(blocks)
            if np.all(cover_count > 0):
                population.append((list(blocks), cover_count.copy()))
        
        # If we don't have enough valid solutions, repair them
        while len(population) < population_size and time.time() < start_time + time_limit:
            # Take a random existing solution and perturb it
            if population:
                base, _ = population[rng.integers(0, len(population))]
                perturbed = list(base)
                # Remove some blocks
                num_remove = max(1, len(perturbed) // 20)
                remove_idx = rng.choice(len(perturbed), size=min(num_remove, len(perturbed)), replace=False)
                remove_set = set(perturbed[i] for i in remove_idx)
                perturbed = [b for i, b in enumerate(perturbed) if i not in set(remove_idx.tolist())]
                repaired = repair_covering(perturbed, max_time=1.0, random_seed=int(rng.integers(0, 2**31)))
                if repaired:
                    cover_count = compute_coverage_counts(repaired)
                    if np.all(cover_count > 0):
                        population.append((repaired, cover_count.copy()))
            else:
                break
        
        if not population:
            return None
        
        best_solution = min(population, key=lambda x: len(x[0]))
        best_blocks = best_solution[0]
        best_count = best_solution[1]
        best_len = len(best_blocks)
        
        start_time = time.time()
        
        for gen in range(num_generations):
            if time.time() - start_time > time_limit:
                break
            
            # Selection: tournament selection
            new_population = []
            
            for _ in range(population_size):
                # Tournament of size 3
                candidates = rng.choice(len(population), size=3, replace=False)
                winner = min(candidates, key=lambda i: len(population[i][0]))
                new_population.append((list(population[winner][0]), population[winner][1].copy()))
            
            # Crossover and mutation
            for i in range(0, len(new_population) - 1, 2):
                parent1 = new_population[i][0]
                parent2 = new_population[i + 1][0]
                
                # Crossover
                child = crossover(parent1, parent2, rng)
                
                # Repair if needed
                cover_count = compute_coverage_counts(child)
                if not np.all(cover_count > 0):
                    child = repair_covering(child, max_time=1.0, random_seed=int(rng.integers(0, 2**31)))
                    cover_count = compute_coverage_counts(child)
                
                # Mutation
                if cover_count is not None and np.all(cover_count > 0):
                    child = mutate_criticality_weighted(child, cover_count, rng, num_mutations=2)
                    cover_count = compute_coverage_counts(child)
                
                if cover_count is not None and np.all(cover_count > 0):
                    new_population[i] = (child, cover_count)
                
                # Keep parent if child is worse
                if len(new_population[i][0]) > len(parent1):
                    new_population[i] = (list(parent1), compute_coverage_counts(parent1))
            
            # Elitism: keep the best solution
            if population:
                old_best = min(population, key=lambda x: len(x[0]))
                if len(new_population[0][0]) > len(old_best[0]):
                    new_population[0] = (list(old_best[0]), old_best[1].copy())
            
            # Update best
            gen_best = min(new_population, key=lambda x: len(x[0]))
            if len(gen_best[0]) < best_len:
                best_len = len(gen_best[0])
                best_blocks = gen_best[0]
                best_count = gen_best[1]
            
            population = new_population
        
        return best_blocks

    def compute_coverage(blocks_indices):
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in blocks_indices:
            for s_idx in block_subset_arrays[b_idx]:
                cover_count[s_idx] += 1
        return cover_count

    best_blocks = None
    best_len = float('inf')
    best_coverage = None
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Generate initial diverse population using multiple greedy runs
    initial_solutions = []
    for seed in range(5):
        greedy_res = run_greedy_full(seed)
        coverage = compute_coverage_counts(greedy_res)
        min_res, min_cov = minimize_with_swaps(greedy_res, coverage, time_limit=1.5, random_seed=seed)
        if np.all(min_cov > 0):
            initial_solutions.append(min_res)
            if len(min_res) < best_len:
                best_len = len(min_res)
                best_blocks = min_res
                best_coverage = min_cov
    
    trial = 0
    while time.time() < deadline:
        # Strategy 1: Genetic Algorithm with diverse initial population
        if initial_solutions and time.time() < deadline - 15:
            ga_time = min(8.0, deadline - time.time() - 5)
            ga_result = run_genetic_algorithm(
                initial_solutions, 
                population_size=8, 
                num_generations=10, 
                time_limit=ga_time,
                random_seed=trial + 100
            )
            if ga_result is not None:
                ga_cov = compute_coverage_counts(ga_result)
                if np.all(ga_cov > 0) and len(ga_result) < best_len:
                    best_len = len(ga_result)
                    best_blocks = ga_result
                    best_coverage = ga_cov
                    # Update initial solutions with the best GA result
                    initial_solutions[0] = ga_result
        
        # Strategy 2: Full Greedy + Local Search (deletion + swap)
        if time.time() < deadline - 10:
            greedy_res = run_greedy_full(trial + 10)
            coverage = compute_coverage_counts(greedy_res)
            
            min_res, min_cov = minimize_with_swaps(greedy_res, coverage, time_limit=2.0, random_seed=trial + 10)
            
            if len(min_res) < best_len:
                best_len = len(min_res)
                best_blocks = min_res
                best_coverage = min_cov
                # Add to initial solutions for GA
                if len(initial_solutions) < 5:
                    initial_solutions.append(min_res)
                elif random.random() < 0.3:
                    initial_solutions[random.randint(0, 4)] = min_res
        
        # Strategy 3: Perturb Best + Repair + Local Search
        if best_blocks is not None and len(best_blocks) > 20 and time.time() < deadline - 10:
            perturbed = list(best_blocks)
            # Remove more blocks to explore larger neighborhood
            num_remove = min(max(2, len(perturbed) // 15), len(perturbed) - 1)
            if num_remove > 0:
                remove_indices = random.sample(range(len(perturbed)), num_remove)
                blocks_to_remove = set(perturbed[i] for i in remove_indices)
                
                cover_count_temp = np.zeros(num_subsets, dtype=np.int32)
                for b_idx in perturbed:
                    if b_idx in blocks_to_remove:
                        continue
                    for s_idx in block_subset_arrays[b_idx]:
                        cover_count_temp[s_idx] += 1
                        
                uncovered_mask = (cover_count_temp == 0).astype(bool)
                
                repair_res = run_greedy_partial(uncovered_mask, trial + 1000)
                
                new_set = [b for b in perturbed if b not in blocks_to_remove]
                new_set.extend(repair_res)
                
                # Compute coverage for new set
                new_coverage = compute_coverage_counts(new_set)
                
                min_res2, min_cov2 = minimize_with_swaps(new_set, new_coverage, time_limit=2.0, random_seed=trial + 2000)
                
                if len(min_res2) < best_len:
                    best_len = len(min_res2)
                    best_blocks = min_res2
                    best_coverage = min_cov2
                    # Add to initial solutions
                    if len(initial_solutions) < 5:
                        initial_solutions.append(min_res2)
        
        # Strategy 4: Pure swap-based diversification on best solution
        if best_blocks is not None and len(best_blocks) > 20 and time.time() < deadline - 5:
            swap_res, swap_cov = minimize_with_swaps(best_blocks, best_coverage, time_limit=3.0, random_seed=trial + 5000)
            
            if len(swap_res) < best_len:
                best_len = len(swap_res)
                best_blocks = swap_res
                best_coverage = swap_cov

        trial += 1
        
    if best_blocks is None:
        greedy_res = run_greedy_full(0)
        coverage = compute_coverage_counts(greedy_res)
        best_blocks, best_coverage = minimize_with_swaps(greedy_res, coverage, time_limit=5.0, random_seed=0)
        
    return [list(all_blocks[i]) for i in best_blocks]
