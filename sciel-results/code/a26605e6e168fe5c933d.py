import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    time_limit = 105
    
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    n_t_subsets = len(all_t_subsets)
    
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Pre-compute all k-subsets
    all_k_subsets = list(itertools.combinations(points, k))
    n_k_subsets = len(all_k_subsets)
    
    # Map block index to list of t-subset indices it covers
    block_t_indices = []
    degree = [0] * n_t_subsets
    
    for block in all_k_subsets:
        covered_ts = []
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            covered_ts.append(idx)
            degree[idx] += 1
        block_t_indices.append(covered_ts)
        
    # 1. LP Duality / Probabilistic Packing Lower Bound Approximation
    # We identify "critical" t-subsets. In the dual of the Set Cover LP,
    # variables y_j correspond to t-subsets. The constraint is sum_{j in S_i} y_j <= 1.
    # A simple heuristic for dual weights is 1/degree.
    # Subsets with low degree are "tight" in the packing sense and critical for covering.
    weights = [1.0 / d if d > 0 else 0.0 for d in degree]
    
    # Sort t-subsets by criticality (highest weight / lowest degree)
    sorted_ts_indices = sorted(range(n_t_subsets), key=lambda i: weights[i], reverse=True)
    
    # Define a subset of "critical" constraints to focus the GA.
    # We focus on the top 15% most critical subsets (tighter focus)
    n_critical = max(1, n_t_subsets // 7)
    critical_ts_set = set(sorted_ts_indices[:n_critical])
    
    # Build reverse map: t_subset_index -> list of block indices
    ts_to_blocks = [[] for _ in range(n_t_subsets)]
    for b_idx, ts_list in enumerate(block_t_indices):
        for ts_idx in ts_list:
            ts_to_blocks[ts_idx].append(b_idx)

    def get_fitness(block_set):
        """
        Returns (uncovered_critical_count, total_blocks).
        Lower is better.
        """
        if not block_set:
            return (n_critical, 0)
        
        # Count coverage of critical subsets
        covered_critical = set()
        for b_idx in block_set:
            for ts_idx in block_t_indices[b_idx]:
                if ts_idx in critical_ts_set:
                    covered_critical.add(ts_idx)
        
        uncovered_critical = n_critical - len(covered_critical)
        return (uncovered_critical, len(block_set))

    def local_search(block_set, time_budget=30):
        """
        Standard local search to minimize block count while maintaining full coverage.
        """
        current_blocks = list(block_set)
        ls_start = time.time()
        
        # Calculate full coverage
        coverage = [0] * n_t_subsets
        for b_idx in current_blocks:
            for ts_idx in block_t_indices[b_idx]:
                coverage[ts_idx] += 1
        
        # If not fully covered, add blocks to fix (Greedy repair)
        uncovered = [i for i in range(n_t_subsets) if coverage[i] == 0]
        while uncovered and time.time() - ls_start < time_budget * 0.3:
            # Pick the uncovered subset with fewest covering blocks (hardest to cover)
            target = min(uncovered, key=lambda i: len(ts_to_blocks[i]))
            candidates = ts_to_blocks[target]
            if not candidates:
                break
            # Greedy: pick block covering most uncovered
            best_b = -1
            best_gain = -1
            for b in candidates:
                gain = 0
                for ts in block_t_indices[b]:
                    if coverage[ts] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_b = b
            if best_b == -1:
                break
            if best_b not in current_blocks:
                current_blocks.append(best_b)
                for ts in block_t_indices[best_b]:
                    coverage[ts] += 1
            uncovered = [i for i in range(n_t_subsets) if coverage[i] == 0]
            
        # Now minimize
        improved = True
        while improved and time.time() - ls_start < time_budget:
            improved = False
            # Deletion pass
            for i in range(len(current_blocks)):
                b_idx = current_blocks[i]
                can_remove = True
                for ts in block_t_indices[b_idx]:
                    if coverage[ts] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for ts in block_t_indices[b_idx]:
                        coverage[ts] -= 1
                    current_blocks.pop(i)
                    improved = True
                    break
            
            if time.time() - ls_start > time_budget * 0.8:
                break
                
            # Swap pass
            if not improved:
                indices = list(range(len(current_blocks)))
                random.shuffle(indices)
                for i in indices:
                    old_b = current_blocks[i]
                    unique_ts = [ts for ts in block_t_indices[old_b] if coverage[ts] == 1]
                    if not unique_ts:
                        continue
                    # Find candidate covering all unique_ts
                    unique_ts.sort(key=lambda x: len(ts_to_blocks[x]))
                    candidates = set(ts_to_blocks[unique_ts[0]])
                    for ts in unique_ts[1:]:
                        candidates.intersection_update(ts_to_blocks[ts])
                    candidates.discard(old_b)
                    if not candidates:
                        continue
                    # Pick best candidate (max overlap with low coverage)
                    best_new = -1
                    best_score = -1
                    cand_list = list(candidates)
                    if len(cand_list) > 15:
                        cand_list = random.sample(cand_list, 15)
                    for nb in cand_list:
                        score = 0
                        for ts in block_t_indices[nb]:
                            if coverage[ts] == 1:
                                score += 1
                        if score > best_score:
                            best_score = score
                            best_new = nb
                    if best_new != -1:
                        # Swap
                        for ts in block_t_indices[old_b]:
                            coverage[ts] -= 1
                        for ts in block_t_indices[best_new]:
                            coverage[ts] += 1
                        current_blocks[i] = best_new
                        improved = True
                        break
                        
        return current_blocks

    # 2. Genetic Algorithm Initialization
    # Create initial population using Greedy Set Cover with different random seeds/orderings
    population = []
    pop_size = 8
    
    for _ in range(pop_size):
        # Greedy construction
        coverage = [0] * n_t_subsets
        selected = []
        uncovered_count = n_t_subsets
        
        # Randomize the order of selection slightly to create diversity
        # We pick a random uncovered target, then best block
        while uncovered_count > 0 and time.time() - start_time < time_limit * 0.15:
            uncovered_indices = [i for i, c in enumerate(coverage) if c == 0]
            if not uncovered_indices:
                break
            target = random.choice(uncovered_indices)
            
            best_block = -1
            best_gain = -1.0
            for b_idx in ts_to_blocks[target]:
                gain = 0.0
                for ts_idx in block_t_indices[b_idx]:
                    if coverage[ts_idx] == 0:
                        gain += weights[ts_idx]
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                break
            if best_block not in selected:
                selected.append(best_block)
                for ts_idx in block_t_indices[best_block]:
                    if coverage[ts_idx] == 0:
                        uncovered_count -= 1
                    coverage[ts_idx] += 1
        
        # Repair and Local Search with time budget
        optimized = local_search(selected, time_budget=5)
        population.append(optimized)

    # 3. Genetic Algorithm Loop
    best_solution = min(population, key=lambda x: len(x))
    
    generations = 0
    max_generations = 150
    
    while time.time() - start_time < time_limit * 0.85 and generations < max_generations:
        generations += 1
        
        # Selection: Tournament
        def tournament():
            p1 = random.choice(population)
            p2 = random.choice(population)
            return p1 if len(p1) < len(p2) else p2
        
        new_population = []
        
        for _ in range(pop_size // 2):
            p1 = tournament()
            p2 = tournament()
            
            # Strategy: "Crossover" as "Merge and Reduce"
            # Merge blocks from p1 and p2, but limit size to avoid explosion
            child_blocks = list(set(p1) | set(p2))
            
            # If child is too large, randomly sample a subset
            max_child_size = min(len(child_blocks), max(len(p1), len(p2)) + 10)
            if len(child_blocks) > max_child_size:
                child_blocks = random.sample(child_blocks, max_child_size)
            
            # Mutation: Randomly swap a few blocks
            if child_blocks and random.random() < 0.3:
                for _ in range(min(3, len(child_blocks))):
                    idx = random.randint(0, len(child_blocks) - 1)
                    # Try to find a better replacement
                    old_b = child_blocks[idx]
                    coverage_temp = [0] * n_t_subsets
                    for b in child_blocks:
                        for ts in block_t_indices[b]:
                            coverage_temp[ts] += 1
                    unique_ts = [ts for ts in block_t_indices[old_b] if coverage_temp[ts] == 1]
                    if unique_ts:
                        unique_ts.sort(key=lambda x: len(ts_to_blocks[x]))
                        candidates = set(ts_to_blocks[unique_ts[0]])
                        for ts in unique_ts[1:]:
                            candidates.intersection_update(ts_to_blocks[ts])
                        candidates.discard(old_b)
                        if candidates:
                            child_blocks[idx] = random.choice(list(candidates))
            
            # Local search with time budget
            remaining_time = time_limit * 0.85 - (time.time() - start_time)
            ls_budget = max(2, min(8, remaining_time / max(1, max_generations - generations)))
            optimized_child = local_search(child_blocks, time_budget=ls_budget)
            new_population.append(optimized_child)
            
            # Elitism: keep the better parent
            new_population.append(p1 if len(p1) <= len(p2) else p2)
            
        # Update population with new candidates, keeping the best pop_size
        population.extend(new_population)
        
        # Sort by size
        population.sort(key=lambda x: len(x))
        population = population[:pop_size]
        
        # Update best
        current_best = population[0]
        if len(current_best) < len(best_solution):
            best_solution = current_best

    # Final cleanup on best_solution with generous time budget
    final_blocks = local_search(best_solution, time_budget=10)
    
    return [list(all_k_subsets[i]) for i in final_blocks]
