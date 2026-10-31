import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110
    
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    n_t_subsets = len(all_t_subsets)
    
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Pre-compute all k-subsets and their t-coverage
    all_k_subsets = list(itertools.combinations(points, k))
    n_k_subsets = len(all_k_subsets)
    
    block_t_indices = []
    degree = [0] * n_t_subsets
    
    for block in all_k_subsets:
        covered_ts = []
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            covered_ts.append(idx)
            degree[idx] += 1
        block_t_indices.append(covered_ts)
    
    # Step 1: LP Duality - Identify critical t-subsets
    # Dual variables y_j represent weights on t-subsets
    # Constraint: sum_{j in S_i} y_j <= 1 for each block i
    # Objective: maximize sum_j y_j
    # Heuristic: y_j proportional to 1/degree(j) for low-degree subsets
    # These are "bottleneck" subsets that are hard to cover
    
    # Compute dual weights using iterative proportional fitting
    y = np.ones(n_t_subsets) / np.maximum(degree, 1)
    
    # Iteratively adjust weights to satisfy dual constraints
    for iteration in range(50):
        if time.time() - start_time > time_limit * 0.05:
            break
        # For each block, compute sum of y over its t-subsets
        block_sums = np.zeros(n_k_subsets)
        for b_idx, ts_list in enumerate(block_t_indices):
            block_sums[b_idx] = np.sum(y[ts_list])
        
        # Scale y down where block constraints are violated
        violated_blocks = np.where(block_sums > 1.0)[0]
        if len(violated_blocks) == 0:
            break
        
        # Reduce y for t-subsets in violated blocks
        for b_idx in violated_blocks[:100]:  # Limit work per iteration
            for ts_idx in block_t_indices[b_idx]:
                y[ts_idx] *= 0.95
        
        # Normalize to keep objective meaningful
        y = y / np.max(y) if np.max(y) > 0 else y
    
    # Critical subsets: those with highest dual weights
    # Sort by weight, take top fraction
    sorted_ts_by_weight = np.argsort(-y)
    n_critical = max(1, n_t_subsets // 4)  # Top 25% most critical
    critical_ts_set = set(sorted_ts_by_weight[:n_critical].tolist())
    
    # Build reverse map for critical subsets only
    ts_to_blocks = [[] for _ in range(n_t_subsets)]
    for b_idx, ts_list in enumerate(block_t_indices):
        for ts_idx in ts_list:
            if ts_idx in critical_ts_set:
                ts_to_blocks[ts_idx].append(b_idx)
    
    # Step 2: Genetic Algorithm optimized for critical constraints
    # Fitness: primarily minimize uncovered critical subsets, secondarily minimize block count
    
    def compute_fitness(block_set):
        """Returns (uncovered_critical, total_blocks) - both minimized"""
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
    
    def local_search(block_set, focus_critical=True):
        """Local search to minimize blocks while maintaining critical coverage"""
        current_blocks = list(block_set)
        
        # Build coverage for all t-subsets
        coverage = [0] * n_t_subsets
        for b_idx in current_blocks:
            for ts_idx in block_t_indices[b_idx]:
                coverage[ts_idx] += 1
        
        # Ensure critical subsets are covered
        critical_uncovered = [i for i in critical_ts_set if coverage[i] == 0]
        while critical_uncovered:
            target = random.choice(critical_uncovered)
            candidates = ts_to_blocks[target]
            if not candidates:
                # Fallback: try all blocks
                candidates = list(range(n_k_subsets))
                candidates = [b for b in candidates if target in block_t_indices[b]]
                if not candidates:
                    break
            
            # Pick block covering most uncovered critical subsets
            best_b = -1
            best_gain = -1
            for b in candidates:
                gain = 0
                for ts_idx in block_t_indices[b]:
                    if ts_idx in critical_ts_set and coverage[ts_idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_b = b
            
            if best_b == -1:
                break
            if best_b not in current_blocks:
                current_blocks.append(best_b)
                for ts_idx in block_t_indices[best_b]:
                    coverage[ts_idx] += 1
            
            critical_uncovered = [i for i in critical_ts_set if coverage[i] == 0]
        
        # Minimize block count
        improved = True
        while improved and time.time() - start_time < time_limit * 0.85:
            improved = False
            
            # Deletion pass: remove blocks that don't cover any unique critical subset
            for i in range(len(current_blocks)):
                b_idx = current_blocks[i]
                can_remove = True
                for ts_idx in block_t_indices[b_idx]:
                    if ts_idx in critical_ts_set and coverage[ts_idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for ts_idx in block_t_indices[b_idx]:
                        coverage[ts_idx] -= 1
                    current_blocks.pop(i)
                    improved = True
                    break
            
            if time.time() - start_time > time_limit * 0.8:
                break
            
            # Swap pass: replace blocks to improve efficiency
            if not improved:
                indices = list(range(len(current_blocks)))
                random.shuffle(indices)
                for i in indices:
                    old_b = current_blocks[i]
                    # Find critical subsets uniquely covered by this block
                    unique_critical = [ts for ts in block_t_indices[old_b] 
                                      if ts in critical_ts_set and coverage[ts] == 1]
                    if not unique_critical:
                        continue
                    
                    # Find candidate covering all unique critical subsets
                    candidates = set(ts_to_blocks[unique_critical[0]])
                    for ts in unique_critical[1:]:
                        candidates.intersection_update(ts_to_blocks[ts])
                    candidates.discard(old_b)
                    
                    if not candidates:
                        continue
                    
                    # Pick best candidate
                    best_new = -1
                    best_score = -1
                    cand_list = list(candidates)
                    if len(cand_list) > 10:
                        cand_list = random.sample(cand_list, 10)
                    for nb in cand_list:
                        score = 0
                        for ts_idx in block_t_indices[nb]:
                            if ts_idx in critical_ts_set and coverage[ts_idx] == 1:
                                score += 1
                        if score > best_score:
                            best_score = score
                            best_new = nb
                    
                    if best_new != -1:
                        for ts_idx in block_t_indices[old_b]:
                            coverage[ts_idx] -= 1
                        for ts_idx in block_t_indices[best_new]:
                            coverage[ts_idx] += 1
                        current_blocks[i] = best_new
                        improved = True
                        break
        
        return current_blocks
    
    # Step 3: Initialize population with greedy + random strategies
    population = []
    pop_size = 12
    
    for _ in range(pop_size):
        # Greedy construction focused on critical subsets
        coverage = [0] * n_t_subsets
        selected = []
        uncovered_critical = set(critical_ts_set)
        
        while uncovered_critical and time.time() - start_time < time_limit * 0.15:
            # Pick random uncovered critical target
            target = random.choice(list(uncovered_critical))
            
            # Find best block covering most uncovered critical subsets
            best_block = -1
            best_gain = -1.0
            candidates = ts_to_blocks[target]
            if not candidates:
                candidates = [b for b in range(n_k_subsets) if target in block_t_indices[b]]
            
            for b_idx in candidates:
                gain = 0.0
                for ts_idx in block_t_indices[b_idx]:
                    if ts_idx in uncovered_critical:
                        gain += 1.0
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                break
            if best_block not in selected:
                selected.append(best_block)
                for ts_idx in block_t_indices[best_block]:
                    coverage[ts_idx] += 1
                    if ts_idx in uncovered_critical:
                        uncovered_critical.discard(ts_idx)
        
        # Local search optimization
        optimized = local_search(selected)
        population.append(optimized)
    
    # Step 4: GA loop
    best_solution = min(population, key=lambda x: compute_fitness(x))
    
    generations = 0
    max_generations = 100
    
    while time.time() - start_time < time_limit * 0.75 and generations < max_generations:
        generations += 1
        
        # Tournament selection
        def tournament():
            p1 = random.choice(population)
            p2 = random.choice(population)
            f1 = compute_fitness(p1)
            f2 = compute_fitness(p2)
            return p1 if f1 < f2 else p2
        
        new_population = []
        
        for _ in range(pop_size // 2):
            p1 = tournament()
            p2 = tournament()
            
            # Crossover: merge and reduce
            child_blocks = list(set(p1) | set(p2))
            
            # Mutation: randomly remove some blocks
            if len(child_blocks) > 5:
                remove_count = random.randint(1, max(1, len(child_blocks) // 5))
                for _ in range(remove_count):
                    if child_blocks:
                        idx = random.randint(0, len(child_blocks) - 1)
                        child_blocks.pop(idx)
            
            # Local search
            optimized_child = local_search(child_blocks)
            new_population.append(optimized_child)
            
            # Elitism: keep parents
            new_population.append(p1)
            new_population.append(p2)
        
        # Update population
        population.extend(new_population)
        population.sort(key=lambda x: compute_fitness(x))
        population = population[:pop_size]
        
        current_best = population[0]
        if compute_fitness(current_best) < compute_fitness(best_solution):
            best_solution = current_best
    
    # Final optimization
    final_blocks = local_search(best_solution)
    
    # Verify full coverage (not just critical)
    coverage = [0] * n_t_subsets
    for b_idx in final_blocks:
        for ts_idx in block_t_indices[b_idx]:
            coverage[ts_idx] += 1
    
    # If not fully covered, add blocks to fix
    uncovered = [i for i in range(n_t_subsets) if coverage[i] == 0]
    while uncovered:
        target = random.choice(uncovered)
        candidates = [b for b in range(n_k_subsets) if target in block_t_indices[b]]
        if not candidates:
            break
        
        best_b = -1
        best_gain = -1
        for b in candidates:
            gain = sum(1 for ts in block_t_indices[b] if coverage[ts] == 0)
            if gain > best_gain:
                best_gain = gain
                best_b = b
        
        if best_b == -1:
            break
        if best_b not in final_blocks:
            final_blocks.append(best_b)
            for ts in block_t_indices[best_b]:
                coverage[ts] += 1
        
        uncovered = [i for i in range(n_t_subsets) if coverage[i] == 0]
    
    return [list(all_k_subsets[i]) for i in final_blocks]
