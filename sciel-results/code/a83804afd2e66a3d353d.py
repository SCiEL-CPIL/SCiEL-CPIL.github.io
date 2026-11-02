<<<<<<< SEARCH
import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Precomputation
    all_tsets = list(itertools.combinations(range(v), t))
    tset_masks = [0] * len(all_tsets)
    mask_to_idx = {}
    for i, comb in enumerate(all_tsets):
        mask = 0
        for p in comb:
            mask |= (1 << p)
        tset_masks[i] = mask
        mask_to_idx[mask] = i
        
    num_tsets = len(all_tsets)
    
    best_blocks = None
    best_len = float('inf')
    
    # Run greedy with different tie-breakers/randomizations
    # Since the problem is small, we can try a few variations
    
    for attempt in range(10): # Try 10 times
        if time.time() > deadline:
            break
            
        uncovered = np.ones(num_tsets, dtype=bool)
        uncovered_indices = set(range(num_tsets))
        blocks = []
        
        while uncovered_indices:
            if time.time() > deadline:
                break
                
            # Pick a start t-set
            # Randomly pick one to diversify
            start_idx = next(iter(uncovered_indices)) # Or random choice
            # To be deterministic but diverse, we can shuffle the order of selection?
            # Let's stick to picking the first one, but randomize the candidate order in gain calculation?
            # Or just pick a random uncovered t-set.
            start_idx = list(uncovered_indices)[attempt % len(uncovered_indices)] # Simple variation
            
            start_mask = tset_masks[start_idx]
            current_points = [p for p in range(v) if start_mask & (1 << p)]
            
            while len(current_points) < k:
                best_x = -1
                best_gain = -1
                
                # Randomize candidate order for tie-breaking diversity
                candidates = [x for x in range(v) if not (start_mask & (1 << x) or (len(current_points)>0 and any(p==x for p in current_points[1:])))]
                # Simpler:
                current_mask = 0
                for p in current_points:
                    current_mask |= (1 << p)
                
                candidates = [x for x in range(v) if not (current_mask & (1 << x))]
                
                if attempt % 2 == 0:
                    candidates.reverse() # Different order for different attempts?
                
                for x in candidates:
                    new_points = current_points + [x]
                    # Calculate gain
                    gain = 0
                    # Generate 5-subsets of new_points
                    # Optimization: Only check subsets that include the new point x?
                    # No, the whole set is covered. But we only care about NEW coverage.
                    # Actually, any t-set in new_points that is in uncovered is a gain.
                    # Since previous points were already in current_points, any t-set NOT containing x was already considered in previous steps?
                    # No, a t-set containing x is new. A t-set not containing x was a subset of current_points.
                    # If it was uncovered, it would have been covered by previous blocks?
                    # Not necessarily.
                    # So we must check all 5-subsets of new_points.
                    
                    for sub in itertools.combinations(new_points, t):
                        sub_mask = 0
                        for p in sub:
                            sub_mask |= (1 << p)
                        idx = mask_to_idx[sub_mask]
                        if uncovered[idx]:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        
                if best_x == -1:
                    # Should not happen if uncovered is not empty
                    break
                    
                current_points.append(best_x)
                
            # Finalize block
            final_points = current_points
            final_mask = 0
            for p in final_points:
                final_mask |= (1 << p)
                
            # Update uncovered
            for sub in itertools.combinations(final_points, t):
                sub_mask = 0
                for p in sub:
                    sub_mask |= (1 << p)
                idx = mask_to_idx[sub_mask]
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_indices.remove(idx)
                    
            blocks.append(final_points)
            
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    return best_blocks
=======
import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Precomputation
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    
    # Map each t-set to an index
    tset_to_idx = {}
    for i, comb in enumerate(all_tsets):
        tset_to_idx[comb] = i
    
    # Precompute all k-subsets (candidate blocks) and their covered t-sets
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For each block, precompute the list of t-set indices it covers
    block_coverage = []
    for block in all_blocks:
        covered = []
        for sub in itertools.combinations(block, t):
            idx = tset_to_idx[sub]
            covered.append(idx)
        block_coverage.append(covered)
    
    # Convert to numpy arrays for fast operations
    block_coverage_sets = [set(cov) for cov in block_coverage]
    
    def greedy_construct(seed_offset=0):
        """Greedy construction with deterministic seed for reproducibility."""
        rng = random.Random(seed_offset)
        uncovered = set(range(num_tsets))
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                break
                
            # Pick a random uncovered t-set to start
            start_tset_idx = rng.choice(list(uncovered))
            start_tset = all_tsets[start_tset_idx]
            current_points = list(start_tset)
            
            # Greedily add points to maximize coverage
            while len(current_points) < k:
                best_x = -1
                best_gain = -1
                current_set = set(current_points)
                
                candidates = [x for x in range(v) if x not in current_set]
                rng.shuffle(candidates)
                
                for x in candidates:
                    new_points = current_points + [x]
                    new_set = set(new_points)
                    
                    # Count how many currently uncovered t-sets are covered by new_points
                    gain = 0
                    for sub in itertools.combinations(new_points, t):
                        idx = tset_to_idx[sub]
                        if idx in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        
                if best_x == -1:
                    break
                current_points.append(best_x)
            
            # Finalize block
            final_points = tuple(current_points)
            if final_points in all_blocks:
                block_idx = all_blocks.index(final_points)
                covered = block_coverage_sets[block_idx]
                uncovered -= covered
                blocks.append(list(final_points))
            
            if time.time() > deadline:
                break
        
        return blocks
    
    def evaluate_solution(blocks):
        """Evaluate a solution: returns (num_blocks, num_uncovered)."""
        if not blocks:
            return (0, num_tsets)
        
        covered = set()
        for block in blocks:
            block_tuple = tuple(block)
            if block_tuple in all_blocks:
                idx = all_blocks.index(block_tuple)
                covered |= block_coverage_sets[idx]
        
        num_uncovered = num_tsets - len(covered)
        return (len(blocks), num_uncovered)
    
    def fitness(blocks):
        """Multi-objective fitness: penalize both block count and uncovered t-sets."""
        num_blocks, num_uncovered = evaluate_solution(blocks)
        # Heavily penalize uncovered t-sets
        return num_blocks + 1000 * num_uncovered
    
    def crossover(parent1, parent2):
        """Crossover: combine blocks from both parents, removing duplicates."""
        if len(parent1) > len(parent2):
            parent1, parent2 = parent2, parent1
        
        child = list(parent1)
        seen = set(tuple(b) for b in child)
        
        for block in parent2:
            block_tuple = tuple(block)
            if block_tuple not in seen:
                child.append(block)
                seen.add(block_tuple)
        
        return child
    
    def mutate(blocks, mutation_rate=0.1):
        """Mutation: randomly replace or remove blocks."""
        child = list(blocks)
        rng = random.Random()
        
        # Remove some blocks
        for i in range(len(child)):
            if rng.random() < mutation_rate:
                del child[i]
                break
        
        # Add some new random blocks
        for _ in range(rng.randint(0, 3)):
            new_block = list(rng.sample(range(v), k))
            if tuple(new_block) not in set(tuple(b) for b in child):
                child.append(new_block)
        
        return child
    
    def local_search(blocks, max_iter=50):
        """Local search: try to improve by swapping blocks."""
        if not blocks:
            return blocks
        
        best = list(blocks)
        best_fit = fitness(best)
        rng = random.Random()
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Try removing a block
            if len(best) > 1:
                idx = rng.randint(0, len(best) - 1)
                candidate = best[:idx] + best[idx+1:]
                fit = fitness(candidate)
                if fit < best_fit:
                    best = candidate
                    best_fit = fit
            
            # Try replacing a block
            if len(best) > 0:
                idx = rng.randint(0, len(best) - 1)
                new_block = list(rng.sample(range(v), k))
                candidate = best[:idx] + [new_block] + best[idx+1:]
                fit = fitness(candidate)
                if fit < best_fit:
                    best = candidate
                    best_fit = fit
        
        return best
    
    # Phase 1: Generate initial population using greedy
    population = []
    for i in range(5):
        if time.time() > deadline:
            break
        blocks = greedy_construct(i)
        if blocks:
            population.append(blocks)
    
    if not population:
        # Fallback: return empty or a trivial solution
        return []
    
    # Track best solution
    best_blocks = population[0]
    best_fit = fitness(best_blocks)
    
    # Phase 2: Genetic algorithm
    pop_size = len(population)
    generation = 0
    max_generations = 20
    
    while generation < max_generations and time.time() < deadline:
        generation += 1
        
        # Evaluate all individuals
        fits = [fitness(ind) for ind in population]
        
        # Update best
        min_fit_idx = np.argmin(fits)
        if fits[min_fit_idx] < best_fit:
            best_fit = fits[min_fit_idx]
            best_blocks = population[min_fit_idx]
        
        # Selection: tournament selection
        new_population = []
        
        # Keep the best individual (elitism)
        new_population.append(population[min_fit_idx])
        
        while len(new_population) < pop_size:
            if time.time() > deadline:
                break
            
            # Tournament selection
            p1_idx = rng.randint(0, pop_size - 1)
            p2_idx = rng.randint(0, pop_size - 1)
            if fits[p1_idx] <= fits[p2_idx]:
                parent1 = population[p1_idx]
            else:
                parent1 = population[p2_idx]
            
            p3_idx = rng.randint(0, pop_size - 1)
            p4_idx = rng.randint(0, pop_size - 1)
            if fits[p3_idx] <= fits[p4_idx]:
                parent2 = population[p3_idx]
            else:
                parent2 = population[p4_idx]
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate(child, mutation_rate=0.15)
            
            # Local search on child
            child = local_search(child, max_iter=10)
            
            new_population.append(child)
        
        population = new_population
    
    # Final local search on best solution
    best_blocks = local_search(best_blocks, max_iter=100)
    
    # Ensure we have a valid solution (all t-sets covered)
    # If not, extend with greedy
    covered = set()
    for block in best_blocks:
        block_tuple = tuple(block)
        if block_tuple in all_blocks:
            idx = all_blocks.index(block_tuple)
            covered |= block_coverage_sets[idx]
    
    uncovered = set(range(num_tsets)) - covered
    
    if uncovered and time.time() < deadline - 5:
        # Extend with greedy to ensure coverage
        rng = random.Random(999)
        while uncovered and time.time() < deadline - 2:
            start_tset_idx = rng.choice(list(uncovered))
            start_tset = all_tsets[start_tset_idx]
            current_points = list(start_tset)
            
            while len(current_points) < k:
                best_x = -1
                best_gain = -1
                current_set = set(current_points)
                
                candidates = [x for x in range(v) if x not in current_set]
                rng.shuffle(candidates)
                
                for x in candidates:
                    new_points = current_points + [x]
                    gain = 0
                    for sub in itertools.combinations(new_points, t):
                        idx = tset_to_idx[sub]
                        if idx in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x == -1:
                    break
                current_points.append(best_x)
            
            final_points = tuple(current_points)
            if final_points in all_blocks:
                block_idx = all_blocks.index(final_points)
                covered_new = block_coverage_sets[block_idx]
                uncovered -= covered_new
                best_blocks.append(list(final_points))
    
    return best_blocks
>>>>>>> REPLACE
