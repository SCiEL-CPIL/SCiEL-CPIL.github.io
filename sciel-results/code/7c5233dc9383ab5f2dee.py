import itertools
import time
import numpy as np
import random
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_index = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Precompute all possible blocks
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute for each block, which t-subsets it covers
    # block_cover[block_idx] = list of t_subset indices covered by that block
    block_cover = []
    for block in all_blocks:
        covered = []
        for t_sub in itertools.combinations(block, t):
            covered.append(t_subset_index[t_sub])
        block_cover.append(covered)
    
    def evaluate_block_set(block_indices):
        """Count uncovered t-subsets. Returns (num_uncovered, coverage_array)."""
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in block_indices:
            for t_idx in block_cover[b_idx]:
                coverage[t_idx] += 1
        num_uncovered = np.sum(coverage == 0)
        return num_uncovered, coverage
    
    def greedy_initial_solution():
        """Standard greedy set cover to get a starting solution."""
        active = np.ones(num_t_subsets, dtype=np.float64)
        selected = []
        
        # Precompute gains for each block
        # gain[block_idx] = number of currently uncovered t-subsets it covers
        gains = np.zeros(num_blocks, dtype=np.float64)
        for b_idx in range(num_blocks):
            gains[b_idx] = len(block_cover[b_idx])
        
        while True:
            if time.time() > deadline - 30.0:
                break
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            b_idx = np.argmax(gains)
            selected.append(b_idx)
            
            # Update gains: for each t-subset covered by this block, 
            # decrement gains for all other blocks that also cover it
            covered_t = block_cover[b_idx]
            for t_idx in covered_t:
                if active[t_idx] > 0:
                    active[t_idx] = 0
                    # Find all blocks covering this t-subset and decrement their gain
                    # This is expensive; instead, we'll just recompute gains periodically
                    pass
            
            # Recompute gains from scratch every few steps for correctness
            if len(selected) % 5 == 0:
                gains = np.zeros(num_blocks, dtype=np.float64)
                for b_idx in range(num_blocks):
                    count = 0
                    for t_idx in block_cover[b_idx]:
                        if active[t_idx] > 0:
                            count += 1
                    gains[b_idx] = count
            
            if np.all(active == 0):
                break
        
        return selected
    
    # Phase 1: Greedy initial solution
    initial_solution = greedy_initial_solution()
    num_uncovered, coverage = evaluate_block_set(initial_solution)
    best_solution = list(initial_solution)
    best_uncovered = num_uncovered
    
    # Phase 2: Genetic Algorithm
    pop_size = 30
    elite_size = 5
    mutation_rate = 0.3
    crossover_rate = 0.7
    
    # Initialize population: elite + random
    population = [list(best_solution)]
    
    # Add some varied solutions
    for i in range(pop_size - 1):
        if time.time() > deadline - 20.0:
            break
        # Random solution with size close to best
        size = max(100, len(best_solution) - 20 + random.randint(-10, 10))
        size = min(size, num_blocks)
        sol = random.sample(range(num_blocks), size)
        population.append(sol)
    
    # Evaluate initial population
    pop_fitness = []
    for sol in population:
        nu, _ = evaluate_block_set(sol)
        pop_fitness.append(nu)
    
    best_in_pop = min(pop_fitness)
    best_idx = pop_fitness.index(best_in_pop)
    if best_in_pop < best_uncovered:
        best_uncovered = best_in_pop
        best_solution = list(population[best_idx])
    
    generation = 0
    no_improvement_count = 0
    
    while time.time() < deadline - 10.0:
        generation += 1
        
        # Sort by fitness
        sorted_indices = np.argsort(pop_fitness)
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [pop_fitness[i] for i in sorted_indices]
        
        # Elitism
        new_population = [list(sorted_population[i]) for i in range(elite_size)]
        new_fitness = sorted_fitness[:elite_size]
        
        # Check for improvement
        if sorted_fitness[0] < best_uncovered:
            best_uncovered = sorted_fitness[0]
            best_solution = list(sorted_population[0])
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # If stuck for too many generations, try more aggressive mutation
        if no_improvement_count > 20:
            mutation_rate = min(0.8, mutation_rate * 1.2)
            no_improvement_count = 0
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 10.0:
                break
            
            # Selection: tournament
            idx1, idx2 = random.sample(range(len(sorted_population)), 2)
            parent1 = sorted_population[idx1] if sorted_fitness[idx1] < sorted_fitness[idx2] else sorted_population[idx2]
            
            idx1, idx2 = random.sample(range(len(sorted_population)), 2)
            parent2 = sorted_population[idx1] if sorted_fitness[idx1] < sorted_fitness[idx2] else sorted_population[idx2]
            
            if random.random() < crossover_rate:
                # Crossover: uniform crossover with size adjustment
                child = list(parent1)
                # With some probability, swap some elements with parent2
                for i in range(len(child)):
                    if random.random() < 0.3:
                        if i < len(parent2):
                            child[i] = parent2[i]
                # Ensure no duplicates
                seen = set()
                unique_child = []
                for x in child:
                    if x not in seen:
                        seen.add(x)
                        unique_child.append(x)
                child = unique_child
            else:
                child = list(parent1)
            
            # Mutation
            if random.random() < mutation_rate:
                # Remove a random block
                if len(child) > 0:
                    child.pop(random.randint(0, len(child) - 1))
                # Add a random block
                if len(child) < num_blocks:
                    child.append(random.randint(0, num_blocks - 1))
                    # Remove duplicates
                    seen = set()
                    unique_child = []
                    for x in child:
                        if x not in seen:
                            seen.add(x)
                            unique_child.append(x)
                    child = unique_child
            
            new_population.append(child)
            nu, _ = evaluate_block_set(child)
            new_fitness.append(nu)
        
        population = new_population
        pop_fitness = new_fitness
        
        # Track best
        min_fit = min(pop_fitness)
        min_idx = pop_fitness.index(min_fit)
        if min_fit < best_uncovered:
            best_uncovered = min_fit
            best_solution = list(population[min_idx])
    
    # Phase 3: Local search refinement on best solution
    # Try to remove redundant blocks
    changed = True
    while changed:
        changed = False
        if time.time() > deadline - 5.0:
            break
        
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in best_solution:
            for t_idx in block_cover[b_idx]:
                coverage[t_idx] += 1
        
        to_remove = []
        for i, b_idx in enumerate(best_solution):
            redundant = True
            for t_idx in block_cover[b_idx]:
                if coverage[t_idx] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(i)
        
        if to_remove:
            for i in to_remove:
                b_idx = best_solution[i]
                for t_idx in block_cover[b_idx]:
                    coverage[t_idx] -= 1
            remove_set = set(to_remove)
            best_solution = [b for i, b in enumerate(best_solution) if i not in remove_set]
            changed = True
    
    # Phase 4: Try to add blocks that cover many uncovered t-subsets (if any remain)
    if time.time() < deadline - 5.0:
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in best_solution:
            for t_idx in block_cover[b_idx]:
                coverage[t_idx] += 1
        
        uncovered = np.where(coverage == 0)[0]
        if len(uncovered) > 0:
            # Greedy add
            existing_blocks = set(best_solution)
            while len(uncovered) > 0 and time.time() < deadline - 5.0:
                best_block = -1
                best_gain = 0
                for b_idx in range(num_blocks):
                    if b_idx in existing_blocks:
                        continue
                    gain = 0
                    for t_idx in block_cover[b_idx]:
                        if coverage[t_idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_block = b_idx
                if best_block == -1 or best_gain == 0:
                    break
                best_solution.append(best_block)
                existing_blocks.add(best_block)
                for t_idx in block_cover[best_block]:
                    coverage[t_idx] += 1
                uncovered = np.where(coverage == 0)[0]
    
    # Final redundancy removal
    coverage = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in best_solution:
        for t_idx in block_cover[b_idx]:
            coverage[t_idx] += 1
    
    changed = True
    while changed:
        changed = False
        if time.time() > deadline - 1.0:
            break
        to_remove = []
        for i, b_idx in enumerate(best_solution):
            redundant = True
            for t_idx in block_cover[b_idx]:
                if coverage[t_idx] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(i)
        if to_remove:
            for i in to_remove:
                b_idx = best_solution[i]
                for t_idx in block_cover[b_idx]:
                    coverage[t_idx] -= 1
            remove_set = set(to_remove)
            best_solution = [b for i, b in enumerate(best_solution) if i not in remove_set]
            changed = True
    
    # Convert to output format
    blocks = [list(all_blocks[b_idx]) for b_idx in best_solution]
    return blocks
