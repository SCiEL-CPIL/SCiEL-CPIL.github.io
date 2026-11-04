import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 145s limit, 5s margin for safety
    
    # Precompute all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(t_subsets)
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute coverage matrix: for each block, which t-subsets it covers
    # Using a list of sets for flexibility, but we'll use a boolean matrix for speed
    # Coverage matrix: shape (n_blocks, n_subsets) - but this is 77520 * 3876 ~ 300M booleans, too big
    # Instead, for each block, store the list of t-subset indices it covers
    block_covered_subsets = []
    for b in all_blocks:
        covered = set()
        for sub in itertools.combinations(b, t):
            # Need index of sub in t_subsets
            # This is slow, so let's precompute a dict
            pass
    
    # Faster precomputation: map each t-subset to its index
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    block_covered_indices = []
    for b in all_blocks:
        idxs = []
        for sub in itertools.combinations(b, t):
            idxs.append(subset_to_idx[sub])
        block_covered_indices.append(idxs)
    
    # Convert to numpy arrays for fast union operations
    # We'll use a different approach: for a given list of blocks, compute the union of their covered subsets
    # Using a bitset would be ideal, but n_subsets=3876 fits in a 60-word bitset or we can use a boolean array
    
    # For fitness evaluation, we need to count how many t-subsets are covered by a set of blocks
    # We'll use a boolean array of size n_subsets
    def evaluate_fitness(block_indices):
        """Count how many t-subsets are covered by the given block indices."""
        covered = np.zeros(n_subsets, dtype=bool)
        for idx in block_indices:
            # Mark all subsets covered by this block
            covered[block_covered_indices[idx]] = True
        return covered.sum()
    
    # Greedy repair: add blocks to cover remaining subsets
    def greedy_repair(block_indices, max_blocks=None):
        """Add greedy blocks to cover all remaining t-subsets."""
        if max_blocks is None:
            max_blocks = 200
        current_blocks = list(block_indices)
        covered = np.zeros(n_subsets, dtype=bool)
        for idx in current_blocks:
            covered[block_covered_indices[idx]] = True
        
        uncovered_mask = ~covered
        while uncovered_mask.any() and len(current_blocks) < max_blocks:
            if time.time() > deadline:
                break
            # Find the block that covers the most uncovered subsets
            best_gain = -1
            best_idx = -1
            # Sample some blocks to speed up
            sample_size = min(1000, n_blocks)
            sample_indices = random.sample(range(n_blocks), sample_size)
            for idx in sample_indices:
                # Count uncovered subsets covered by this block
                covered_by_block = block_covered_indices[idx]
                # Count how many of these are in uncovered_mask
                gain = sum(uncovered_mask[i] for i in covered_by_block)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                elif gain == best_gain and gain > 0 and random.random() < 0.1:
                    best_idx = idx
            
            if best_idx == -1 or best_gain == 0:
                break
            
            current_blocks.append(best_idx)
            covered[block_covered_indices[best_idx]] = True
            uncovered_mask = ~covered
        
        return current_blocks
    
    # Genetic Algorithm
    population_size = 50
    max_generations = 200
    mutation_rate = 0.1
    crossover_rate = 0.7
    
    # Initialize population with random sets of blocks
    # Start with a reasonable number of blocks, e.g., 150
    target_size = 150
    
    population = []
    for _ in range(population_size):
        # Start with a random subset of blocks
        n_initial = random.randint(100, 160)
        blocks = random.sample(range(n_blocks), n_initial)
        population.append(blocks)
    
    best_fitness = -1
    best_individual = None
    
    for gen in range(max_generations):
        if time.time() > deadline:
            break
        
        # Evaluate fitness
        fitnesses = []
        for individual in population:
            fitness = evaluate_fitness(individual)
            fitnesses.append(fitness)
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual[:]
        
        # Check if we have a complete covering
        if best_fitness == n_subsets:
            break
        
        # Create next generation
        next_population = []
        
        # Elitism: keep the best individuals
        sorted_indices = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
        elite_count = max(5, population_size // 10)
        for i in range(elite_count):
            next_population.append(population[sorted_indices[i]][:])
        
        # Fill the rest with offspring
        while len(next_population) < population_size:
            if time.time() > deadline:
                break
            
            # Selection: tournament selection
            def tournament_select(pop, fit, k=3):
                candidates = random.sample(range(len(pop)), min(k, len(pop)))
                best_cand = max(candidates, key=lambda i: fit[i])
                return pop[best_cand]
            
            if random.random() < crossover_rate:
                parent1 = tournament_select(population, fitnesses)
                parent2 = tournament_select(population, fitnesses)
                
                # Crossover: uniform crossover
                child = []
                for i in range(max(len(parent1), len(parent2))):
                    if i < len(parent1) and random.random() < 0.5:
                        child.append(parent1[i])
                    elif i < len(parent2):
                        child.append(parent2[i])
                
                # Remove duplicates
                child = list(set(child))
            else:
                parent = tournament_select(population, fitnesses)
                child = parent[:]
            
            # Mutation: randomly add or remove blocks
            if random.random() < mutation_rate:
                if random.random() < 0.5 and child:
                    # Remove a block
                    idx = random.randint(0, len(child) - 1)
                    child.pop(idx)
                else:
                    # Add a block
                    child.append(random.randint(0, n_blocks - 1))
            
            next_population.append(child)
        
        population = next_population
    
    # Final repair: use greedy to complete the best individual
    if best_individual is None:
        best_individual = random.sample(range(n_blocks), target_size)
    
    # Repair the best individual
    repaired = greedy_repair(best_individual, max_blocks=200)
    
    # Check if repaired is complete
    covered = np.zeros(n_subsets, dtype=bool)
    for idx in repaired:
        covered[block_covered_indices[idx]] = True
    
    if covered.all():
        # Convert back to actual blocks
        result_blocks = [all_blocks[idx] for idx in repaired]
        return [list(b) for b in result_blocks]
    
    # If not complete, do a full greedy from scratch as fallback
    uncovered = np.ones(n_subsets, dtype=bool)
    blocks = []
    while uncovered.any():
        if time.time() > deadline:
            break
        best_gain = -1
        best_idx = -1
        for idx in range(n_blocks):
            covered_by_block = block_covered_indices[idx]
            gain = sum(uncovered[i] for i in covered_by_block)
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx == -1:
            break
        blocks.append(best_idx)
        uncovered[block_covered_indices[best_idx]] = False
    
    if blocks:
        result_blocks = [all_blocks[idx] for idx in blocks]
        return [list(b) for b in result_blocks]
    
    return []
