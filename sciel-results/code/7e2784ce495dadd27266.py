"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np

# EVOLVE-BLOCK-START
def _fitness_batch(population):
    """
    Compute fitness for a batch of individuals using NumPy.
    population: list of lists of integers.
    Returns: list of fitness scores.
    """
    n_pop = len(population)
    if n_pop == 0:
        return []
    
    # Create a boolean matrix of shape (n_pop, 512)
    # pop_matrix[i, x-1] = True if x is in population[i]
    pop_matrix = np.zeros((n_pop, 512), dtype=bool)
    for i, ind in enumerate(population):
        for x in ind:
            pop_matrix[i, x-1] = True
            
    # Initialize scores
    scores = np.zeros(n_pop, dtype=int)
    
    # For each possible difference d
    for d in range(1, 256):
        # We need to check if there exists a such that:
        # pop_matrix[i, a-1] AND pop_matrix[i, a+d-1] AND pop_matrix[i, a+2d-1]
        # Indices: a-1, a+d-1, a+2d-1
        # Max a is 512 - 2d. So a-1 goes up to 511 - 2d.
        # Let's slice the matrix.
        # col1: 0 to 511-2d
        # col2: d to 511-d
        # col3: 2d to 511
        
        max_a_idx = 511 - 2 * d
        if max_a_idx < 0:
            continue
            
        slice1 = pop_matrix[:, 0 : max_a_idx + 1]
        slice2 = pop_matrix[:, d : max_a_idx + 1 + d]
        slice3 = pop_matrix[:, 2 * d : max_a_idx + 1 + 2 * d]
        
        # Check if any column has all three True
        # any_col = np.any(slice1 & slice2 & slice3, axis=1)
        # If any_col[i] is True, then d is a valid difference for individual i.
        # We only want to count it once per individual.
        valid = np.any(slice1 & slice2 & slice3, axis=1)
        scores += valid
        
    return scores.tolist()

def _random_individual():
    """Generate a random 32-element subset of 1..512."""
    return random.sample(range(1, 513), 32)

def _structured_seed():
    """
    Generate a structured seed based on union of arithmetic progressions.
    This creates clusters which generate many local differences and some cross differences.
    """
    # Strategy: Split 32 elements into k clusters.
    # Each cluster is an AP.
    k = random.choice([2, 3, 4])
    sizes = [32 // k] * k
    remainder = 32 % k
    for i in range(remainder):
        sizes[i] += 1
        
    elements = []
    used = set()
    
    # Place clusters in different parts of the range to allow cross-diffs
    # Or place them close?
    # If clusters are far, cross diffs are large and distinct.
    # If clusters are close, cross diffs overlap with local diffs?
    # Let's try random placement of cluster starts.
    
    for size in sizes:
        if size == 0: continue
        start = random.randint(1, 512 - size * 10) # Leave gap
        step = random.randint(1, 5)
        cluster = [start + i * step for i in range(size)]
        
        # Ensure uniqueness and range
        valid_cluster = []
        for x in cluster:
            if 1 <= x <= 512 and x not in used:
                valid_cluster.append(x)
                used.add(x)
        
        elements.extend(valid_cluster)
        
    # Fill remaining if needed
    while len(elements) < 32:
        x = random.randint(1, 512)
        if x not in used:
            elements.append(x)
            used.add(x)
            
    return elements[:32]

def _crossover(parent1, parent2):
    """
    Uniform crossover on lists of 32 integers.
    """
    child = []
    for i in range(32):
        if random.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    
    child_set = set(child)
    if len(child_set) < 32:
        unique_child = list(child_set)
        while len(unique_child) < 32:
            new_val = random.randint(1, 512)
            if new_val not in child_set:
                child_set.add(new_val)
                unique_child.append(new_val)
        return unique_child
    else:
        return child

def _mutate(individual, mutation_rate=0.1):
    """
    Mutation: replace random elements with new random unique values.
    """
    individual = list(individual)
    ind_set = set(individual)
    
    for i in range(32):
        if random.random() < mutation_rate:
            old_val = individual[i]
            ind_set.remove(old_val)
            
            while True:
                new_val = random.randint(1, 512)
                if new_val not in ind_set:
                    ind_set.add(new_val)
                    individual[i] = new_val
                    break
                    
    return individual

def _local_search_numpy(individual, max_iters=200):
    """
    Hill climbing using single individual evaluation (fast enough for 1 individual).
    """
    def _fitness_single(lst):
        mask = 0
        for x in lst:
            mask |= (1 << (x - 1))
        count = 0
        for d in range(1, 256):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    best_ind = list(individual)
    best_fit = _fitness_single(best_ind)
    ind_set = set(best_ind)
    
    for it in range(max_iters):
        improved = False
        for _ in range(50):
            i = random.randint(0, 31)
            old_val = best_ind[i]
            
            while True:
                new_val = random.randint(1, 512)
                if new_val not in ind_set:
                    break
            
            # Temporarily swap
            temp_ind = list(best_ind)
            temp_ind[i] = new_val
            new_fit = _fitness_single(temp_ind)
            
            if new_fit > best_fit:
                best_fit = new_fit
                best_ind = temp_ind
                ind_set.remove(old_val)
                ind_set.add(new_val)
                improved = True
        
        if not improved:
            break
            
    return best_ind, best_fit

def construct():
    """
    Run a genetic algorithm with batch fitness evaluation and local search.
    """
    random.seed(42)
    np.random.seed(42)
    
    population_size = 60
    generations = 300
    elite_size = 10
    mutation_rate = 0.15
    
    # Initialize population
    population = []
    for _ in range(10):
        population.append(_structured_seed())
    while len(population) < population_size:
        population.append(_random_individual())
    
    # Calculate initial fitnesses
    fitnesses = _fitness_batch(population)
    
    best_idx = max(range(population_size), key=lambda i: fitnesses[i])
    best_individual = population[best_idx]
    best_fitness = fitnesses[best_idx]
    
    for gen in range(generations):
        # Sort population by fitness
        paired = sorted(zip(fitnesses, population), key=lambda x: -x[0])
        fitnesses = [p[0] for p in paired]
        population = [p[1] for p in paired]
        
        # Local search on the best individual
        ls_individual, ls_fitness = _local_search_numpy(population[0], max_iters=50)
        if ls_fitness > best_fitness:
            best_fitness = ls_fitness
            best_individual = ls_individual
            population[0] = ls_individual
            fitnesses[0] = ls_fitness
        else:
            # Update population[0] fitness if it changed? No, we only updated best.
            # But population[0] might have been improved by LS even if not global best?
            # Actually, if ls_fitness <= best_fitness, we don't update best.
            # But we should still put the improved individual back if it's better than the old population[0]?
            # The old population[0] had fitnesses[0].
            if ls_fitness > fitnesses[0]:
                population[0] = ls_individual
                fitnesses[0] = ls_fitness

        # Keep elite
        new_population = [list(population[i]) for i in range(elite_size)]
        
        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection
            indices = random.sample(range(len(population)), 3)
            best_idx_t = max(indices, key=lambda i: fitnesses[i])
            parent1 = population[best_idx_t]
            
            indices2 = random.sample(range(len(population)), 3)
            best_idx_t2 = max(indices2, key=lambda i: fitnesses[i])
            parent2 = population[best_idx_t2]
            
            child = _crossover(parent1, parent2)
            child = _mutate(child, mutation_rate)
            new_population.append(child)
        
        population = new_population
        fitnesses = _fitness_batch(population)
        
        # Update best from population
        current_best_idx = max(range(population_size), key=lambda i: fitnesses[i])
        if fitnesses[current_best_idx] > best_fitness:
            best_fitness = fitnesses[current_best_idx]
            best_individual = population[current_best_idx]
            
        # Early stopping
        if best_fitness > 200: # Heuristic max
            break
    
    # Final intensive local search
    final_ind, final_fit = _local_search_numpy(best_individual, max_iters=300)
    if final_fit > best_fitness:
        best_fitness = final_fit
        best_individual = final_ind
    
    return sorted(best_individual)

# EVOLVE-BLOCK-END
