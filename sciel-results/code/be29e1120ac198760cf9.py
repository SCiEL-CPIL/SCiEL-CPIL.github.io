"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np

# EVOLVE-BLOCK-START
def _fitness(lst):
    """
    Compute the number of distinct d > 0 such that a, a+d, a+2d are all in the set.
    Uses bitmask for efficient evaluation.
    """
    mask = 0
    for x in lst:
        mask |= (1 << (x - 1))
    
    count = 0
    # Max d is 255
    for d in range(1, 256):
        # Check if intersection of mask, mask>>d, mask>>2d is non-empty
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _fitness_incremental(mask, changed_old, changed_new):
    """
    Quick fitness check: only recompute d values that could be affected.
    A change at position x affects d where x, x±d, x±2d are in range.
    For simplicity, we still do full check but this is a placeholder for optimization.
    """
    return _fitness_from_mask(mask)

def _fitness_from_mask(mask):
    """Compute fitness directly from bitmask."""
    count = 0
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _random_individual():
    """Generate a random 32-element subset of 1..512."""
    return random.sample(range(1, 513), 32)

def _structured_seed():
    """
    Generate a structured seed based on known good patterns.
    """
    r = random.random()
    if r < 0.3:
        # Dense block
        start = random.randint(1, 480)
        return [start + i for i in range(32)]
    elif r < 0.6:
        # Two blocks
        start1 = random.randint(1, 200)
        start2 = random.randint(300, 480)
        res = [start1 + i for i in range(16)] + [start2 + i for i in range(16)]
        return res
    else:
        # Random
        return _random_individual()

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

def _local_search_exhaustive(individual, max_iters=20):
    """
    Exhaustive 1-move local search.
    Tries every possible replacement for every element.
    """
    best_ind = list(individual)
    best_mask = 0
    for x in best_ind:
        best_mask |= (1 << (x - 1))
    best_fit = _fitness_from_mask(best_mask)
    ind_set = set(best_ind)
    
    for _ in range(max_iters):
        improved = False
        best_move_i = -1
        best_move_new_val = -1
        best_move_fit = best_fit
        best_move_new_mask = 0
        
        # Try replacing each element
        for i in range(32):
            old_val = best_ind[i]
            old_bit = 1 << (old_val - 1)
            
            # Iterate through all possible new values
            for new_val in range(1, 513):
                if new_val in ind_set:
                    continue
                
                new_bit = 1 << (new_val - 1)
                temp_mask = (best_mask & ~old_bit) | new_bit
                new_fit = _fitness_from_mask(temp_mask)
                
                if new_fit > best_move_fit:
                    best_move_fit = new_fit
                    best_move_i = i
                    best_move_new_val = new_val
                    best_move_new_mask = temp_mask
        
        if best_move_i != -1:
            old_val = best_ind[best_move_i]
            ind_set.remove(old_val)
            ind_set.add(best_move_new_val)
            best_ind[best_move_i] = best_move_new_val
            best_mask = best_move_new_mask
            best_fit = best_move_fit
            improved = True
            
        if not improved:
            break
            
    return best_ind, best_fit

def _simulated_annealing(individual, start_temp=10.0, end_temp=0.1, max_iters=200):
    """
    Simulated annealing to escape local optima.
    """
    current_ind = list(individual)
    current_mask = 0
    for x in current_ind:
        current_mask |= (1 << (x - 1))
    current_fit = _fitness_from_mask(current_mask)
    ind_set = set(current_ind)
    
    best_ind = list(current_ind)
    best_mask = current_mask
    best_fit = current_fit
    
    for it in range(max_iters):
        temp = start_temp * (end_temp / start_temp) ** (it / max_iters)
        
        i = random.randint(0, 31)
        old_val = current_ind[i]
        
        while True:
            new_val = random.randint(1, 512)
            if new_val not in ind_set:
                break
        
        old_bit = 1 << (old_val - 1)
        new_bit = 1 << (new_val - 1)
        
        temp_mask = (current_mask & ~old_bit) | new_bit
        new_fit = _fitness_from_mask(temp_mask)
        
        delta = new_fit - current_fit
        
        if delta > 0 or random.random() < np.exp(delta / temp):
            # Accept
            current_mask = temp_mask
            ind_set.remove(old_val)
            ind_set.add(new_val)
            current_ind[i] = new_val
            current_fit = new_fit
            
            if current_fit > best_fit:
                best_fit = current_fit
                best_ind = list(current_ind)
                best_mask = current_mask
    
    return best_ind, best_fit

def construct():
    """
    Run a genetic algorithm with exhaustive local search refinement.
    """
    random.seed(42)
    np.random.seed(42)
    
    population_size = 50
    generations = 100
    elite_size = 10
    mutation_rate = 0.1
    
    # Initialize population with mix of random and structured seeds
    population = []
    # Add structured seeds
    for _ in range(10):
        population.append(_structured_seed())
    # Add random individuals
    while len(population) < population_size:
        population.append(_random_individual())
    
    fitnesses = [_fitness(ind) for ind in population]
    
    best_idx = max(range(population_size), key=lambda i: fitnesses[i])
    best_individual = population[best_idx]
    best_fitness = fitnesses[best_idx]
    
    for gen in range(generations):
        # Sort population by fitness
        paired = sorted(zip(fitnesses, population), key=lambda x: -x[0])
        fitnesses = [p[0] for p in paired]
        population = [p[1] for p in paired]
        
        # Local search on the best individual every 5 generations
        if gen % 5 == 0:
            ls_individual, ls_fitness = _local_search_exhaustive(population[0], max_iters=5)
            if ls_fitness > best_fitness:
                best_fitness = ls_fitness
                best_individual = ls_individual
                population[0] = ls_individual
                fitnesses[0] = ls_fitness
        
        # Keep elite
        new_population = [list(population[i]) for i in range(elite_size)]
        
        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection
            def tournament():
                indices = random.sample(range(len(population)), 3)
                best_idx_t = max(indices, key=lambda i: fitnesses[i])
                return population[best_idx_t]
            
            parent1 = tournament()
            parent2 = tournament()
            child = _crossover(parent1, parent2)
            child = _mutate(child, mutation_rate)
            new_population.append(child)
        
        population = new_population
        fitnesses = [_fitness(ind) for ind in population]
        
        # Update best from population
        current_best_idx = max(range(population_size), key=lambda i: fitnesses[i])
        if fitnesses[current_best_idx] > best_fitness:
            best_fitness = fitnesses[current_best_idx]
            best_individual = population[current_best_idx]
            
        # Early stopping if we hit a theoretical max? 
        if best_fitness == 255:
            break
    
    # Final intensive local search on the best individual
    final_ind, final_fit = _local_search_exhaustive(best_individual, max_iters=20)
    if final_fit > best_fitness:
        best_fitness = final_fit
        best_individual = final_ind
    
    return sorted(best_individual)

# EVOLVE-BLOCK-END
