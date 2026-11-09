"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np
import time

# EVOLVE-BLOCK-START

def _fitness_from_mask(mask):
    """
    Compute the number of distinct d > 0 such that a, a+d, a+2d are all in the set.
    Uses bitmask for efficient evaluation.
    """
    count = 0
    # Max d is 255 because max difference in 1..512 is 511, and a+2d <= 512 implies 2d <= 511
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _list_to_mask(lst):
    mask = 0
    for x in lst:
        mask |= (1 << (x - 1))
    return mask

def _mask_to_list(mask):
    return [i + 1 for i in range(512) if (mask >> i) & 1]

def _random_mask():
    """Generate a random 32-element subset of 1..512 as a bitmask."""
    indices = random.sample(range(512), 32)
    mask = 0
    for i in indices:
        mask |= (1 << i)
    return mask

def _structured_seed_mask():
    """
    Generate a structured seed based on known good patterns.
    """
    # Try different step sizes for the base AP
    step = random.choice([2, 3, 4, 5, 6, 7, 8, 10, 12, 16])
    start = random.randint(1, 512 - 31 * step)
    base = [start + i * step for i in range(32)]
    
    # Perturb some elements
    for i in range(32):
        if random.random() < 0.4:
            base[i] += random.randint(-10, 10)
            base[i] = max(1, min(512, base[i]))
    
    # Ensure uniqueness and convert to mask
    unique_vals = set()
    mask = 0
    for x in base:
        if 1 <= x <= 512 and x not in unique_vals:
            unique_vals.add(x)
            mask |= (1 << (x - 1))
    
    # Fill if needed
    while mask.bit_count() < 32:
        x = random.randint(1, 512)
        if not (mask & (1 << (x - 1))):
            mask |= (1 << (x - 1))
            
    return mask

def _crossover_mask(parent1, parent2):
    """
    Uniform crossover on bitmasks.
    """
    # Randomly select bits from parent1 or parent2
    # This might result in a mask with != 32 bits, so we need to repair
    mask = 0
    for i in range(512):
        if random.random() < 0.5:
            if (parent1 >> i) & 1:
                mask |= (1 << i)
        else:
            if (parent2 >> i) & 1:
                mask |= (1 << i)
    
    # Repair: if bit_count != 32, add or remove random bits
    current_count = mask.bit_count()
    if current_count > 32:
        # Remove random bits
        indices = [i for i in range(512) if (mask >> i) & 1]
        to_remove = random.sample(indices, current_count - 32)
        for i in to_remove:
            mask &= ~(1 << i)
    elif current_count < 32:
        # Add random bits
        while mask.bit_count() < 32:
            i = random.randint(0, 511)
            mask |= (1 << i)
            
    return mask

def _mutate_mask(mask, mutation_rate=0.1):
    """
    Mutation: flip random bits.
    """
    # Determine how many bits to flip based on mutation rate
    num_flips = int(32 * mutation_rate)
    if num_flips == 0 and random.random() < mutation_rate:
        num_flips = 1
        
    for _ in range(num_flips):
        # Pick a bit to flip
        # To maintain population size, we should swap a bit in with a bit out
        # But simple flip might change size. Let's do a swap mutation.
        # Pick an index in the set and an index not in the set
        in_indices = [i for i in range(512) if (mask >> i) & 1]
        if not in_indices:
            continue
        out_indices = [i for i in range(512) if not ((mask >> i) & 1)]
        if not out_indices:
            continue
            
        i_in = random.choice(in_indices)
        i_out = random.choice(out_indices)
        
        mask &= ~(1 << i_in)
        mask |= (1 << i_out)
        
    return mask

def _local_search_mask(mask, max_iters=50):
    """
    Greedy local search: try to improve by swapping one element.
    Instead of random swaps, we evaluate all possible swaps? No, 32*480 is too many.
    We will try random swaps but with a "greedy" bias:
    Actually, let's try a more directed approach:
    1. Calculate fitness.
    2. For each element x in the set, try removing it and adding a random y.
    3. Accept if fitness increases.
    
    To be faster, we will do a limited number of random swaps per iteration,
    but we will run more iterations.
    """
    best_mask = mask
    best_fit = _fitness_from_mask(mask)
    
    for it in range(max_iters):
        improved = False
        # Try 10 random swaps per iteration
        for _ in range(10):
            # Pick a random element to remove
            in_indices = [i for i in range(512) if (best_mask >> i) & 1]
            if not in_indices:
                break
            i_remove = random.choice(in_indices)
            
            # Pick a random element to add
            out_indices = [i for i in range(512) if not ((best_mask >> i) & 1)]
            if not out_indices:
                break
            i_add = random.choice(out_indices)
            
            new_mask = (best_mask & ~(1 << i_remove)) | (1 << i_add)
            new_fit = _fitness_from_mask(new_mask)
            
            if new_fit > best_fit:
                best_fit = new_fit
                best_mask = new_mask
                improved = True
                # We could break here to restart the "try 10" loop, or just continue
                # Continuing allows finding multiple improvements in one pass
        if not improved:
            break
            
    return best_mask, best_fit

def construct():
    """
    Run a genetic algorithm with local search refinement.
    """
    random.seed(42)
    np.random.seed(42)
    
    population_size = 60
    generations = 100
    elite_size = 10
    mutation_rate = 0.15
    
    # Initialize population
    population = []
    for _ in range(10):
        population.append(_structured_seed_mask())
    while len(population) < population_size:
        population.append(_random_mask())
    
    fitnesses = [_fitness_from_mask(m) for m in population]
    
    best_idx = max(range(population_size), key=lambda i: fitnesses[i])
    best_mask = population[best_idx]
    best_fitness = fitnesses[best_idx]
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    for gen in range(generations):
        if time.time() - start_time > time_limit:
            break
            
        # Sort population by fitness
        paired = sorted(zip(fitnesses, population), key=lambda x: -x[0])
        fitnesses = [p[0] for p in paired]
        population = [p[1] for p in paired]
        
        # Local search on the top 5 individuals
        for i in range(min(5, population_size)):
            ls_mask, ls_fit = _local_search_mask(population[i], max_iters=20)
            if ls_fit > fitnesses[i]:
                population[i] = ls_mask
                fitnesses[i] = ls_fit
            if ls_fit > best_fitness:
                best_fitness = ls_fit
                best_mask = ls_mask
                
        # Keep elite
        new_population = [population[i] for i in range(elite_size)]
        
        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection
            def tournament():
                indices = random.sample(range(len(population)), 3)
                best_idx_t = max(indices, key=lambda i: fitnesses[i])
                return population[best_idx_t]
            
            parent1 = tournament()
            parent2 = tournament()
            child = _crossover_mask(parent1, parent2)
            child = _mutate_mask(child, mutation_rate)
            new_population.append(child)
        
        population = new_population
        fitnesses = [_fitness_from_mask(m) for m in population]
        
        # Update best from population
        current_best_idx = max(range(population_size), key=lambda i: fitnesses[i])
        if fitnesses[current_best_idx] > best_fitness:
            best_fitness = fitnesses[current_best_idx]
            best_mask = population[current_best_idx]
            
        # Early stopping if we hit a theoretical max? 
        if best_fitness == 255:
            break
    
    # Final intensive local search on the best individual
    final_mask, final_fit = _local_search_mask(best_mask, max_iters=100)
    if final_fit > best_fitness:
        best_fitness = final_fit
        best_mask = final_mask
    
    return sorted(_mask_to_list(best_mask))

# EVOLVE-BLOCK-END
