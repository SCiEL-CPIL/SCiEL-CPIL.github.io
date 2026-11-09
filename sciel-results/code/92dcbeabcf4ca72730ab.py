"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np

# EVOLVE-BLOCK-START
def _fitness(population_mask):
    """
    Compute the number of distinct d > 0 such that there exists a
    with a, a+d, a+2d all in the set represented by population_mask.
    
    population_mask: a 512-bit integer or numpy uint64 array representing membership.
    For efficiency, we use a set of elements.
    """
    # Extract elements from mask
    # We'll represent the set as a Python set for clarity, but use bit operations for speed
    # Actually, let's use a boolean array or set
    elements = set()
    m = population_mask
    # This is slow if done naively. Let's use a different approach.
    # Since we're in a GA, let's keep the population as lists of elements during evolution
    # and only compute fitness when needed.
    # But the directive says "leveraging bitmasks for efficient fitness evaluation"
    # So let's use a bitmask as an integer and precompute something.
    
    # Alternative: use a set for the population and compute distinct differences.
    # For a 32-element set, the number of pairs is 32*31/2 = 496.
    # For each pair (x, y) with x < y, d = y - x.
    # We need to check if x + 2d is in the set.
    # So for each d, we need to know if there's an a such that a, a+d, a+2d are in the set.
    # This is equivalent to: for each d, check if the set intersects with (set - d) and (set - 2d) in a way that forms a progression.
    # More simply: for each d in 1..255 (max d is (512-1)/2 = 255.5, so d <= 255),
    #   for each a in set such that a + 2d <= 512:
    #       if a+d in set and a+2d in set: count d and break.
    
    # But we want the number of distinct d. So we can iterate d from 1 to 255 and check.
    # For a given d, we need to check if there exists a such that a, a+d, a+2d are all in the set.
    # This can be done by: for each a in set, if a+2d <= 512 and (a+d) in set and (a+2d) in set.
    
    # To make this fast, let's use a set for membership.
    # The population is given as a mask, so let's convert it to a set once.
    
    # Actually, for the GA, it's better to keep the population as a list of 32 integers.
    # Let me redefine: the GA will work with lists of 32 distinct integers.
    # The fitness function will take such a list and compute the score.
    
    pass  # This function is not used as-is; see _fitness_from_set below

def _fitness_from_set(s):
    """
    s: a set of 32 distinct integers in 1..512
    Returns the number of distinct d > 0 such that a, a+d, a+2d are all in s for some a.
    """
    distinct_d = set()
    # Max possible d: if a >= 1 and a+2d <= 512, then d <= (512-1)//2 = 255
    for d in range(1, 256):
        found = False
        for a in s:
            if a + 2 * d > 512:
                continue
            if (a + d) in s and (a + 2 * d) in s:
                found = True
                break
        if found:
            distinct_d.add(d)
    return len(distinct_d)

def _random_individual():
    """Generate a random 32-element subset of 1..512."""
    return set(random.sample(range(1, 513), 32))

def _crossover(parent1, parent2):
    """
    Crossover: uniform crossover on the sets.
    For each element in 1..512, include it in the child with probability 0.5 from parent1 or parent2.
    But we need exactly 32 elements. So we do:
    - Combine both parents, then sample 32 from the union? No, that loses structure.
    - Better: take elements from parent1 and parent2 with some probability, then fill in from the union or randomly.
    
    Let's do: for each element in 1..512, with prob 0.5 take from parent1, with prob 0.5 from parent2.
    But this may not give 32 elements. So we adjust:
    - Start with an empty set.
    - For each element in the union of parent1 and parent2, add it with probability 0.5.
    - If we have more than 32, randomly remove extras.
    - If we have less than 32, randomly add from 1..512 not in the set.
    """
    union = parent1 | parent2
    child = set()
    for x in union:
        if random.random() < 0.5:
            child.add(x)
    
    # Adjust size to 32
    if len(child) > 32:
        # Remove random extras
        extras = list(child)
        random.shuffle(extras)
        for x in extras[32:]:
            child.discard(x)
    elif len(child) < 32:
        # Add random elements from 1..512 not in child
        available = [x for x in range(1, 513) if x not in child]
        needed = 32 - len(child)
        for x in random.sample(available, needed):
            child.add(x)
    
    return child

def _mutate(individual, mutation_rate=0.1):
    """
    Mutation: for each element in the individual, with probability mutation_rate,
    replace it with a random element from 1..512 not in the individual.
    """
    individual = set(individual)
    for x in list(individual):
        if random.random() < mutation_rate:
            individual.discard(x)
            available = [y for y in range(1, 513) if y not in individual]
            if available:
                individual.add(random.choice(available))
    return individual

def construct():
    """
    Run a genetic algorithm to find a 32-element subset of 1..512
    that maximizes the number of distinct 3-term progression differences.
    """
    random.seed(42)  # For reproducibility
    
    population_size = 50
    generations = 200
    elite_size = 5
    mutation_rate = 0.1
    
    # Initialize population
    population = [_random_individual() for _ in range(population_size)]
    fitnesses = [_fitness_from_set(ind) for ind in population]
    
    best_individual = max(population, key=lambda x: _fitness_from_set(x))
    best_fitness = _fitness_from_set(best_individual)
    
    for gen in range(generations):
        # Sort population by fitness (descending)
        paired = sorted(zip(fitnesses, population), key=lambda x: -x[0])
        fitnesses = [p[0] for p in paired]
        population = [p[1] for p in paired]
        
        # Keep elite
        new_population = [set(population[i]) for i in range(elite_size)]
        
        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection
            def tournament():
                indices = random.sample(range(len(population)), 3)
                best_idx = max(indices, key=lambda i: fitnesses[i])
                return population[best_idx]
            
            parent1 = tournament()
            parent2 = tournament()
            child = _crossover(parent1, parent2)
            child = _mutate(child, mutation_rate)
            new_population.append(child)
        
        population = new_population
        fitnesses = [_fitness_from_set(ind) for ind in population]
        
        # Update best
        current_best = max(fitnesses)
        if current_best > best_fitness:
            best_fitness = current_best
            best_individual = population[fitnesses.index(current_best)]
    
    return sorted(best_individual)

# EVOLVE-BLOCK-END
