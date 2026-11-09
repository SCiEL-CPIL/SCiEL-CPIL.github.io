# EVOLVE-BLOCK-START
import numpy as np

def _makespan(processing_times, order):
    """Compute makespan for a given job order."""
    if len(order) == 0:
        return 0.0
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    # Use a vectorized approach where possible, but the dependency chain makes it tricky.
    # C[i][j] = max(C[i-1][j], C[i][j-1]) + P[i][j]
    # This is essentially the critical path.
    
    # We can compute this row by row or machine by machine.
    # C[i] depends on C[i-1] and C[i][0..j-1].
    # Let's stick to the loop but optimize inner steps if possible.
    # For N=100, M=10, this is 1000 ops. Very fast.
    
    completion = np.zeros((n_jobs, n_machines))
    for i, job_idx in enumerate(order):
        for j in range(n_machines):
            p = processing_times[job_idx, j]
            if j == 0:
                if i == 0:
                    completion[i, j] = p
                else:
                    completion[i, j] = completion[i-1, j] + p
            else:
                if i == 0:
                    completion[i, j] = completion[i, j-1] + p
                else:
                    completion[i, j] = max(completion[i-1, j], completion[i, j-1]) + p
    
    return completion[-1, -1]

def _pmx_crossover(parent1, parent2):
    """Partially Mapped Crossover for permutations."""
    n = len(parent1)
    if n < 2:
        return list(parent1), list(parent2)
    
    # Choose two cut points
    i1, i2 = sorted(np.random.choice(n, 2, replace=False))
    
    # Create children
    child1 = [None] * n
    child2 = [None] * n
    
    # Copy segment from parent1 to child1 and parent2 to child2
    for i in range(i1, i2 + 1):
        child1[i] = parent1[i]
        child2[i] = parent2[i]
    
    # Fill the rest of child1
    # The segment from parent2 at [i1, i2] is the mapping
    map1 = {parent1[i]: parent2[i] for i in range(i1, i2 + 1)}
    
    # Positions to fill in child1
    positions = [i for i in range(n) if i < i1 or i > i2]
    values_to_fill = [v for v in parent2 if v not in child1]
    
    # We need to fill child1 positions [0..i1-1] and [i2+1..n-1] with values from parent2
    # that are not in the copied segment.
    # The order should be preserved from parent2.
    
    # Let's use the standard PMX algorithm more carefully.
    # 1. Copy segment.
    # 2. For each position outside segment in child1:
    #    - Take value from parent2 at that position.
    #    - If it's in the mapping (i.e., conflicts with child1 segment), replace with mapped value.
    #    - Repeat until no conflict.
    
    for pos in positions:
        val = parent2[pos]
        while val in child1:
            val = map1[val]
        child1[pos] = val
        
    # Same for child2
    map2 = {parent2[i]: parent1[i] for i in range(i1, i2 + 1)}
    for pos in positions:
        val = parent1[pos]
        while val in child2:
            val = map2[val]
        child2[pos] = val
            
    return child1, child2

def _swap_mutation(individual, rate=0.1):
    """Swap mutation: swap two random elements with probability rate."""
    n = len(individual)
    if n < 2:
        return individual
    
    # With probability rate, perform a swap
    if np.random.random() < rate:
        i, j = np.random.choice(n, 2, replace=False)
        individual[i], individual[j] = individual[j], individual[i]
        
    return individual

def _tournament_select(population, fitnesses, k=3):
    """Tournament selection."""
    indices = np.random.choice(len(population), k, replace=False)
    best_idx = indices[np.argmin(fitnesses[indices])]
    return population[best_idx]

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using Genetic Algorithm.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Parameters
    pop_size = max(20, n_jobs) # Population size proportional to problem size
    generations = 100
    mutation_rate = 0.1
    tournament_k = 3
    
    # Initialize population
    population = []
    # Add a heuristic start (LPT) to help convergence
    lpt_order = list(np.argsort(-processing_times.sum(axis=1)))
    population.append(lpt_order)
    
    # Add random permutations
    rng = np.random.RandomState(42)
    for _ in range(pop_size - 1):
        population.append(list(rng.permutation(n_jobs)))
    
    # Calculate initial fitnesses
    fitnesses = np.array([_makespan(processing_times, ind) for ind in population])
    
    best_order = population[np.argmin(fitnesses)]
    best_fitness = np.min(fitnesses)
    
    for gen in range(generations):
        new_population = []
        
        # Elitism: keep the best individual
        new_population.append(best_order.copy())
        
        while len(new_population) < pop_size:
            # Selection
            p1 = _tournament_select(population, fitnesses, tournament_k)
            p2 = _tournament_select(population, fitnesses, tournament_k)
            
            # Crossover
            c1, c2 = _pmx_crossover(p1, p2)
            
            # Mutation
            c1 = _swap_mutation(c1, mutation_rate)
            c2 = _swap_mutation(c2, mutation_rate)
            
            new_population.append(c1)
            if len(new_population) < pop_size:
                new_population.append(c2)
        
        population = new_population
        fitnesses = np.array([_makespan(processing_times, ind) for ind in population])
        
        # Update best
        current_best_idx = np.argmin(fitnesses)
        if fitnesses[current_best_idx] < best_fitness:
            best_fitness = fitnesses[current_best_idx]
            best_order = population[current_best_idx].copy()
            
    return best_order
# EVOLVE-BLOCK-END
