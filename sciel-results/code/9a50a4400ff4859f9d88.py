"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def partition(adjacency):
    """
    Solve Max-Cut using a Genetic Algorithm (GA).
    
    Approach:
    1. Represent partitions as vectors of +/-1.
    2. Use vectorized fitness evaluation based on matrix operations.
    3. Evolutionary steps: Tournament selection, Uniform crossover, Bit-flip mutation.
    4. Elitism to preserve best solutions.
    
    This is a population-based metaheuristic, distinct from single-trajectory 
    methods like Simulated Annealing.
    """
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # total_weight = sum of all entries in A = 2 * E
    total_weight = np.sum(adj)
    
    # Parameters
    pop_size = 150
    num_generations = 800
    mutation_rate = 0.05
    elite_size = 10
    
    # Initialize population with random +/-1 vectors
    # Shape: (pop_size, n)
    population = np.random.choice([-1, 1], size=(pop_size, n))
    
    def get_fitnesses(pop):
        """
        Vectorized calculation of cut size for a population.
        cut(s) = E - 0.5 * s^T A s
        where E = total_weight / 2
        """
        # temp = pop @ (adj @ pop.T) has shape (pop_size, pop_size)
        # diag(temp) contains s_i^T A s_i
        temp = pop @ (adj @ pop.T)
        diag = np.diag(temp)
        # E = 0.5 * total_weight
        fitnesses = 0.5 * total_weight - 0.5 * diag
        return fitnesses

    # Initial fitness
    fitnesses = get_fitnesses(population)
    
    best_cut = -1
    best_labels = None
    
    for gen in range(num_generations):
        # Sort indices by fitness descending
        sorted_idx = np.argsort(fitnesses)[::-1]
        
        # Track global best
        if fitnesses[sorted_idx[0]] > best_cut:
            best_cut = fitnesses[sorted_idx[0]]
            best_labels = population[sorted_idx[0]]
            
        # Create new population
        new_population = np.empty_like(population)
        
        # Elitism
        new_population[:elite_size] = population[sorted_idx[:elite_size]]
        
        # Fill rest with offspring
        for i in range(elite_size, pop_size):
            # Tournament Selection (size 2)
            idx1, idx2 = np.random.randint(0, pop_size, 2)
            p1 = population[idx1]
            p2 = population[idx2]
            
            # Uniform Crossover
            mask = np.random.rand(n) < 0.5
            child = np.where(mask, p1, p2)
            
            # Mutation (flip sign)
            mut_mask = np.random.rand(n) < mutation_rate
            child[mut_mask] *= -1
            
            new_population[i] = child
            
        population = new_population
        fitnesses = get_fitnesses(population)
        
    # Convert best_labels from +/-1 to 0/1
    # 1 -> 1, -1 -> 0
    return (best_labels + 1) // 2
# EVOLVE-BLOCK-END
