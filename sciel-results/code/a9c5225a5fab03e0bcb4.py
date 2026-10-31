import numpy as np
import time

def construct():
    N = 16
    D = 2
    n_vars = N * D
    
    # Objective function: Minimize (max_d / min_d)^2
    # Returns a large number if points are too close (degenerate)
    def objective(x):
        pts = x.reshape(N, D)
        # Compute pairwise distances
        # Using broadcasting for vectorization
        dists = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=2))
        # Get upper triangle indices to avoid duplicates and self-distances
        iu = np.triu_indices(N, 1)
        pairwise_dists = dists[iu]
        
        min_d = np.min(pairwise_dists)
        max_d = np.max(pairwise_dists)
        
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d) ** 2

    # Genetic Algorithm Parameters
    pop_size = 60
    elite_size = 5
    max_generations = 2000
    time_limit = 110.0 # Leave 10s buffer for overhead
    
    # Initialize Population
    # Mix of random and structured initializers to improve exploration
    def init_individual(seed=None):
        if seed is None:
            seed = np.random.randint(0, 100000)
        rng = np.random.RandomState(seed)
        # Random uniform in [-1, 1]
        x = rng.uniform(-1, 1, n_vars)
        # Center it
        pts = x.reshape(N, D)
        pts -= np.mean(pts, axis=0)
        return pts.flatten()

    population = []
    # Add some diverse starts
    for i in range(10):
        population.append(init_individual(i))
    # Fill rest with random
    for i in range(pop_size - 10):
        population.append(init_individual())
    
    # Evaluate initial population
    fitnesses = np.array([objective(ind) for ind in population])
    
    best_score = np.min(fitnesses)
    best_x = population[np.argmin(fitnesses)]
    
    start_time = time.time()
    gen = 0
    
    # GA Loop
    while gen < max_generations:
        if time.time() - start_time > time_limit:
            break
            
        # Sort population by fitness
        sorted_indices = np.argsort(fitnesses)
        population = [population[i] for i in sorted_indices]
        fitnesses = fitnesses[sorted_indices]
        
        # Update best
        if fitnesses[0] < best_score:
            best_score = fitnesses[0]
            best_x = population[0]
            
        # Elitism: Keep top elite_size
        new_population = population[:elite_size]
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament Selection
            def tournament_select(pop, fit, k=3):
                indices = np.random.choice(len(pop), k, replace=False)
                best_idx = indices[np.argmin(fit[indices])]
                return pop[best_idx]
            
            parent1 = tournament_select(population, fitnesses)
            parent2 = tournament_select(population, fitnesses)
            
            # BLX-alpha Crossover
            alpha = 0.5
            child = np.zeros_like(parent1)
            for i in range(len(parent1)):
                low = min(parent1[i], parent2[i])
                high = max(parent1[i], parent2[i])
                range_val = high - low
                child[i] = np.random.uniform(low - alpha * range_val, high + alpha * range_val)
            
            # Mutation
            # Adaptive mutation rate: start high, decay
            mutation_rate = 0.2 * (1 - gen / max_generations) + 0.01
            mutation_strength = 0.1 * (1 - gen / max_generations) + 0.01
            
            mask = np.random.random(len(child)) < mutation_rate
            if mask.any():
                child[mask] += np.random.normal(0, mutation_strength, mask.sum())
            
            # Center the child to reduce search space (translation invariance)
            pts = child.reshape(N, D)
            pts -= np.mean(pts, axis=0)
            child = pts.flatten()
            
            new_population.append(child)
            
        # Evaluate new population
        population = new_population
        fitnesses = np.array([objective(ind) for ind in population])
        
        gen += 1
        
    # Final check
    final_pts = best_x.reshape(N, D)
    final_pts -= np.mean(final_pts, axis=0)
    
    return final_pts
