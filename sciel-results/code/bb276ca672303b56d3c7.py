"""Genetic algorithm with vectorized scoring for minimizing monochromatic 4-APs."""
import numpy as np
import time

def construct():
    N = 256
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Precompute all 4-term arithmetic progressions as index tuples (0-based)
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Convert to numpy array for fast vectorized operations
    aps_arr = np.array(aps, dtype=np.int32)
    n_aps = len(aps_arr)
    
    # Split into 4 separate index arrays for vectorized access
    i1 = aps_arr[:, 0]
    i2 = aps_arr[:, 1]
    i3 = aps_arr[:, 2]
    i4 = aps_arr[:, 3]
    
    def count_mono_4ap(colors):
        """Count monochromatic 4-APs using vectorized numpy operations."""
        c1 = colors[i1]
        c2 = colors[i2]
        c3 = colors[i3]
        c4 = colors[i4]
        # Monochromatic means all four are equal
        return np.sum((c1 == c2) & (c2 == c3) & (c3 == c4))
    
    # Initialize population with diverse starting points
    np.random.seed(42)
    pop_size = 200
    
    # Start with various structured initializations for diversity
    population = []
    
    # Block patterns of various sizes
    for block_size in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64]:
        pattern = np.zeros(N, dtype=np.int8)
        for i in range(N):
            pattern[i] = (i // block_size) % 2
        population.append(pattern.copy())
    
    # Random initializations
    while len(population) < pop_size:
        population.append(np.random.randint(0, 2, N).astype(np.int8))
    
    # Evaluate initial population
    scores = np.array([count_mono_4ap(pop) for pop in population], dtype=np.float64)
    best_idx = np.argmin(scores)
    best_score = scores[best_idx]
    best_pop = population[best_idx].copy()
    
    # Genetic algorithm
    mutation_rate = 0.02
    crossover_rate = 0.8
    
    while time.time() < deadline:
        # Sort by fitness
        order = np.argsort(scores)
        population = [population[i] for i in order]
        scores = scores[order]
        
        best_score = scores[0]
        best_pop = population[0].copy()
        
        # Create next generation
        new_population = []
        
        # Elitism: keep top individuals
        elite_count = max(2, pop_size // 10)
        for i in range(elite_count):
            new_population.append(population[i].copy())
        
        # Fill rest with offspring
        while len(new_population) < pop_size and time.time() < deadline:
            # Tournament selection
            def select():
                t1 = np.random.randint(0, pop_size)
                t2 = np.random.randint(0, pop_size)
                return population[t1] if scores[t1] <= scores[t2] else population[t2]
            
            parent1 = select()
            parent2 = select()
            
            # Crossover (uniform)
            if np.random.random() < crossover_rate:
                mask = np.random.randint(0, 2, N).astype(np.int8)
                child = np.where(mask == 1, parent1, parent2).astype(np.int8)
            else:
                child = parent1.copy()
            
            # Mutation
            if np.random.random() < 0.9:
                num_mutations = np.random.poisson(N * mutation_rate)
                if num_mutations > 0:
                    mut_positions = np.random.choice(N, size=min(num_mutations, N), replace=False)
                    child[mut_positions] = 1 - child[mut_positions]
            
            new_population.append(child)
        
        # Evaluate new population
        population = new_population
        scores = np.array([count_mono_4ap(pop) for pop in population], dtype=np.float64)
        
        # Update best
        current_best_idx = np.argmin(scores)
        if scores[current_best_idx] < best_score:
            best_score = scores[current_best_idx]
            best_pop = population[current_best_idx].copy()
    
    # Convert to list of ints as required
    return [int(c) for c in best_pop]
