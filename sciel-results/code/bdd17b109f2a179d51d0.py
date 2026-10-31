"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    # Approach: Memetic Algorithm (Genetic Algorithm + Local Search)
    # This uses population-based evolution with crossover and mutation,
    # followed by individual local refinement (greedy bit-flip).
    
    rng = np.random.default_rng(42)
    fft_size = 2 * N
    
    def calc_cost(seq):
        f = np.fft.fft(seq, fft_size)
        acf = np.fft.ifft(f * np.conj(f)).real[:N]
        return np.sum(acf[1:] ** 2)

    def local_polish(h):
        # Greedy 1-pass local search to polish an individual
        current_cost = calc_cost(h)
        for i in range(N):
            h[i] = -h[i]
            new_cost = calc_cost(h)
            if new_cost < current_cost:
                current_cost = new_cost
            else:
                h[i] = -h[i]
        return h, current_cost

    # Initialize population
    pop_size = 50
    population = [rng.choice([-1.0, 1.0], size=N) for _ in range(pop_size)]
    
    # Initial local polish for each individual
    polished_population = []
    for h in population:
        h, cost = local_polish(h)
        polished_population.append((cost, h))
        
    # Sort by cost (ascending)
    polished_population.sort(key=lambda x: x[0])
    
    best_cost = polished_population[0][0]
    best_h = polished_population[0][1].copy()
    
    start_time = time.time()
    time_limit = 100.0 # Leave 20s buffer for safety
    
    # GA Loop
    while time.time() - start_time < time_limit:
        new_population = []
        
        # Elitism: keep the best few
        for i in range(3):
            new_population.append(polished_population[i])
            
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection (random pick from current population)
            a_idx = rng.integers(0, pop_size)
            b_idx = rng.integers(0, pop_size)
            
            h1 = polished_population[a_idx][1]
            h2 = polished_population[b_idx][1]
            
            # Uniform Crossover
            mask = rng.random(N) > 0.5
            child = np.where(mask, h1, h2)
            
            # Mutation (bit flip)
            mut_mask = rng.random(N) < 0.02
            child[mut_mask] = -child[mut_mask]
            
            # Local Polish
            child, cost = local_polish(child)
            new_population.append((cost, child))
            
        # Update population
        polished_population = new_population
        polished_population.sort(key=lambda x: x[0])
        
        # Update global best
        if polished_population[0][0] < best_cost:
            best_cost = polished_population[0][0]
            best_h = polished_population[0][1].copy()

    return best_h
# EVOLVE-BLOCK-END
