"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(f):
    """
    Computes the score for a given signed step function.
    Metric: Sum of cubes of absolute values of circular autocorrelation sidelobes.
    Excludes the DC term (k=0) as it is constant.
    Lower is better.
    """
    fft_f = np.fft.fft(f)
    power_spectrum = fft_f * np.conj(fft_f)
    autocorr = np.fft.ifft(power_spectrum).real
    
    # Exclude DC term (index 0)
    score = np.sum(np.abs(autocorr[1:])**3)
    return score

def _local_search(f, max_time=30.0):
    """
    Perform hill-climbing local search to minimize the score.
    """
    start_time = time.time()
    current_score = _compute_score(f)
    
    while (time.time() - start_time) < max_time:
        improved = False
        for i in range(N):
            f[i] *= -1
            new_score = _compute_score(f)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                f[i] *= -1 # Revert
        if not improved:
            break
            
    return f, current_score

def construct():
    """
    Construct a signed step function with N=400 equal intervals on [-1/4, 1/4].
    Uses a Genetic Algorithm followed by Local Search to minimize the L3 norm of the circular autocorrelation sidelobes.
    """
    start_time = time.time()
    time_limit = 115.0 # Leave 5s buffer
    
    # Parameters
    pop_size = 80
    mutation_rate = 0.05
    tournament_size = 5
    
    # Initialize population with random signs
    population = np.array([np.random.choice([-1, 1], size=N) for _ in range(pop_size)])
    
    # Evaluate initial fitness
    fitness = np.array([_compute_score(ind) for ind in population])
    
    best_score = np.min(fitness)
    best_individual = population[np.argmin(fitness)].copy()
    
    # GA Phase
    while (time.time() - start_time) < (time_limit - 35.0):
        new_population = []
        
        # Elitism
        best_idx = np.argmin(fitness)
        new_population.append(best_individual.copy())
        
        while len(new_population) < pop_size:
            # Tournament Selection
            indices = np.random.choice(pop_size, tournament_size, replace=False)
            winner1 = indices[np.argmin(fitness[indices])]
            parent1 = population[winner1].copy()
            
            indices2 = np.random.choice(pop_size, tournament_size, replace=False)
            winner2 = indices2[np.argmin(fitness[indices2])]
            parent2 = population[winner2].copy()
            
            # Uniform Crossover
            mask = np.random.random(N) < 0.5
            child = np.where(mask, parent1, parent2)
            
            # Mutation
            mutation_mask = np.random.random(N) < mutation_rate
            child[mutation_mask] *= -1
            
            new_population.append(child)
        
        population = np.array(new_population)
        fitness = np.array([_compute_score(ind) for ind in population])
        
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_score:
            best_score = fitness[current_best_idx]
            best_individual = population[current_best_idx].copy()
            
    # Local Search Phase
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5.0:
        best_individual, best_score = _local_search(best_individual, max_time=remaining_time - 5.0)
        
    return best_individual
# EVOLVE-BLOCK-END
