"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def _compute_score(f):
    """
    Computes the score for a given signed step function.
    Metric: Sum of cubes of absolute values of circular autocorrelation coefficients.
    This measures the 'flatness' of the autocorrelation. Lower is better.
    """
    # Compute circular autocorrelation using FFT for efficiency
    # R[k] = sum_i f[i] f[i+k]
    # Using FFT: R = ifft(fft(f) * conj(fft(f)))
    fft_f = np.fft.fft(f)
    power_spectrum = fft_f * np.conj(fft_f)
    autocorr = np.fft.ifft(power_spectrum).real
    
    # The autocorrelation values should be real.
    # We want to minimize the L3 norm of the autocorrelation vector.
    # Note: The DC term (k=0) is always sum(f^2) = N, which is constant.
    # Minimizing the sum of cubes of the non-DC terms is equivalent to minimizing the total sum
    # since the DC term is constant.
    score = np.sum(np.abs(autocorr)**3)
    return score

def construct():
    """
    Construct a signed step function with N=400 equal intervals on [-1/4, 1/4].
    Uses a Genetic Algorithm to minimize the L3 norm of the circular autocorrelation.
    """
    # Parameters
    pop_size = 60
    generations = 200
    mutation_rate = 0.05
    tournament_size = 5
    
    # Initialize population with random signs
    # Values in {-1, 1}
    population = np.array([np.random.choice([-1, 1], size=N) for _ in range(pop_size)])
    
    # Evaluate initial fitness
    fitness = np.array([_compute_score(ind) for ind in population])
    
    best_score = np.min(fitness)
    best_individual = population[np.argmin(fitness)].copy()
    
    for gen in range(generations):
        new_population = []
        
        # Elitism: Keep the best individual
        best_idx = np.argmin(fitness)
        new_population.append(best_individual.copy())
        
        while len(new_population) < pop_size:
            # Tournament Selection
            indices = np.random.choice(pop_size, tournament_size, replace=False)
            winners = indices[np.argmin(fitness[indices])]
            parent1 = population[winners].copy()
            
            indices2 = np.random.choice(pop_size, tournament_size, replace=False)
            winners2 = indices2[np.argmin(fitness[indices2])]
            parent2 = population[winners2].copy()
            
            # Uniform Crossover
            mask = np.random.random(N) < 0.5
            child = np.where(mask, parent1, parent2)
            
            # Mutation
            mutation_mask = np.random.random(N) < mutation_rate
            child[mutation_mask] *= -1
            
            new_population.append(child)
        
        population = np.array(new_population)
        
        # Evaluate new fitness
        fitness = np.array([_compute_score(ind) for ind in population])
        
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_score:
            best_score = fitness[current_best_idx]
            best_individual = population[current_best_idx].copy()
            
    return best_individual
# EVOLVE-BLOCK-END
