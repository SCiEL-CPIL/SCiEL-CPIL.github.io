import numpy as np
import time

N = 400
POP_SIZE = 100

def _compute_scores_batch(population):
    """Compute max abs autocorrelation for a batch of individuals."""
    n_fft = 1 << (2 * N).bit_length()
    F = np.fft.rfft(population, n=n_fft, axis=1)
    power = F * np.conj(F)
    R_full = np.fft.irfft(power, n=n_fft, axis=1)
    R = R_full[:, 1:N]
    return np.max(np.abs(R), axis=1)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    rng = np.random.default_rng(42)
    
    # Initialize population
    population = rng.choice([-1, 1], size=(POP_SIZE, N)).astype(np.float64)
    scores = _compute_scores_batch(population)
    
    best_score = np.min(scores)
    best_h = population[np.argmin(scores)].copy()
    
    # GA Parameters
    tournament_size = 5
    mutation_prob = 1.0 / N
    generations = 0
    
    while time.time() < deadline:
        # Selection and Reproduction
        new_population = []
        
        # Elitism: Copy the best individual
        best_idx = np.argmin(scores)
        new_population.append(population[best_idx].copy())
        
        # Generate rest of the population
        for _ in range(POP_SIZE - 1):
            # Tournament selection for parent 1
            idxs1 = rng.choice(POP_SIZE, size=tournament_size, replace=False)
            parent1 = population[idxs1][np.argmin(scores[idxs1])]
            
            # Tournament selection for parent 2
            idxs2 = rng.choice(POP_SIZE, size=tournament_size, replace=False)
            parent2 = population[idxs2][np.argmin(scores[idxs2])]
            
            # Uniform Crossover
            mask = rng.random(N) < 0.5
            child = np.where(mask, parent1, parent2)
            
            # Mutation
            flip_mask = rng.random(N) < mutation_prob
            child[flip_mask] *= -1
            
            new_population.append(child)
        
        population = np.array(new_population)
        scores = _compute_scores_batch(population)
        
        # Update best
        current_best_idx = np.argmin(scores)
        if scores[current_best_idx] < best_score:
            best_score = scores[current_best_idx]
            best_h = population[current_best_idx].copy()
            
        generations += 1
        
        # Adaptive mutation: increase if no improvement
        # Simple check: if best_score hasn't changed in last 100 generations, increase mutation
        # We don't track history, so we use a simple heuristic based on time or just keep it constant.
        # For now, constant is safer.
        
    return best_h
