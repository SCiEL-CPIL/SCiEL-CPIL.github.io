import numpy as np
import time

N = 400
DEADLINE = 120.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    The signed step function has N equal intervals on [-1/4, 1/4].
    C3 = max_t |autoconv(t)| where autoconv is the convolution of f with itself.
    """
    M = 4 * N  # fine grid
    dx = (0.5) / M  # domain length is 0.5
    
    # Build f on fine grid using vectorized repeat
    f_fine = np.repeat(signed_step, M // N)
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    # The autoconvolution is defined on [-0.5, 0.5]
    # C3 = max |autoconv|
    c3 = np.max(np.abs(autoconv))
    
    return c3

def _evaluate_population(pop):
    """Evaluate a population of step functions, returning their C3 values."""
    scores = np.zeros(len(pop))
    for i in range(len(pop)):
        scores[i] = _compute_c3(pop[i])
    return scores

def _tournament_select(pop, scores, k=3):
    """Tournament selection: pick k random individuals, return the best."""
    idxs = np.random.randint(0, len(pop), size=k)
    best_idx = idxs[np.argmin(scores[idxs])]
    return pop[best_idx].copy()

def _crossover(parent1, parent2, crossover_rate=0.8):
    """Single-point crossover with mutation."""
    if np.random.random() > crossover_rate:
        return parent1.copy()
    
    point = np.random.randint(1, N)
    child = np.concatenate([parent1[:point], parent2[point:]])
    return child

def _mutate(individual, mutation_rate=0.01):
    """Bit-flip mutation."""
    child = individual.copy()
    mask = np.random.random(N) < mutation_rate
    child[mask] = -child[mask]
    return child

def construct():
    start_time = time.time()
    
    # Initialize population
    pop_size = 64
    np.random.seed(42)
    
    # Start with all ones, then add random diversity
    population = np.zeros((pop_size, N), dtype=np.int8)
    population[0] = 1  # seed
    for i in range(1, pop_size):
        population[i] = np.random.choice([-1, 1], size=N)
    
    # Initial evaluation
    scores = _evaluate_population(population)
    best_idx = np.argmin(scores)
    best_step = population[best_idx].copy()
    best_c3 = scores[best_idx]
    
    # GA parameters
    generations = 0
    no_improve_count = 0
    patience = 15  # generations without improvement
    
    while time.time() - start_time < DEADLINE - 5.0:
        generations += 1
        
        # Check deadline
        if time.time() - start_time > DEADLINE - 3.0:
            break
        
        # Selection and reproduction
        new_population = np.zeros((pop_size, N), dtype=np.int8)
        
        # Elitism: keep the best individual
        new_population[0] = best_step
        
        for i in range(1, pop_size):
            # Tournament selection for two parents
            parent1 = _tournament_select(population, scores, k=3)
            parent2 = _tournament_select(population, scores, k=3)
            
            # Crossover
            child = _crossover(parent1, parent2, crossover_rate=0.8)
            
            # Mutation
            child = _mutate(child, mutation_rate=0.01)
            
            new_population[i] = child
        
        population = new_population
        scores = _evaluate_population(population)
        
        # Update best
        gen_best_idx = np.argmin(scores)
        if scores[gen_best_idx] < best_c3:
            best_c3 = scores[gen_best_idx]
            best_step = population[gen_best_idx].copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # If stuck, inject some random diversity
        if no_improve_count >= patience:
            n_inject = np.random.randint(3, 10)
            for _ in range(n_inject):
                idx = np.random.randint(pop_size)
                population[idx] = np.random.choice([-1, 1], size=N)
            scores = _evaluate_population(population)
            gen_best_idx = np.argmin(scores)
            if scores[gen_best_idx] < best_c3:
                best_c3 = scores[gen_best_idx]
                best_step = population[gen_best_idx].copy()
            no_improve_count = 0
        
        # Periodic local search polish on the best individual
        if generations % 5 == 0 and time.time() - start_time < DEADLINE - 10.0:
            # Quick local search on best_step
            local_step = best_step.copy()
            local_c3 = best_c3
            for _ in range(100):
                if time.time() - start_time > DEADLINE - 5.0:
                    break
                idx = np.random.randint(N)
                local_step[idx] = -local_step[idx]
                new_c3 = _compute_c3(local_step)
                if new_c3 < local_c3:
                    local_c3 = new_c3
                else:
                    local_step[idx] = -local_step[idx]
            
            if local_c3 < best_c3:
                best_c3 = local_c3
                best_step = local_step.copy()
                # Inject improved individual into population
                population[np.random.randint(pop_size)] = best_step
    
    return best_step
