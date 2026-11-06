import time
import math
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    N_COEFFS = 24
    SQRT_N = math.sqrt(N_COEFFS)
    
    # Fast evaluation using FFT with golden-section refinement
    def evaluate(coeffs):
        # coeffs is a list/array of +1/-1
        N = 4096  # Higher resolution for better peak location
        c = np.zeros(N, dtype=np.complex128)
        c[:len(coeffs)] = coeffs
        vals = np.fft.fft(c)
        mags = np.abs(vals)
        
        # Find top few peaks to be safe
        top_k = 5
        top_indices = np.argsort(mags)[-top_k:][::-1]
        
        # Golden section refinement for each top peak
        def refine_peak(a, b, coeffs):
            gr = (math.sqrt(5) - 1) / 2
            c_pt = b - gr * (b - a)
            d_pt = a + gr * (b - a)
            fc = np.abs(np.sum(coeffs * np.exp(1j * np.arange(len(coeffs)) * c_pt)))
            fd = np.abs(np.sum(coeffs * np.exp(1j * np.arange(len(coeffs)) * d_pt)))
            
            for _ in range(30):  # More iterations for accuracy
                if fc < fd:
                    a = c_pt
                    c_pt = d_pt
                    fc = fd
                    d_pt = a + gr * (b - a)
                    fd = np.abs(np.sum(coeffs * np.exp(1j * np.arange(len(coeffs)) * d_pt)))
                else:
                    b = d_pt
                    d_pt = c_pt
                    fd = fc
                    c_pt = b - gr * (b - a)
                    fc = np.abs(np.sum(coeffs * np.exp(1j * np.arange(len(coeffs)) * c_pt)))
            return max(fc, fd)
        
        best_refined = 0.0
        for idx in top_indices:
            theta = 2 * np.pi * idx / N
            # Refine in a window around this peak
            a = theta - np.pi / N
            b = theta + np.pi / N
            refined = refine_peak(a, b, coeffs)
            if refined > best_refined:
                best_refined = refined
        
        return best_refined
    
    # Generate Rudin-Shapiro polynomial
    def get_rs_poly(n, sign_seed=1):
        p, q = [sign_seed], [sign_seed]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return p[:n]
    
    # Initialize diverse population
    def init_population(pop_size):
        pop = []
        # Add several Rudin-Shapiro variants with different starting signs
        for s in [1, -1]:
            pop.append(get_rs_poly(N_COEFFS, s))
        # Add some random individuals
        while len(pop) < pop_size:
            pop.append([random.choice([1, -1]) for _ in range(N_COEFFS)])
        return pop
    
    # Genetic Algorithm
    POP_SIZE = 60
    
    # Initialize population
    population = init_population(POP_SIZE)
    
    # Evaluate initial population
    scores = [evaluate(ind) / SQRT_N for ind in population]
    
    # Track best
    best_idx = np.argmin(scores)
    best_coeffs = population[best_idx][:]
    best_score = scores[best_idx]
    
    # GA parameters
    tournament_size = 3
    crossover_prob = 0.8
    mutation_rate = 0.1  # Per bit
    
    def tournament_select(pop, scores):
        candidates = random.sample(range(len(pop)), tournament_size)
        best_cand = min(candidates, key=lambda i: scores[i])
        return pop[best_cand][:]
    
    def crossover(parent1, parent2):
        # Uniform crossover
        child = []
        for i in range(N_COEFFS):
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child
    
    def mutate(individual):
        child = individual[:]
        for i in range(N_COEFFS):
            if random.random() < mutation_rate:
                child[i] = -child[i]
        return child
    
    generation = 0
    while time.time() - start_time < deadline:
        # Check time more frequently
        if time.time() - start_time > deadline - 2:
            break
        
        new_population = []
        new_scores = []
        
        # Elitism: keep the best individual
        best_idx_gen = np.argmin(scores)
        new_population.append(population[best_idx_gen][:])
        new_scores.append(scores[best_idx_gen])
        
        # Generate rest of population
        while len(new_population) < POP_SIZE:
            if time.time() - start_time > deadline - 1:
                break
            
            # Selection
            parent1 = tournament_select(population, scores)
            parent2 = tournament_select(population, scores)
            
            # Crossover
            if random.random() < crossover_prob:
                child = crossover(parent1, parent2)
            else:
                child = parent1[:]
            
            # Mutation
            child = mutate(child)
            
            # Evaluate
            score = evaluate(child) / SQRT_N
            
            new_population.append(child)
            new_scores.append(score)
            
            # Update best
            if score < best_score:
                best_score = score
                best_coeffs = child[:]
        
        # Replace population
        population = new_population
        scores = new_scores
        generation += 1
        
        # Occasionally add some random individuals to maintain diversity
        if generation % 5 == 0 and time.time() - start_time < deadline - 5:
            # Replace a few worst individuals with random ones
            for _ in range(5):
                worst_idx = np.argmax(scores)
                population[worst_idx] = [random.choice([1, -1]) for _ in range(N_COEFFS)]
                scores[worst_idx] = evaluate(population[worst_idx]) / SQRT_N
    
    return best_coeffs
# EVOLVE-BLOCK-END
