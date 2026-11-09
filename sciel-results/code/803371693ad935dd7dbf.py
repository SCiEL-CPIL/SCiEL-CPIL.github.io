import numpy as np
import time
import random

def construct():
    N = 24
    # Precompute the matrix Z for evaluation
    # Dense sampling to approximate max modulus
    num_samples = 4096
    angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
    k_vals = np.arange(N)
    Z = np.exp(1j * np.outer(k_vals, angles)) # Shape (N, num_samples)
    
    def eval_score(c):
        # c is shape (N,)
        p_vals = Z.T @ c
        return np.max(np.abs(p_vals))

    def local_search(c):
        # Hill climbing on single bit flips with randomized order
        c = c.copy()
        current_score = eval_score(c)
        improved = True
        while improved:
            improved = False
            indices = list(range(N))
            random.shuffle(indices)
            for i in indices:
                old_val = c[i]
                c[i] = -old_val
                new_score = eval_score(c)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    c[i] = old_val
        return c, current_score

    # GA Parameters
    pop_size = 100
    generations = 5000
    mutation_rate = 0.1
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer

    # Initialize Population
    population = []
    scores = []
    for _ in range(pop_size):
        c = np.random.choice([-1.0, 1.0], size=N)
        c, score = local_search(c)
        population.append(c)
        scores.append(score)
    
    best_score = min(scores)
    best_coeffs = population[scores.index(best_score)].copy()

    # GA Loop
    gen = 0
    while gen < generations:
        if time.time() - start_time > time_limit:
            break
            
        # Selection: Tournament Selection (size 2)
        def select():
            i1 = random.randint(0, pop_size - 1)
            i2 = random.randint(0, pop_size - 1)
            if scores[i1] < scores[i2]:
                return population[i1]
            else:
                return population[i2]
        
        parent1 = select()
        parent2 = select()
        
        # Crossover: Uniform Crossover
        child = np.zeros(N)
        for i in range(N):
            if random.random() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        
        # Mutation: Bit Flip
        for i in range(N):
            if random.random() < mutation_rate:
                child[i] = -child[i]
        
        # Local Search on Child
        child, child_score = local_search(child)
        
        # Replacement: Replace worst individual if child is better
        worst_idx = np.argmax(scores)
        if child_score < scores[worst_idx]:
            population[worst_idx] = child
            scores[worst_idx] = child_score
            if child_score < best_score:
                best_score = child_score
                best_coeffs = child.copy()
                
        gen += 1

    return [int(x) for x in best_coeffs]
