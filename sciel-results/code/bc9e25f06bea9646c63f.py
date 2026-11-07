import numpy as np
from scipy.optimize import minimize

N = 50

def compute_value(h):
    h = np.asarray(h)
    if np.any(h < 0) or not np.all(np.isfinite(h)):
        return -np.inf
    S = np.sum(h)
    Q = np.sum(h**2)
    if S == 0 or Q == 0:
        return -np.inf
    g = np.convolve(h, h, mode='full')
    g_sq_sum = np.sum(g**2)
    cross_sum = np.sum(g[:-1] * g[1:])
    A = 2 * g_sq_sum + cross_sum
    return A / (3 * S**2 * Q)

def construct():
    # Evolutionary Strategy approach
    # Population size
    pop_size = 20
    # Number of iterations
    max_iter = 1000
    # Mutation sigma
    sigma = 0.2
    
    # Initialize population
    # Start with diverse initial points
    pop = []
    # Seed 1: Uniform
    pop.append(np.ones(N))
    # Seed 2: Gaussian
    t = np.linspace(-1, 1, N)
    pop.append(np.exp(-t**2 * 5))
    # Seed 3: Triangular
    pop.append(1 - np.abs(t))
    # Seed 4: Random
    pop.append(np.random.rand(N))
    
    # Fill rest of population with random variations of the best seed so far
    # Or just random
    while len(pop) < pop_size:
        pop.append(np.random.rand(N))
    
    # Normalize initial population
    for i in range(len(pop)):
        pop[i] = pop[i] / np.linalg.norm(pop[i])
    
    # Evaluate initial population
    scores = [compute_value(h) for h in pop]
    
    best_h = pop[np.argmax(scores)]
    best_score = max(scores)
    
    for _ in range(max_iter):
        # Create new population
        new_pop = []
        
        # Elitism: keep the best few
        sort_idx = np.argsort(scores)[::-1]
        for i in range(min(5, pop_size)):
            new_pop.append(pop[sort_idx[i]].copy())
        
        # Generate offspring
        while len(new_pop) < pop_size:
            # Pick parents
            # Use tournament selection or just pick randomly from top half
            p1_idx = np.random.randint(0, max(1, pop_size // 2))
            p2_idx = np.random.randint(0, max(1, pop_size // 2))
            
            p1 = pop[p1_idx]
            p2 = pop[p2_idx]
            
            # Crossover (simple blend)
            alpha = np.random.rand()
            child = alpha * p1 + (1 - alpha) * p2
            
            # Mutation
            # Gaussian noise
            noise = np.random.normal(0, sigma, N)
            child = child + noise
            
            # Ensure non-negative and finite
            child = np.abs(child)
            if not np.all(np.isfinite(child)) or np.sum(child) == 0:
                child = np.random.rand(N)
            
            # Normalize
            child = child / np.linalg.norm(child)
            
            new_pop.append(child)
        
        pop = new_pop
        scores = [compute_value(h) for h in pop]
        
        current_best_idx = np.argmax(scores)
        current_best_score = scores[current_best_idx]
        
        if current_best_score > best_score:
            best_score = current_best_score
            best_h = pop[current_best_idx]
            
        # Adaptive sigma: increase if improvement is small, decrease if large
        # Simple heuristic: if best score improved in last 10 iters, decrease sigma
        # Otherwise increase.
        # For simplicity, we'll just keep sigma constant or slowly decay.
        sigma = max(0.01, sigma * 0.99)
        
    return list(best_h)
