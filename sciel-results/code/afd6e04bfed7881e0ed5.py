import numpy as np
from scipy.spatial import ConvexHull
import time

N = 13

def compute_min_triangle_area(points):
    n = points.shape[0]
    # Pre-compute indices once globally if possible, but here we keep it local for clarity
    # However, for performance in GA, we should pre-compute indices outside the fitness function
    # Let's create a helper that uses pre-computed indices
    pass

# Global pre-computation for indices to speed up fitness evaluation
_i, _j, _k = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing='ij')
_mask = (_i < _j) & (_j < _k)
_I = _i[_mask]
_J = _j[_mask]
_K = _k[_mask]

def fast_min_tri_area(points):
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def compute_hull_area(points):
    try:
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def score(points):
    hull_area = compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    scale = 1.0 / np.sqrt(hull_area)
    pts_scaled = points * scale
    return fast_min_tri_area(pts_scaled)

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    np.random.seed(42)
    
    # Genetic Algorithm Parameters
    pop_size = 200
    mutation_rate = 0.1
    mutation_sigma = 0.1
    elite_count = 5
    tournament_size = 5
    generations = 1000 # We will loop until time limit anyway
    
    # Initialize Population
    # Mix of random and structured starts
    population = np.zeros((pop_size, N, 2))
    
    # 1. Random points in unit circle
    for i in range(pop_size // 2):
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        population[i] = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
    # 2. Regular Polygon variations
    for i in range(pop_size // 2, pop_size):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        angles += np.random.uniform(0, 2 * np.pi) # Random rotation
        radii = np.ones(N)
        radii += np.random.normal(0, 0.1, N) # Perturb radii
        population[i] = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    # Evaluate initial population
    fitness = np.array([score(ind) for ind in population])
    
    best_overall_fitness = np.max(fitness)
    best_overall_individual = population[np.argmax(fitness)].copy()
    
    gen = 0
    while time.time() - start_time < time_limit:
        gen += 1
        
        # Sort population by fitness descending
        sorted_indices = np.argsort(fitness)[::-1]
        population = population[sorted_indices]
        fitness = fitness[sorted_indices]
        
        # Update best
        if fitness[0] > best_overall_fitness:
            best_overall_fitness = fitness[0]
            best_overall_individual = population[0].copy()
            
        # Create new population
        new_population = np.zeros((pop_size, N, 2))
        
        # Elitism: Copy best individuals
        for i in range(elite_count):
            new_population[i] = population[i]
            
        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Selection: Tournament
            parents = []
            for _ in range(2):
                contenders = np.random.randint(0, pop_size, tournament_size)
                winner = contenders[np.argmax(fitness[contenders])]
                parents.append(population[winner])
            
            p1, p2 = parents
            
            # Crossover: Uniform
            child = np.zeros((N, 2))
            mask = np.random.rand(N, 2) > 0.5
            child[mask] = p1[mask]
            child[~mask] = p2[~mask]
            
            # Mutation
            if np.random.rand() < mutation_rate:
                # Mutate a random number of points
                num_mutations = np.random.randint(1, 3)
                mut_indices = np.random.choice(N, num_mutations, replace=False)
                for idx in mut_indices:
                    child[idx] += np.random.normal(0, mutation_sigma, 2)
            
            new_population[len(new_population)] = child
            
        # Evaluate new population
        # To save time, we can evaluate only new offspring and keep old fitness for elites?
        # No, simpler to just re-evaluate all, but that's expensive.
        # Let's evaluate only the new ones (indices elite_count to end) and reuse old for elites.
        new_fitness = np.zeros(pop_size)
        new_fitness[:elite_count] = fitness[:elite_count]
        
        for i in range(elite_count, pop_size):
            new_fitness[i] = score(new_population[i])
            
        population = new_population
        fitness = new_fitness
        
        # Adaptive mutation sigma (optional, but helps)
        # If best fitness improves, keep sigma; if not, reduce it slightly?
        # Or keep it fixed for simplicity in this timeframe.
        
    return best_overall_individual
