# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    start_time = time.time()
    deadline = start_time + 600  # Safety margin before 630s limit
    
    n_points = 14
    dim = 3
    pop_size = 50
    elite_size = 5
    
    # Initialize population
    def init_point():
        # Random point on unit sphere
        p = np.random.randn(dim)
        p /= np.linalg.norm(p)
        return p
    
    def init_individual():
        return np.array([init_point() for _ in range(n_points)])
    
    # Initial population
    population = [init_individual() for _ in range(pop_size)]
    
    # Evaluate initial population
    scores = [_score(ind) for ind in population]
    
    # Sort by score
    indices = np.argsort(scores)[::-1]
    population = [population[i] for i in indices]
    scores = [scores[i] for i in indices]
    
    best_points = population[0].copy()
    best_score = scores[0]
    
    # Genetic Algorithm Loop
    mutation_rate = 0.2
    mutation_scale = 0.1
    crossover_rate = 0.8
    
    while time.time() < deadline - 10:
        new_population = []
        
        # Elitism: keep the best individuals
        for i in range(elite_size):
            new_population.append(population[i].copy())
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 5:
                break
            
            # Selection: Tournament selection
            def tournament_select(pop, scores, k=3):
                candidates = np.random.randint(0, len(pop), k)
                best_idx = candidates[np.argmax([scores[i] for i in candidates])]
                return pop[best_idx].copy()
            
            parent1 = tournament_select(population, scores)
            parent2 = tournament_select(population, scores)
            
            if np.random.rand() < crossover_rate:
                # Crossover: Blend crossover
                alpha = np.random.rand()
                child = alpha * parent1 + (1 - alpha) * parent2
            else:
                child = parent1.copy()
            
            # Mutation
            if np.random.rand() < mutation_rate:
                # Mutate a random number of points
                n_mut = np.random.randint(1, 4)
                for _ in range(n_mut):
                    idx = np.random.randint(n_points)
                    child[idx] += np.random.randn(dim) * mutation_scale
                    # Normalize to unit sphere to maintain scale
                    norm = np.linalg.norm(child[idx])
                    if norm > 0:
                        child[idx] /= norm
            
            new_population.append(child)
        
        # Evaluate new population
        new_scores = [_score(ind) for ind in new_population]
        
        # Combine and sort
        all_population = new_population
        all_scores = new_scores
        
        indices = np.argsort(all_scores)[::-1]
        population = [all_population[i] for i in indices]
        scores = [all_scores[i] for i in indices]
        
        # Update best
        if scores[0] > best_score:
            best_score = scores[0]
            best_points = population[0].copy()
            
            # Adaptive mutation scale: decrease if we're improving
            mutation_scale *= 0.95
            if mutation_scale < 0.001:
                mutation_scale = 0.001
        
        # Occasionally re-inject diversity
        if np.random.rand() < 0.1:
            population[-1] = init_individual()
            scores[-1] = _score(population[-1])
            indices = np.argsort(scores)[::-1]
            population = [population[i] for i in indices]
            scores = [scores[i] for i in indices]
            if scores[0] > best_score:
                best_score = scores[0]
                best_points = population[0].copy()

    return best_points

# EVOLVE-BLOCK-END
