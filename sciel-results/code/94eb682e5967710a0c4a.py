import numpy as np
import time
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3)

# Precompute indices for all unique triples (i, j, k) with i < j < k
_triples = []
for i in range(N - 2):
    for j in range(i + 1, N - 1):
        for k in range(j + 1, N):
            _triples.append((i, j, k))
_TRIPLES = np.array(_triples, dtype=np.int32)

def _min_area_bary(p):
    """
    p: array of shape (N, 3) with barycentric coordinates.
    Returns the minimum normalized triangle area.
    """
    r1 = p[_TRIPLES[:, 0]]
    r2 = p[_TRIPLES[:, 1]]
    r3 = p[_TRIPLES[:, 2]]
    
    # dets = det([r1, r2, r3]) = r1 dot (r2 cross r r3)
    # In barycentric coords, area is 0.5 * |det|
    # We compute the signed determinant for gradient purposes
    cross_r2_r3 = np.cross(r2, r3)
    dets = np.einsum('ij,ij->i', r1, cross_r2_r3)
    
    areas = 0.5 * np.abs(dets)
    return np.min(areas)

def _bary_to_cart(bary):
    """Convert barycentric (u,v,w) to Cartesian (x,y)."""
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = (SQRT3 / 2) * bary[:, 2]
    return np.column_stack([x, y])

def _project_bary(bary):
    """
    Project barycentric coordinates to the simplex.
    """
    b = bary.copy()
    b_sum = np.sum(b, axis=1, keepdims=True)
    b_sum[b_sum == 0] = 1.0
    b = b / b_sum
    b = np.clip(b, 0, 1)
    b_sum = np.sum(b, axis=1, keepdims=True)
    b_sum[b_sum == 0] = 1.0
    b = b / b_sum
    return b

def _ga_fitness(bary):
    """
    Compute fitness for a single individual (barycentric coords).
    Returns the minimum triangle area (higher is better).
    """
    bary = _project_bary(bary)
    return _min_area_bary(bary)

def _ga_crossover(parent1, parent2, rng):
    """
    Uniform crossover between two individuals.
    Returns two children.
    """
    mask = rng.random(N, 3) > 0.5
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return child1, child2

def _ga_mutation(individual, sigma, rng):
    """
    Gaussian mutation.
    """
    noise = rng.normal(0, sigma, size=individual.shape)
    return individual + noise

def _ga_evolve(population, fitnesses, rng, sigma=0.1, tournament_size=5, 
               elite_count=2, max_generations=100, deadline=None):
    """
    Evolve the population using a genetic algorithm.
    Returns the best individual found.
    """
    best_idx = np.argmax(fitnesses)
    best_individual = population[best_idx].copy()
    best_fitness = fitnesses[best_idx]
    
    for gen in range(max_generations):
        if deadline is not None and time.time() > deadline:
            break
            
        # Tournament selection
        new_population = []
        
        # Keep elites
        elite_indices = np.argsort(fitnesses)[-elite_count:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # Generate offspring
        pop_size = len(population)
        while len(new_population) < pop_size:
            # Select parents via tournament
            p1_idx = np.random.choice(pop_size, tournament_size)
            p2_idx = np.random.choice(pop_size, tournament_size)
            p1 = population[p1_idx[np.argmax(fitnesses[p1_idx])]]
            p2 = population[p2_idx[np.argmax(fitnesses[p2_idx])]]
            
            # Crossover
            child1, child2 = _ga_crossover(p1, p2, rng)
            
            # Mutation
            child1 = _ga_mutation(child1, sigma, rng)
            child2 = _ga_mutation(child2, sigma, rng)
            
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
        
        population = np.array(new_population)
        
        # Evaluate fitness
        fitnesses = np.array([_ga_fitness(ind) for ind in population])
        
        # Update best
        best_idx = np.argmax(fitnesses)
        if fitnesses[best_idx] > best_fitness:
            best_fitness = fitnesses[best_idx]
            best_individual = population[best_idx].copy()
    
    return best_individual, best_fitness

def _structured_init_bary():
    """
    Generate structured initial populations for GA.
    """
    inits = []
    
    # Random Dirichlet samples
    for seed in range(5):
        r = np.random.default_rng(seed)
        inits.append(r.dirichlet([1, 1, 1], size=N))
    
    # Symmetric init with corners
    corners = np.array([
        [0.85, 0.075, 0.075],
        [0.075, 0.85, 0.075],
        [0.075, 0.075, 0.85]
    ])
    r = np.random.default_rng(99)
    rest = r.dirichlet([1, 1, 1], size=N-3)
    inits.append(np.vstack([corners, rest]))
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_area = 0.0
    best_bary = None
    
    rng = np.random.default_rng(42)
    
    # Generate initial population from multiple structured inits
    inits = _structured_init_bary()
    
    # Create initial population by sampling from inits with perturbation
    pop_size = 40
    population = []
    for init in inits:
        for _ in range(pop_size // len(inits) + 1):
            # Add small random perturbation
            perturbed = init + rng.normal(0, 0.05, size=(N, 3))
            perturbed = _project_bary(perturbed)
            population.append(perturbed)
        if len(population) >= pop_size:
            break
    
    population = np.array(population[:pop_size])
    
    # Evaluate initial fitness
    fitnesses = np.array([_ga_fitness(ind) for ind in population])
    
    # Run GA with adaptive mutation
    sigma_start = 0.15
    sigma_end = 0.02
    
    # Multiple GA runs with different parameters
    ga_configs = [
        {'sigma': sigma_start, 'tournament_size': 5, 'elite_count': 2, 'max_generations': 50},
        {'sigma': 0.08, 'tournament_size': 3, 'elite_count': 3, 'max_generations': 40},
        {'sigma': 0.05, 'tournament_size': 4, 'elite_count': 2, 'max_generations': 30},
    ]
    
    for config in ga_configs:
        if time.time() > deadline - 10.0:
            break
        
        # Reset population for each GA run
        current_population = population.copy()
        current_fitnesses = fitnesses.copy()
        
        # Adaptive sigma: decrease over generations
        sigma = config['sigma']
        decay = (sigma_end / sigma_start) ** (1.0 / config['max_generations'])
        
        for gen in range(config['max_generations']):
            if time.time() > deadline - 5.0:
                break
            
            # Tournament selection
            new_population = []
            
            # Keep elites
            elite_indices = np.argsort(current_fitnesses)[-config['elite_count']:]
            for idx in elite_indices:
                new_population.append(current_population[idx].copy())
            
            # Generate offspring
            while len(new_population) < pop_size:
                # Select parents via tournament
                p1_idx = rng.choice(pop_size, config['tournament_size'])
                p2_idx = rng.choice(pop_size, config['tournament_size'])
                p1 = current_population[p1_idx[np.argmax(current_fitnesses[p1_idx])]]
                p2 = current_population[p2_idx[np.argmax(current_fitnesses[p2_idx])]]
                
                # Crossover
                child1, child2 = _ga_crossover(p1, p2, rng)
                
                # Mutation with current sigma
                child1 = _ga_mutation(child1, sigma, rng)
                child2 = _ga_mutation(child2, sigma, rng)
                
                new_population.append(child1)
                if len(new_population) < pop_size:
                    new_population.append(child2)
            
            current_population = np.array(new_population)
            
            # Evaluate fitness
            current_fitnesses = np.array([_ga_fitness(ind) for ind in current_population])
            
            # Update best
            best_idx = np.argmax(current_fitnesses)
            if current_fitnesses[best_idx] > best_area:
                best_area = current_fitnesses[best_idx]
                best_bary = current_population[best_idx].copy()
            
            # Decay sigma
            sigma *= decay
        
        # Update global population with best individuals from this GA run
        top_indices = np.argsort(current_fitnesses)[-10:]
        for idx in top_indices:
            if len(population) < pop_size:
                population = np.vstack([population, current_population[idx][np.newaxis, :]])
            else:
                # Replace worst individual
                worst_idx = np.argmin(fitnesses)
                population[worst_idx] = current_population[idx]
                fitnesses[worst_idx] = current_fitnesses[idx]
    
    # Final local optimization polish on the best solution
    if best_bary is not None and time.time() < deadline - 3.0:
        try:
            # Use Nelder-Mead for final polish
            def _objective(x_flat):
                bary = x_flat.reshape(N, 3)
                bary = _project_bary(bary)
                return -_min_area_bary(bary)
            
            res = minimize(
                _objective,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            polished_bary = _project_bary(res.x.reshape(N, 3))
            polished_area = _min_area_bary(polished_bary)
            
            if polished_area > best_area:
                best_area = polished_area
                best_bary = polished_bary.copy()
                
        except Exception:
            pass
    
    if best_bary is None:
        # Fallback: random
        r = np.random.default_rng(0)
        best_bary = r.dirichlet([1, 1, 1], size=N)
        best_bary = _project_bary(best_bary)
    
    return _bary_to_cart(best_bary)
