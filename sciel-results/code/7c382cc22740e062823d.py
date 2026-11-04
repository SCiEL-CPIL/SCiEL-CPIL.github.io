import numpy as np
import time
from scipy.optimize import minimize

N = 11

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples (i, j, k) with i < j < k
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_IDX_I = _IDX_TRIPLES[:, 0]
_IDX_J = _IDX_TRIPLES[:, 1]
_IDX_K = _IDX_TRIPLES[:, 2]

def _min_triangle_area_vectorized(points):
    """
    Compute the smallest area among all triangles formed by triples of points, vectorized.
    points: (N, 2) array
    Returns: float min area
    """
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    if len(areas) == 0:
        return 0.0
    return np.min(areas)

def _bary_to_cart(bary):
    """
    Convert barycentric coordinates (N, 3) to Cartesian (N, 2).
    Each row must sum to 1 and be non-negative.
    """
    bary = np.clip(bary, 0.0, None)
    # Normalize to ensure sum is 1
    sums = bary.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    bary = bary / sums
    
    x = bary[:, 1] * V1[0] + bary[:, 2] * V2[0]
    y = bary[:, 1] * V1[1] + bary[:, 2] * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(points):
    """
    Convert Cartesian coordinates (N, 2) to barycentric coordinates (N, 3).
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # l2 = 2*y / sqrt(3)
    l2 = 2.0 * y / np.sqrt(3)
    # l1 = x - y/sqrt(3)
    l1 = x - y / np.sqrt(3)
    # l0 = 1 - l1 - l2
    l0 = 1.0 - l1 - l2
    
    bary = np.column_stack([l0, l1, l2])
    bary = np.clip(bary, 0.0, None)
    sums = bary.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    bary = bary / sums
    
    return bary

def _fitness(bary):
    """
    Compute fitness: minimize triangle area, penalize boundary violations.
    bary: (N, 3) array of barycentric coordinates
    Returns: float (lower is better)
    """
    points = _bary_to_cart(bary)
    min_area = _min_triangle_area_vectorized(points)
    
    # Penalty for points outside the triangle (shouldn't happen after clipping, but just in case)
    # Check barycentric coordinates are valid
    penalty = 0.0
    for i in range(N):
        b = bary[i]
        if b[0] < -1e-6 or b[1] < -1e-6 or b[2] < -1e-6:
            penalty += abs(min(b[0], b[1], b[2])) * 100.0
    
    return -min_area + penalty

def _crossover(parent1, parent2, rng):
    """
    Uniform crossover: for each gene (barycentric coord), randomly pick from parent1 or parent2.
    """
    mask = rng.random(parent1.shape) < 0.5
    child = np.where(mask, parent1, parent2)
    return child

def _mutation(individual, rng, rate=0.1, scale=0.1):
    """
    Gaussian mutation: perturb each gene with probability `rate`.
    """
    child = individual.copy()
    mask = rng.random(child.shape) < rate
    if np.any(mask):
        noise = rng.normal(0, scale, size=child.shape)
        child[mask] += noise[mask]
    # Clip to valid range
    child = np.clip(child, -0.1, 1.1)
    # Renormalize each row to sum to 1
    sums = child.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    child = child / sums
    return child

def _tournament_selection(population, fitnesses, k=3, rng=None):
    """
    Tournament selection: pick k random individuals, return the best.
    """
    if rng is None:
        rng = np.random.RandomState()
    idx = rng.choice(len(population), size=k, replace=False)
    best_idx = idx[np.argmin(fitnesses[idx])]
    return population[best_idx].copy()

def _ga_optimize(initial_pop, rng, max_generations=200, pop_size=50, elite_size=5):
    """
    Genetic algorithm to optimize the point configuration.
    """
    start_time = time.time()
    deadline = start_time + 120.0
    
    def time_left():
        return deadline - time.time()
    
    # Initialize population
    population = np.array(initial_pop[:pop_size])
    while len(population) < pop_size:
        # Add random individuals
        bary = rng.dirichlet(np.ones(N), size=pop_size - len(population))
        population = np.vstack([population, bary])
    
    # Evaluate initial population
    fitnesses = np.array([_fitness(ind) for ind in population])
    
    best_individual = population[np.argmin(fitnesses)].copy()
    best_fitness = np.min(fitnesses)
    
    for gen in range(max_generations):
        if time_left() < 10.0:
            break
        
        # Sort by fitness
        sorted_idx = np.argsort(fitnesses)
        sorted_pop = population[sorted_idx]
        sorted_fit = fitnesses[sorted_idx]
        
        # Create new population
        new_population = []
        
        # Elitism: keep best individuals
        for i in range(min(elite_size, len(sorted_pop))):
            new_population.append(sorted_pop[i].copy())
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Selection
            parent1 = _tournament_selection(population, fitnesses, k=3, rng=rng)
            parent2 = _tournament_selection(population, fitnesses, k=3, rng=rng)
            
            # Crossover
            if rng.random() < 0.7:  # 70% crossover rate
                child = _crossover(parent1, parent2, rng)
            else:
                child = parent1.copy()
            
            # Mutation
            child = _mutation(child, rng, rate=0.1, scale=0.05)
            
            new_population.append(child)
        
        population = np.array(new_population)
        fitnesses = np.array([_fitness(ind) for ind in population])
        
        # Update best
        gen_best_idx = np.argmin(fitnesses)
        if fitnesses[gen_best_idx] < best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_individual = population[gen_best_idx].copy()
    
    return best_individual, best_fitness

def _generate_fps_initialization(rng, n_points=N, n_candidates=500):
    """
    Generate an initial configuration using Greedy Farthest Point Sampling.
    Returns barycentric coordinates.
    """
    points = np.zeros((n_points, 2))
    
    # Start with centroid
    points[0] = (V0 + V1 + V2) / 3.0
    
    for k in range(1, n_points):
        r1 = rng.uniform(0, 1, n_candidates)
        r2 = rng.uniform(0, 1, n_candidates)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        
        candidates = np.column_stack([
            r1 * V1[0] + r2 * V2[0],
            r1 * V1[1] + r2 * V2[1]
        ])
        
        diffs = candidates[:, np.newaxis, :] - points[np.newaxis, :k, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = np.min(dists, axis=1)
        
        best_idx = np.argmax(min_dists)
        points[k] = candidates[best_idx]
        
    return _cart_to_bary(points)

def _structured_inits(rng):
    """
    Generate structured initializations based on grid or symmetric patterns.
    Returns list of barycentric coordinate arrays.
    """
    inits = []
    
    # Grid-like patterns
    for grid_n in [3, 4, 5]:
        # Generate points on a triangular grid
        points = []
        for i in range(grid_n):
            for j in range(grid_n - i):
                # Barycentric coordinates
                l0 = 1 - (i + j) / (grid_n - 1) if grid_n > 1 else 1
                l1 = i / (grid_n - 1) if grid_n > 1 else 0
                l2 = j / (grid_n - 1) if grid_n > 1 else 0
                
                pt = l0 * V0 + l1 * V1 + l2 * V2
                points.append(pt)
        
        # If we have more than N points, randomly select N
        points = np.array(points)
        if len(points) >= N:
            idx = rng.choice(len(points), N, replace=False)
            inits.append(_cart_to_bary(points[idx]))
        elif len(points) > 0:
            # Pad with random points if needed
            while len(points) < N:
                r1 = rng.uniform(0, 1)
                r2 = rng.uniform(0, 1)
                if r1 + r2 > 1:
                    r1 = 1 - r1
                    r2 = 1 - r2
                pt = r1 * V1 + r2 * V2
                points = np.vstack([points, pt])
            inits.append(_cart_to_bary(points))
    
    # Symmetric patterns: 1 center + 10 around
    center = (V0 + V1 + V2) / 3.0
    pts = [center]
    for angle in np.linspace(0, 2*np.pi, 5, endpoint=False):
        r = 0.3
        pt = center + r * np.array([np.cos(angle), np.sin(angle)])
        pts.append(pt)
    for angle in np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5:
        r = 0.6
        pt = center + r * np.array([np.cos(angle), np.sin(angle)])
        pts.append(pt)
    
    inits.append(_cart_to_bary(np.array(pts)))
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def time_left():
        return deadline - time.time()
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initial populations
    initial_bary = []
    
    # FPS-based inits
    for i in range(10):
        if time_left() < 60.0:
            break
        bary = _generate_fps_initialization(rng)
        initial_bary.append(bary)
    
    # Structured inits
    for bary in _structured_inits(rng):
        if time_left() < 60.0:
            break
        initial_bary.append(bary)
    
    # Random inits
    for i in range(15):
        if time_left() < 60.0:
            break
        bary = rng.dirichlet(np.ones(N))
        initial_bary.append(bary)
    
    # Run genetic algorithm with multiple restarts
    best_bary = None
    best_score = np.inf
    
    # Use multiple GA runs with different subsets of initial population
    n_ga_runs = 3
    if time_left() > 90.0:
        n_ga_runs = 4
    elif time_left() > 60.0:
        n_ga_runs = 2
    else:
        n_ga_runs = 1
    
    for run in range(n_ga_runs):
        if time_left() < 30.0:
            break
        
        # Select a subset of initial population for this run
        if len(initial_bary) > 20:
            idx = rng.choice(len(initial_bary), size=20, replace=False)
            pop_subset = [initial_bary[i] for i in idx]
        else:
            pop_subset = initial_bary
        
        try:
            best_ind, best_fit = _ga_optimize(
                pop_subset, rng,
                max_generations=150,
                pop_size=40,
                elite_size=4
            )
            
            if best_fit < best_score:
                best_score = best_fit
                best_bary = best_ind.copy()
        except Exception:
            pass
    
    # If no good result from GA, use best from initial population
    if best_bary is None:
        best_bary = initial_bary[0]
        best_score = _fitness(best_bary)
    
    # Final local refinement using Nelder-Mead on the best solution
    if time_left() > 20.0:
        try:
            def _nm_objective(bary_flat):
                bary = bary_flat.reshape(N, 3)
                return _fitness(bary)
            
            bary_flat = best_bary.flatten()
            res = minimize(
                _nm_objective, bary_flat, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            res_bary = res.x.reshape(N, 3)
            res_bary = np.clip(res_bary, 0.0, None)
            sums = res_bary.sum(axis=1, keepdims=True)
            sums[sums == 0] = 1.0
            res_bary = res_bary / sums
            
            res_score = _fitness(res_bary)
            if res_score < best_score:
                best_score = res_score
                best_bary = res_bary
        except Exception:
            pass
    
    # Convert to Cartesian and return
    points = _bary_to_cart(best_bary)
    
    # Final safety check: ensure all points are inside the triangle
    def _clip_to_triangle(pts):
        for i in range(len(pts)):
            x, y = pts[i]
            l2 = 2 * y / np.sqrt(3)
            l1 = x - y / np.sqrt(3)
            l0 = 1 - l1 - l2
            
            if l0 < 0 or l1 < 0 or l2 < 0:
                l0 = max(0, l0)
                l1 = max(0, l1)
                l2 = max(0, l2)
                s = l0 + l1 + l2
                if s > 0:
                    l0 /= s
                    l1 /= s
                    l2 /= s
                else:
                    l0 = 1
                    l1 = 0
                    l2 = 0
                
                pts[i] = l0 * V0 + l1 * V1 + l2 * V2
        
        return pts
    
    points = _clip_to_triangle(points)
    
    return np.array(points)
