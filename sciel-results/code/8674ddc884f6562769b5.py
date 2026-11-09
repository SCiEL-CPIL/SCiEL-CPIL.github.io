import numpy as np
from scipy.spatial import ConvexHull
import time

def heilbronn_convex13() -> np.ndarray:
    """
    Solve the Heilbronn problem for 13 points using Differential Evolution.
    """
    
    # Precompute triangle indices for efficient area calculation
    n = 13
    tri_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    
    def evaluate(points):
        """
        Calculate the score: A_min / A_hull.
        Returns 0 if invalid.
        """
        if not np.all(np.isfinite(points)):
            return 0.0
        
        # Check for convex hull validity
        try:
            hull = ConvexHull(points)
            a_hull = hull.volume
            if a_hull <= 0 or not np.isfinite(a_hull):
                return 0.0
        except Exception:
            return 0.0
            
        # Calculate min triangle area
        # points[indices] gives shape (286, 3, 2)
        p1 = points[tri_indices[:, 0]]
        p2 = points[tri_indices[:, 1]]
        p3 = points[tri_indices[:, 2]]
        
        # Area = 0.5 * | (x2-x1)(y3-y1) - (y2-y1)(x3-x1) |
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        a_min = np.min(areas)
        
        if a_min <= 0:
            return 0.0
            
        return a_min / a_hull

    # Differential Evolution Parameters
    pop_size = 40
    F = 0.8  # Mutation factor
    CR = 0.9  # Crossover probability
    bounds_low = -2.0
    bounds_high = 2.0
    
    # Initialize Population
    # Mix of random and structured points
    rng = np.random.default_rng(42)
    
    # Generate structured seeds
    structured_seeds = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    r13 = 1.0
    p13 = np.column_stack([r13 * np.cos(angles), r13 * np.sin(angles)])
    structured_seeds.append(p13)
    
    # 2. Regular 12-gon + center
    angles12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    r12 = 1.0
    p12 = np.column_stack([r12 * np.cos(angles12), r12 * np.sin(angles12)])
    p12c = np.vstack([p12, [[0.0, 0.0]]])
    structured_seeds.append(p12c)
    
    # 3. Hexagonal lattice subset
    sqrt3 = np.sqrt(3)
    s = 1.0
    p_hex = np.array([
        [0, 0], [s, 0], [-s, 0], [0, s*sqrt3/2], [0, -s*sqrt3/2],
        [s/2, s*sqrt3/2], [-s/2, s*sqrt3/2], [s/2, -s*sqrt3/2], [-s/2, -s*sqrt3/2],
        [1.5*s, s*sqrt3/2], [-1.5*s, s*sqrt3/2], [1.5*s, -s*sqrt3/2], [-1.5*s, -s*sqrt3/2]
    ], dtype=float)
    structured_seeds.append(p_hex)
    
    # 4. Two concentric hexagons
    p_hex2 = np.array([[0.0, 0.0]])
    for k in range(6):
        angle = np.pi / 3 * k
        p_hex2 = np.vstack([p_hex2, [0.8 * np.cos(angle), 0.8 * np.sin(angle)]])
    for k in range(6):
        angle = np.pi / 3 * k + np.pi / 6
        p_hex2 = np.vstack([p_hex2, [1.2 * np.cos(angle), 1.2 * np.sin(angle)]])
    structured_seeds.append(p_hex2)

    # Create initial population
    population = []
    for seed in structured_seeds:
        population.append(seed.copy())
        
    # Fill rest with random points in a circle of radius 2
    while len(population) < pop_size:
        angles = rng.uniform(0, 2 * np.pi, 13)
        radii = rng.uniform(0.5, 2.0, 13)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        population.append(pts)
        
    population = np.array(population)
    
    # Evaluate initial population
    fitness = np.array([evaluate(pop) for pop in population])
    best_idx = np.argmax(fitness)
    best_points = population[best_idx].copy()
    best_score = fitness[best_idx]
    
    start_time = time.time()
    max_time = 550.0  # Leave some buffer
    
    generation = 0
    while time.time() - start_time < max_time:
        generation += 1
        new_population = []
        new_fitness = []
        
        for i in range(pop_size):
            # Select 3 distinct random indices
            r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
            
            # Mutation
            mutant = population[r1] + F * (population[r2] - population[r3])
            
            # Clip to bounds
            mutant = np.clip(mutant, bounds_low, bounds_high)
            
            # Crossover
            trial = population[i].copy()
            cross_mask = rng.random((13, 2)) < CR
            # Ensure at least one gene is taken from mutant
            if not np.any(cross_mask):
                idx = rng.integers(0, 13)
                cross_mask[idx, 0] = True
            
            trial[cross_mask] = mutant[cross_mask]
            
            # Evaluate trial
            trial_score = evaluate(trial)
            
            if trial_score >= fitness[i]:
                new_population.append(trial)
                new_fitness.append(trial_score)
            else:
                new_population.append(population[i])
                new_fitness.append(fitness[i])
                
        population = np.array(new_population)
        fitness = np.array(new_fitness)
        
        # Update best
        current_best_idx = np.argmax(fitness)
        if fitness[current_best_idx] > best_score:
            best_score = fitness[current_best_idx]
            best_points = population[current_best_idx].copy()
            
        # Adaptive F: reduce F slightly over time to refine
        if generation % 50 == 0:
            F = max(0.1, F * 0.9)
            
    return best_points
