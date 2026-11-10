import numpy as np
import time
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 620  # 620s to leave 10s margin
    
    # Theoretical context:
    # We want to maximize (d_min / d_max)^2.
    # This is equivalent to minimizing d_max while keeping d_min as large as possible, 
    # or maximizing the "packing" efficiency within the convex hull defined by the farthest points.
    
    def get_score(pts):
        # Calculate squared pairwise distances
        dists = pdist(pts, metric='sqeuclidean')
        if len(dists) == 0:
            return 0.0
        d_min_sq = np.min(dists)
        d_max_sq = np.max(dists)
        
        if d_max_sq < 1e-12:
            return 0.0
        
        # Metric is (d_min / d_max)^2 = d_min_sq / d_max_sq
        return d_min_sq / d_max_sq

    # 1. Construct a high-quality initial geometric seed
    # Start with a regular Icosahedron (12 vertices) which has high symmetry and good min/max ratio locally.
    # Then add 2 points. A good guess for the 2 extra points in a 14-point problem in 3D
    # might be near the center or specific high-symmetry positions, but random perturbation
    # around the icosahedron is a safe bet for the DE population initialization.
    
    phi = (1 + np.sqrt(5)) / 2
    # Standard Icosahedron vertices
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    # Normalize to unit sphere
    norms = np.linalg.norm(ico_verts, axis=1, keepdims=True)
    ico_verts = ico_verts / norms
    
    # For the 14th and 15th points (indices 12, 13), we can try placing them at the origin 
    # or slightly off-center. However, for a diverse population, we will generate the 
    # initial population by taking the icosahedron and adding random noise to all points,
    # but ensuring the "base" structure is preserved for many individuals.
    
    # Better Initial Strategy:
    # Individual 0: Perfect Icosahedron + 2 points at Origin (0,0,0) -> Wait, coincident points give 0.
    # So 2 points near origin but distinct? Or 2 points on the sphere?
    # Actually, adding points to a sphere usually decreases d_min. 
    # The optimal configuration for 14 points is likely NOT a subset of a sphere.
    # It might be a 3D lattice-like structure or a distorted polytope.
    
    # Let's use a "Cluster" initialization strategy for the population.
    # Pop size: 20
    pop_size = 20
    population = np.zeros((pop_size, n, d))
    
    # Initialize with a mix of:
    # 1. Icosahedron + 2 random points on sphere (radius 1)
    # 2. Icosahedron + 2 random points near center
    # 3. Random Gaussian
    # 4. Random Uniform in cube
    
    for i in range(pop_size):
        if i < 5:
            # Icosahedron + 2 random on sphere
            p = ico_verts.copy()
            extra1 = np.random.randn(d)
            extra1 = extra1 / np.linalg.norm(extra1)
            extra2 = np.random.randn(d)
            extra2 = extra2 / np.linalg.norm(extra2)
            p = np.vstack([p, [extra1], [extra2]])
            # Add small noise to break symmetry and allow DE to move
            p += np.random.randn(n, d) * 0.01
        elif i < 10:
            # Icosahedron + 2 random near center
            p = ico_verts.copy()
            extra1 = np.random.randn(d) * 0.2
            extra2 = np.random.randn(d) * 0.2
            p = np.vstack([p, [extra1], [extra2]])
            p += np.random.randn(n, d) * 0.01
        elif i < 15:
            # Random Gaussian
            p = np.random.randn(n, d)
        else:
            # Random Uniform
            p = np.random.rand(n, d) * 2 - 1
            
        population[i] = p

    # Differential Evolution Parameters
    F = 0.8  # Scale factor
    CR = 0.9  # Crossover rate
    max_iter = 5000
    
    best_individual = population[0].copy()
    best_score = get_score(best_individual)
    
    # Track population scores
    pop_scores = np.array([get_score(population[i]) for i in range(pop_size)])
    
    # Find initial best
    best_idx = np.argmax(pop_scores)
    best_individual = population[best_idx].copy()
    best_score = pop_scores[best_idx]
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        for i in range(pop_size):
            if time.time() > deadline:
                break
                
            # Select 3 distinct random individuals
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            
            # Differential Evolution Mutation
            # v = x[r1] + F * (x[r2] - x[r3])
            v = population[r1] + F * (population[r2] - population[r3])
            
            # Crossover
            # u = trial vector
            u = np.copy(population[i])
            # Random crossover per coordinate (flattened)
            # We operate on the flattened vector of size n*d
            v_flat = v.flatten()
            x_flat = population[i].flatten()
            u_flat = x_flat.copy()
            
            # Choose a random starting index for crossover to ensure at least one gene is from mutant
            j_rand = np.random.randint(0, n * d)
            
            for j in range(n * d):
                if (j == j_rand) or (np.random.rand() < CR):
                    u_flat[j] = v_flat[j]
            
            u = u_flat.reshape(n, d)
            
            # Evaluate
            u_score = get_score(u)
            
            # Selection
            if u_score > pop_scores[i]:
                population[i] = u
                pop_scores[i] = u_score
                
                if u_score > best_score:
                    best_score = u_score
                    best_individual = u.copy()
        
        # Elitism is handled by standard DE selection (trial vs target).
        # However, we might want to ensure the global best is preserved if it gets overwritten?
        # Standard DE replaces population[i] only if trial is better. So the global best in population
        # is preserved unless a better one is found. But if the global best is at index k, and we
        # mutate index k, and the mutant is worse, population[k] stays the same. So it's safe.
        
        # Optional: Inject the global best into the population to prevent it from being lost?
        # Actually, if the global best is at index k, it is only replaced if a BETTER solution is found.
        # So it is never lost.
        
        # Early stopping if score is very high?
        if best_score > 0.8:
             break

    return best_individual
