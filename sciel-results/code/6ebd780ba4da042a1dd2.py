"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _score(positions):
    """Compute the radius for a given set of positions."""
    if positions.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(positions)):
        return 0.0
    
    # Check if all points are inside the unit circle
    dist_from_origin = np.linalg.norm(positions, axis=1)
    if np.any(dist_from_origin > 1.0):
        return 0.0
    
    # Compute minimum distance between any pair of points
    # Using broadcasting for pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to infinity to ignore self-distances
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # The radius is half the minimum distance between centers
    r_pair = min_pair_dist / 2.0
    
    # The radius is also limited by the distance from the boundary
    r_boundary = np.min(1.0 - dist_from_origin)
    
    return min(r_pair, r_boundary)

def _hexagonal_init(n_circles, radius_scale=0.25):
    """Generate a hexagonal grid of points centered at origin."""
    points = []
    sqrt3 = np.sqrt(3)
    # Generate a grid large enough to contain n_circles
    # Approximate number of points in hexagonal packing within radius 1
    # Area = pi. Area per hex = (sqrt(3)/2) * s^2. 
    # We just generate a big grid and filter.
    for j in range(-10, 11):
        for i in range(-10, 11):
            x = (i + 0.5 * (j % 2)) * radius_scale
            y = j * radius_scale * sqrt3 / 2
            if x**2 + y**2 <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < n_circles:
        # Fallback to random if grid is too small
        angles = np.random.uniform(0, 2*np.pi, n_circles)
        radii = np.sqrt(np.random.uniform(0, 1, n_circles))
        return np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    
    # Sort by distance from origin and take the closest N
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:n_circles]]

def _construct_pso():
    """Improved Particle Swarm Optimization for circle packing."""
    np.random.seed(42)
    
    n_particles = 30
    max_iter = 3000
    w_start = 0.9
    w_end = 0.4
    c1 = 1.5
    c2 = 1.5
    
    # Initialize particles with hexagonal grid + noise
    particles = np.zeros((n_particles, N, 2))
    velocities = np.zeros((n_particles, N, 2))
    
    for i in range(n_particles):
        base = _hexagonal_init(N)
        # Add small random noise to break symmetry
        noise = np.random.normal(0, 0.01, (N, 2))
        particles[i] = base + noise
        # Ensure inside circle
        dist = np.linalg.norm(particles[i], axis=1)
        mask = dist > 0.999
        if np.any(mask):
            particles[i][mask] *= 0.999 / dist[mask][:, np.newaxis]
            
    # Evaluate initial fitness
    fitness = np.array([_score(particles[i]) for i in range(n_particles)])
    personal_best = particles.copy()
    personal_best_fitness = fitness.copy()
    
    global_best_idx = np.argmax(fitness)
    global_best = particles[global_best_idx].copy()
    global_best_fitness = fitness[global_best_idx]
    
    for iteration in range(max_iter):
        # Decay inertia weight
        w = w_start - (w_start - w_end) * (iteration / max_iter)
        
        # Update velocities and positions
        for i in range(n_particles):
            r1 = np.random.uniform(0, 1, (N, 2))
            r2 = np.random.uniform(0, 1, (N, 2))
            
            velocities[i] = (w * velocities[i] +
                           c1 * r1 * (personal_best[i] - particles[i]) +
                           c2 * r2 * (global_best - particles[i]))
            
            # Limit velocity to prevent divergence
            v_max = 0.1
            v_norm = np.linalg.norm(velocities[i], axis=1, keepdims=True)
            mask_v = v_norm > v_max
            if np.any(mask_v):
                velocities[i][mask_v] *= v_max / v_norm[mask_v]
                
            particles[i] = particles[i] + velocities[i]
            
            # Project back into unit circle if needed
            dist = np.linalg.norm(particles[i], axis=1)
            mask = dist > 1.0
            if np.any(mask):
                particles[i][mask] /= dist[mask][:, np.newaxis]
                particles[i][mask] *= 0.999  # Keep slightly inside
        
        # Evaluate new fitness
        for i in range(n_particles):
            new_fitness = _score(particles[i])
            
            # Update personal best
            if new_fitness > personal_best_fitness[i]:
                personal_best[i] = particles[i].copy()
                personal_best_fitness[i] = new_fitness
                
                # Update global best
                if new_fitness > global_best_fitness:
                    global_best = particles[i].copy()
                    global_best_fitness = new_fitness
                    
        # Early stopping if we've reached a very high score
        if global_best_fitness > 0.1258:
            break
            
    return global_best

def construct():
    """Construct the optimal packing of 50 circles in a unit circle."""
    # Get a good initial guess from PSO
    positions_pso = _construct_pso()
    
    # Refine using local optimization
    def negative_score(x_flat):
        positions = x_flat.reshape(N, 2)
        return -_score(positions)
    
    # Use L-BFGS-B for local refinement
    result = minimize(negative_score, positions_pso.flatten(), method='L-BFGS-B',
                     options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
    
    positions = result.x.reshape(N, 2)
    
    # Ensure all points are strictly inside the unit circle
    dist = np.linalg.norm(positions, axis=1)
    mask = dist > 0.9999
    if np.any(mask):
        positions[mask] *= 0.9999 / dist[mask][:, np.newaxis]
        
    return positions
# EVOLVE-BLOCK-END
