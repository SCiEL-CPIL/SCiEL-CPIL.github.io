import numpy as np

N = 50

def construct():
    # 1. Initialization: Hexagonal lattice
    # We want to fit N=50 points.
    # Estimated optimal radius is ~0.1258.
    # Let's start with a lattice spacing that allows for this.
    
    # Generate a large grid of hexagonal points
    # Side length of hexagon packing ~ 2*r
    # Let's try a range of spacings and pick the one that fits N points best initially
    # Or just start with a reasonable spacing and relax.
    
    # Initial spacing
    spacing = 0.25 # Initial guess, will be optimized
    points = []
    k = 10
    for i in range(-k, k+1):
        for j in range(-k, k+1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) < 1.0:
                points.append([x, y])
    
    points = np.array(points)
    
    # Sort by distance to center and take N closest
    dists = np.linalg.norm(points, axis=1)
    points = points[np.argsort(dists)[:N]]
    
    # 2. Relaxation
    # We want to maximize the minimum distance.
    # We will simulate repulsive forces.
    
    # Parameters
    steps = 10000
    dt = 0.01
    # We don't know the optimal r yet, so we let the system find a configuration
    # that maximizes the minimum pairwise distance and boundary distance.
    # Objective: Maximize min( min_{i<j} |pi-pj|, min_i (1-|pi|) )
    
    # To do this efficiently, we can use a "soft" repulsion that becomes very strong
    # when distance < current_best_min_dist.
    
    # Initial estimate of best radius
    current_min_dist = 1e9
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(points[i] - points[j])
            if d < current_min_dist:
                current_min_dist = d
    for i in range(N):
        d = 1.0 - np.linalg.norm(points[i])
        if d < current_min_dist:
            current_min_dist = d
    
    # This is the initial 'radius' (half of min dist for pairs, or dist to wall)
    # Note: The score is min( min_pair_dist/2, min_wall_dist ).
    # So let's track R = min( min_pair_dist/2, min_wall_dist ).
    
    best_R = 0.0
    current_R = min(current_min_dist / 2.0, min(1.0 - np.linalg.norm(points, axis=1)))
    
    # Simulated Annealing / Langevin
    T = 0.05 # Initial temperature
    cool_rate = 0.999
    
    for step in range(steps):
        forces = np.zeros_like(points)
        
        # Pairwise repulsion
        # Calculate all pairwise distances
        # To save memory/time for N=50, we can do it in a loop or vectorize carefully.
        # N=50 is small, so O(N^2) is fine.
        
        for i in range(N):
            for j in range(i+1, N):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    # Random push
                    forces[i] += np.random.normal(size=2) * 0.1
                    forces[j] -= np.random.normal(size=2) * 0.1
                else:
                    # We want to push them apart if they are too close.
                    # "Too close" is defined relative to the current best R.
                    # If dist < 2 * current_R, they are overlapping in the "score" sense.
                    # Let's use a soft potential that pushes if dist < 2 * current_R.
                    if dist < 2.0 * current_R:
                        force_mag = (2.0 * current_R - dist) / dist
                        # Scale force
                        forces[i] += force_mag * diff
                        forces[j] -= force_mag * diff
        
        # Boundary repulsion
        for i in range(N):
            r_i = np.linalg.norm(points[i])
            if r_i > 1.0 - current_R:
                # Push inward
                force_mag = (r_i - (1.0 - current_R)) / (r_i + 1e-9)
                forces[i] -= force_mag * points[i]
        
        # Add noise
        noise = np.random.normal(scale=T, size=(N, 2))
        
        # Update positions
        points += dt * forces + noise
        
        # Keep points inside container (hard constraint to avoid divergence)
        # If a point goes outside, pull it back
        r_norms = np.linalg.norm(points, axis=1)
        mask = r_norms > 1.0
        if np.any(mask):
            points[mask] *= (1.0 / r_norms[mask])[:, None]
            
        # Recalculate current_R
        min_pair = 1e9
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points[i] - points[j])
                if d < min_pair:
                    min_pair = d
        min_wall = np.min(1.0 - np.linalg.norm(points, axis=1))
        
        new_R = min(min_pair / 2.0, min_wall)
        
        if new_R > current_R:
            current_R = new_R
            best_R = max(best_R, current_R)
            
        T *= cool_rate
        
    return points

# EVOLVE-BLOCK-END
