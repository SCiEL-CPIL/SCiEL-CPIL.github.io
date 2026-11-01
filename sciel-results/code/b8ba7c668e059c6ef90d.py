import numpy as np
from scipy.optimize import minimize
import time

N = 55

def lennard_jones_energy(positions):
    """Calculate the total Lennard-Jones energy of the cluster."""
    # positions: (N, 3) array
    # Calculate all pair distances
    # Use broadcasting for speed
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    # Avoid division by zero (self-interaction)
    np.fill_diagonal(dist_sq, np.inf)
    
    # LJ potential: 4 * (r^-12 - r^-6)
    # r^-6 = 1 / (r^2)^3
    r_inv6 = 1.0 / (dist_sq ** 3)
    r_inv12 = r_inv6 ** 2
    
    # Sum over all pairs (each pair counted twice in the matrix, so divide by 2)
    # The formula in the prompt is 4 * sum(r^-12 - r^-6) over all pairs.
    # Usually "over all pairs" means i < j.
    # If we sum the upper triangle, we don't divide by 2.
    # Let's sum the upper triangle to be safe.
    iu = np.triu_indices(N, k=1)
    energy = 4.0 * np.sum(r_inv12[iu] - r_inv6[iu])
    
    return energy

def construct():
    # Generate 13-atom icosahedron
    phi = (1 + np.sqrt(5)) / 2.0
    # Vertices of regular icosahedron
    # (0, 1, phi), (0, -1, phi), ... etc
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    # Normalize to have edge length or specific radius
    # The distance between (0,1,phi) and (1,phi,0) is sqrt(1 + (phi-1)^2 + phi^2)
    # phi^2 = phi + 1
    # dist^2 = 1 + (1/phi)^2 + phi^2 = 1 + (2-phi) + (1+phi) = 4?
    # Let's just normalize the vectors to a unit sphere and then scale.
    norms = np.linalg.norm(v, axis=1)
    v = v / norms[:, np.newaxis]
    
    # Scale inner shell. 
    # Optimal inner shell radius for 55-atom cluster is roughly 1.1-1.2
    # Let's start with radius 1.1
    inner_radius = 1.1
    inner_positions = v * inner_radius
    center = np.array([[0.0, 0.0, 0.0]])
    
    # Generate 42 outer atoms
    # Random points on a sphere of radius ~2.2
    outer_radius = 2.2
    # Use random spherical coordinates
    z = np.random.uniform(-1, 1, 42)
    theta = np.random.uniform(0, 2 * np.pi, 42)
    x = np.sqrt(1 - z**2) * np.cos(theta)
    y = np.sqrt(1 - z**2) * np.sin(theta)
    outer_positions = np.column_stack((x, y, z)) * outer_radius
    
    # Combine
    positions = np.vstack((center, inner_positions, outer_positions))
    
    # Ensure minimum distance constraint? 
    # The prompt says "Atoms must be at least 0.5 apart".
    # L-J potential goes to +inf as r -> 0, so minimization will push them apart.
    # However, if they start too close, L-BFGS might fail or take long.
    # We can check and perturb if necessary, but random start on sphere is usually fine.
    
    start_time = time.time()
    best_energy = lennard_jones_energy(positions)
    best_positions = positions.copy()
    
    # Basin Hopping
    # We have 120s. L-BFGS for 55 atoms is fast (<0.1s per iteration).
    # We can afford ~100-200 iterations.
    max_iterations = 200
    kick_std = 0.5
    
    for i in range(max_iterations):
        if time.time() - start_time > 110:
            break
            
        # Local minimization
        res = minimize(lennard_jones_energy, positions, method='L-BFGS-B', options={'maxiter': 200})
        positions = res.x
        energy = res.fun
        
        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()
            
        # Kick
        # Random displacement
        kick = np.random.normal(0, kick_std, size=positions.shape)
        positions = positions + kick
        
        # Optional: Reduce kick size over time
        kick_std *= 0.99
        
    return best_positions
